import csv
import json
from pathlib import Path

import numpy as np
import albumentations as A
from PIL import Image, ImageDraw, ImageOps

DRIVE = Path("/content/drive/MyDrive")
LOGO = DRIVE / "logo"
POOL = DRIVE / "logo_pool"

BACKGROUND, LEAF, LESION = 0, 1, 2
CLASS_NAMES = ["background", "leaf", "GLS_lesion"]
NUM_CLASSES = 3
IMGSZ = 800
SEED = int(__import__("os").environ.get("GLS_SEED", 1))
SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

DEVICE = "cuda"


def fold_stems(fold, split):
    f = LOGO / "unet" / f"fold_{fold}_{split}.txt"
    return [s for s in f.read_text().split() if s.strip()]


def image_path(stem):
    hits = sorted(POOL.glob(f"images/{stem}.*"))
    if not hits:
        raise FileNotFoundError(f"no image for stem {stem}")
    return hits[0]


def confusion(gt, pred, n=NUM_CLASSES):
    k = gt.astype(np.int64) * n + pred.astype(np.int64)
    return np.bincount(k.ravel(), minlength=n * n).reshape(n, n)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    pt, tt = cm.sum(0).astype(np.float64), cm.sum(1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(pt + tt > 0, 2 * tp / (pt + tt), np.nan)
        iou = np.where(tp + (pt - tp) + (tt - tp) > 0,
                       tp / (tp + (pt - tp) + (tt - tp)), np.nan)
        prec = np.where(pt > 0, tp / pt, np.nan)
        rec = np.where(tt > 0, tp / tt, np.nan)
    return dice, iou, prec, rec


def severity(sem):
    les, leaf = int((sem == LESION).sum()), int((sem == LEAF).sum())
    return 100.0 * les / (les + leaf) if les + leaf else float("nan")


def gt_semantic_from_yolo(stem, h, w):
    sem = Image.new("L", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(sem)
    p = POOL / "labels" / f"{stem}.txt"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            s = line.split()
            if len(s) < 7:
                continue
            try:
                cid = int(float(s[0]))
                co = [float(x) for x in s[1:]]
            except ValueError:
                continue
            if len(co) % 2 == 0 and len(co) >= 6:
                rows.append((cid + 1, co))
    for target in (LEAF, LESION):
        for cid, co in rows:
            if cid != target:
                continue
            pts = [(co[i] * w, co[i + 1] * h) for i in range(0, len(co), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=target)
    return np.array(sem, dtype=np.uint8)


def write_results(out_dir, fold, model_name, cm, per_image_cms, sev_pairs,
                  names, threshold=None, extra=None):
    dice, iou, prec, rec = metrics_from_cm(cm)
    pairs = np.array(sev_pairs, dtype=float)
    ok = np.isfinite(pairs).all(axis=1)
    if ok.sum() > 1:
        t, p = pairs[ok, 0], pairs[ok, 1]
        err = p - t
        sev = {"mae_pp": float(np.mean(np.abs(err))),
               "bias_pp": float(np.mean(err)),
               "pearson_r": float(np.corrcoef(t, p)[0, 1]),
               "n_images": int(ok.sum()),
               "mean_true_pct": float(t.mean()),
               "mean_pred_pct": float(p.mean())}
    else:
        sev = {}

    res = {"fold": fold, "model": model_name, "split": "test (held-out group)",
           "n_images": len(names),
           "confusion_matrix": cm.tolist(),
           "dice": dict(zip(CLASS_NAMES, dice.tolist())),
           "iou": dict(zip(CLASS_NAMES, iou.tolist())),
           "precision": dict(zip(CLASS_NAMES, prec.tolist())),
           "recall": dict(zip(CLASS_NAMES, rec.tolist())),
           "mean_dice": float(np.nanmean(dice)),
           "mean_iou": float(np.nanmean(iou)),
           "severity": sev}
    if threshold is not None:
        res["score_threshold"] = threshold
    if extra:
        res.update(extra)

    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))

    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "fold", "model", "mean_dice", "mean_iou"]
                   + [f"dice_{c}" for c in CLASS_NAMES]
                   + [f"iou_{c}" for c in CLASS_NAMES]
                   + ["true_lesion_px", "pred_lesion_px", "true_leaf_px",
                      "has_lesion",
                      "severity_true_pct", "severity_pred_pct",
                      "severity_error_pp"])
        for nm, c, (gs, ps) in zip(names, per_image_cms, sev_pairs):
            c = np.asarray(c)
            d_i, i_i, _, _ = metrics_from_cm(c)
            true_les = int(c[LESION].sum())
            pred_les = int(c[:, LESION].sum())
            true_leaf = int(c[LEAF].sum())
            e = ps - gs if np.isfinite([gs, ps]).all() else float("nan")

            def fm(v):
                return "" if (v is None or (isinstance(v, float) and v != v)) \
                    else f"{v:.6f}"

            w.writerow([nm, fold, model_name,
                        fm(float(np.nanmean(d_i))), fm(float(np.nanmean(i_i)))]
                       + [fm(x) for x in d_i] + [fm(x) for x in i_i]
                       + [true_les, pred_les, true_leaf,
                          int(true_les > 0), fm(gs), fm(ps), fm(e)])

    print(f"\n  held-out group '{fold}' — {model_name}")
    print(f"    lesion Dice {dice[LESION]:.4f}   mean Dice {np.nanmean(dice):.4f}")
    if sev:
        print(f"    severity MAE {sev['mae_pp']:.2f} pp   "
              f"bias {sev['bias_pp']:+.2f} pp   r {sev['pearson_r']:.4f}")
    r = cm[LESION].sum()
    if r:
        print(f"    of true lesion px: {100*cm[LESION,LESION]/r:.1f}% correct, "
              f"{100*cm[LESION,LEAF]/r:.1f}% -> leaf, "
              f"{100*cm[LESION,BACKGROUND]/r:.1f}% -> background")
    n_nolesion = sum(1 for c in per_image_cms if np.asarray(c)[LESION].sum() == 0)
    if n_nolesion:
        print(f"    note: {n_nolesion} image(s) contain no true lesion — "
              f"their lesion Dice is degenerate (0 unless the model also "
              f"predicts none). Filter on has_lesion when analysing.")
    return {"lesion_dice": float(dice[LESION]),
            "mean_dice": float(np.nanmean(dice)),
            "severity_mae_pp": sev.get("mae_pp"),
            "threshold": threshold}


def sweep_and_pick(sweep):
    print(f"\n    {'thr':>6}{'lesion Dice':>13}")
    best, best_t = -1.0, None
    for t in sorted(sweep):
        v = sweep[t]
        print(f"    {t:>6.2f}{v:>13.4f}")
        if np.isfinite(v) and v > best:
            best, best_t = v, t
    print(f"    -> threshold {best_t:.2f} selected on val")
    return best_t


def _history_writer(out_dir, header):
    f = open(out_dir / "history.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(header)
    f.flush()

    def row(vals):
        w.writerow(vals)
        f.flush()

    return row, f.close


def _yolo_pred_semantic(res, h, w, thr):
    sem = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None or len(res.boxes) == 0:
        return sem
    masks = res.masks.data.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int) + 1
    for target in (LEAF, LESION):
        for i in range(len(scores)):
            if scores[i] < thr or classes[i] != target:
                continue
            m = masks[i]
            if m.shape != (h, w):
                m = np.array(Image.fromarray((m > 0.5).astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            else:
                m = m > 0.5
            sem[m] = target
    return sem


def _yolo_eval_split(model, stems, thresholds):
    cms = {t: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for t in thresholds}
    sev = {t: [] for t in thresholds}
    pid = {t: [] for t in thresholds}
    names = []
    for stem in stems:
        ip = image_path(stem)
        with Image.open(ip) as im:
            w, h = im.size
        gt = gt_semantic_from_yolo(stem, h, w)
        gsev = severity(gt)
        names.append(ip.name)
        r = model.predict(str(ip), imgsz=IMGSZ, conf=min(thresholds), iou=0.7,
                          max_det=300, retina_masks=True, verbose=False)[0]
        for t in thresholds:
            pr = _yolo_pred_semantic(r, h, w, t)
            c = confusion(gt, pr)
            cms[t] += c
            pid[t].append(c)
            sev[t].append((gsev, severity(pr)))
    return cms, sev, pid, names


def evaluate_yolo(fold, weights, out_dir):
    from ultralytics import YOLO
    model = YOLO(str(weights))

    print("\n  selecting threshold on val...")
    val_stems = fold_stems(fold, "val")
    cms, _, _, _ = _yolo_eval_split(model, val_stems, SCORE_THRESHOLDS)
    sweep = {t: metrics_from_cm(cms[t])[0][LESION] for t in SCORE_THRESHOLDS}
    thr = sweep_and_pick(sweep)

    print(f"\n  evaluating held-out group '{fold}'...")
    test_stems = fold_stems(fold, "test")
    cms, sev, pid, names = _yolo_eval_split(model, test_stems, [thr])
    return write_results(out_dir, fold, "YOLO11m-seg", cms[thr], pid[thr],
                         sev[thr], names, threshold=thr)


def _unet_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=3, classes=NUM_CLASSES)


def _unet_dataset(stems, augment):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torch
    from torch.utils.data import Dataset

    tf = [A.Resize(IMGSZ, IMGSZ)]
    if augment:
        tf += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
               A.ShiftScaleRotate(0.05, 0.15, 25, border_mode=0, p=0.7),
               A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
               A.HueSaturationValue(10, 15, 10, p=0.3)]
    tf += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ToTensorV2()]
    transform = A.Compose(tf)

    class DS(Dataset):
        def __len__(self):
            return len(stems)

        def __getitem__(self, i):
            stem = stems[i]
            img = ImageOps.exif_transpose(Image.open(image_path(stem))).convert("RGB")
            mp = POOL / "masks" / f"{stem}.png"
            mask = Image.open(mp)
            if mask.mode != "L":
                mask = mask.convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            out = transform(image=np.array(img), mask=np.array(mask))
            return out["image"], out["mask"].long()

    return DS()


def train_unet_fold(fold, out_dir):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr = DataLoader(_unet_dataset(fold_stems(fold, "train"), True),
                    batch_size=4, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    va = DataLoader(_unet_dataset(fold_stems(fold, "val"), False),
                    batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = _unet_model().to(DEVICE)
    weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
    ce = nn.CrossEntropyLoss(weight=weights)

    def dice_loss(logits, target, smooth=1.0):
        probs = F.softmax(logits, 1)
        oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        card = probs.sum(dims) + oh.sum(dims)
        return 1 - ((2 * inter + smooth) / (card + smooth)).mean()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    EPOCHS, PATIENCE, WARMUP = 150, 20, 5
    best, since = -1.0, 0
    log, close_log = _history_writer(
        out_dir, ["epoch", "lr", "train_loss", "val_mean_dice"]
        + [f"val_dice_{c}" for c in CLASS_NAMES])

    for epoch in range(EPOCHS):
        lr = 3e-4 * (epoch + 1) / WARMUP if epoch < WARMUP else \
            0.5 * 3e-4 * (1 + np.cos(np.pi * (epoch - WARMUP) / max(1, EPOCHS - WARMUP)))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        run_loss, nb = 0.0, 0
        for x, y in tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            nb += 1
        train_loss = run_loss / max(nb, 1)

        model.eval()
        inter = np.zeros(NUM_CLASSES)
        card = np.zeros(NUM_CLASSES)
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    pred = model(x).float().argmax(1)
                for c in range(NUM_CLASSES):
                    p, t = (pred == c), (y == c)
                    inter[c] += (p & t).sum().item()
                    card[c] += p.sum().item() + t.sum().item()
        dice = np.where(card > 0, 2 * inter / np.maximum(card, 1), np.nan)
        mdice = float(np.nanmean(dice))

        log([epoch + 1, f"{lr:.6e}", f"{train_loss:.6f}", f"{mdice:.6f}"]
            + [f"{d:.6f}" if d == d else "" for d in dice])

        if mdice > best:
            best, since = mdice, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mean_dice": mdice}, out_dir / "best.pt")
            tag = " *best*"
        else:
            since += 1
            tag = f" ({since}/{PATIENCE})"
        if epoch % 5 == 0 or tag == " *best*":
            print(f"    epoch {epoch+1:>3}  loss {train_loss:.4f}  "
                  f"val mDice {mdice:.4f}  lesion {dice[LESION]:.4f}{tag}",
                  flush=True)
        if since >= PATIENCE:
            print(f"    early stop at epoch {epoch+1}, best {best:.4f}")
            break

    close_log()


def _unet_predict_semantic(model, stem):
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    img = ImageOps.exif_transpose(Image.open(image_path(stem))).convert("RGB")
    w, h = img.size
    tf = A.Compose([A.Resize(IMGSZ, IMGSZ),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)), ToTensorV2()])
    x = tf(image=np.array(img))["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pred = model(x).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    return pred, h, w


def evaluate_unet(fold, ckpt, out_dir):
    import torch
    model = _unet_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                     weights_only=False)["model"])
    model.eval()

    print(f"\n  evaluating held-out group '{fold}'...")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image, sev_pairs, names = [], [], []
    for stem in fold_stems(fold, "test"):
        pred, h, w = _unet_predict_semantic(model, stem)
        gt_full = gt_semantic_from_yolo(stem, h, w)
        gt = np.array(Image.fromarray(gt_full).resize((IMGSZ, IMGSZ), Image.NEAREST))
        c = confusion(gt, pred)
        cm += c
        per_image.append(c)
        sev_pairs.append((severity(gt), severity(pred)))
        names.append(image_path(stem).name)
    return write_results(out_dir, fold, "U-Net (ImageNet)", cm, per_image,
                         sev_pairs, names)


def _mrcnn_model():
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT", weights_backbone="DEFAULT", num_classes=91,
        min_size=IMGSZ, max_size=IMGSZ, box_detections_per_img=300,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=2000)
    f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(f, NUM_CLASSES)
    fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(fm, 256, NUM_CLASSES)
    return m


def _mrcnn_train_transform(augment=True):
    tf = []
    if augment:
        tf += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                               rotate_limit=25, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
        ]
    return A.Compose(
        tf,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                                 min_visibility=0.1),
    )


def _mrcnn_mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max()) + 1, float(ys.max()) + 1]


def _mrcnn_dataset(fold, split, train):
    import torch
    import torchvision
    from pycocotools.coco import COCO
    from torch.utils.data import Dataset

    ann = LOGO / "coco" / f"fold_{fold}_{split}.json"
    coco = COCO(str(ann))
    ids = sorted(coco.imgs.keys())

    class DS(Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            img_id = ids[i]
            info = coco.loadImgs(img_id)[0]
            stem = Path(info["file_name"]).stem
            img = Image.open(image_path(stem)).convert("RGB")
            W, H = img.size
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
            boxes, labels, masks = [], [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue
                m = coco.annToMask(a)
                if m.sum() == 0:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(a["category_id"])
                masks.append(m)

            img_np = np.array(img)

            if boxes:
                tf = _mrcnn_train_transform(augment=train)
                result = tf(image=img_np, bboxes=boxes, labels=labels,
                           masks=masks)
                transformed_masks = result["masks"]
                transformed_labels = result["labels"]

                kept_boxes, kept_masks, kept_labels = [], [], []
                for mask, lbl in zip(transformed_masks, transformed_labels):
                    box = _mrcnn_mask_to_box(mask)
                    if box is None:
                        continue
                    kept_boxes.append(box)
                    kept_masks.append(mask)
                    kept_labels.append(lbl)

                t = torchvision.transforms.functional.to_tensor(result["image"])
                if kept_boxes:
                    tgt = {"boxes": torch.as_tensor(kept_boxes, dtype=torch.float32),
                          "labels": torch.as_tensor(kept_labels, dtype=torch.int64),
                          "masks": torch.as_tensor(np.stack(kept_masks),
                                                   dtype=torch.uint8)}
                else:
                    tgt = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                          "labels": torch.zeros((0,), dtype=torch.int64),
                          "masks": torch.zeros((0, H, W), dtype=torch.uint8)}
            else:
                t = torchvision.transforms.functional.to_tensor(img)
                tgt = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros((0,), dtype=torch.int64),
                      "masks": torch.zeros((0, H, W), dtype=torch.uint8)}

            tgt["image_id"] = torch.tensor([img_id])
            return t, tgt

    return DS(), coco


def _collate(b):
    return tuple(zip(*b))


def train_maskrcnn_fold(fold, out_dir):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    tr_ds, _ = _mrcnn_dataset(fold, "train", True)
    va_ds, va_coco = _mrcnn_dataset(fold, "val", False)
    tr = DataLoader(tr_ds, batch_size=4, shuffle=True, num_workers=4,
                    collate_fn=_collate, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=4,
                    collate_fn=_collate, pin_memory=True)

    model = _mrcnn_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    EPOCHS, PATIENCE, WARMUP = 50, 20, 200
    best, since, it = -1.0, 0, 0
    log, close_log = _history_writer(
        out_dir, ["epoch", "lr", "train_loss", "val_mean_dice"]
        + [f"val_dice_{c}" for c in CLASS_NAMES])

    for epoch in range(EPOCHS):
        model.train()
        run_loss, nb = 0.0, 0
        for images, targets in tr:
            lr = 1e-4 * (it + 1) / WARMUP if it < WARMUP else \
                0.5 * 1e-4 * (1 + np.cos(np.pi * (it - WARMUP) /
                                         max(1, EPOCHS * len(tr) - WARMUP)))
            for g in opt.param_groups:
                g["lr"] = lr
            images = [i.to(DEVICE) for i in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda"):
                loss = sum(model(images, targets).values())
            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item()
                nb += 1
            it += 1
        train_loss = run_loss / max(nb, 1)

        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for images, targets in va:
                imgs = [i.to(DEVICE) for i in images]
                with torch.amp.autocast("cuda"):
                    outs = model(imgs)
                for img, tgt, out in zip(images, targets, outs):
                    h, w = img.shape[-2:]
                    stem = Path(va_coco.loadImgs(
                        int(tgt["image_id"].item()))[0]["file_name"]).stem
                    gt = gt_semantic_from_yolo(stem, h, w)
                    cm += confusion(gt, _mrcnn_pred_semantic(out, h, w, 0.5))
        dice = metrics_from_cm(cm)[0]
        score = float(np.nanmean(dice))

        log([epoch + 1, f"{lr:.6e}", f"{train_loss:.6f}", f"{score:.6f}"]
            + [f"{d:.6f}" if d == d else "" for d in dice])

        if score > best:
            best, since = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "mean_dice": score}, out_dir / "best.pt")
            tag = " *best*"
        else:
            since += 1
            tag = f" ({since}/{PATIENCE})"
        print(f"    epoch {epoch+1:>3}  loss {train_loss:.4f}  "
              f"val mDice {score:.4f}  lesion {dice[LESION]:.4f}{tag}", flush=True)
        if since >= PATIENCE:
            print(f"    early stop at epoch {epoch+1}, best {best:.4f}")
            break

    close_log()


def _mrcnn_pred_semantic(out, h, w, thr):
    sem = np.zeros((h, w), dtype=np.uint8)
    scores = out["scores"].float().cpu().numpy()
    labels = out["labels"].cpu().numpy()
    masks = (out["masks"].float() > 0.5).cpu().numpy()
    for target in (LEAF, LESION):
        for i in range(len(scores)):
            if scores[i] < thr or int(labels[i]) != target:
                continue
            m = masks[i, 0]
            if m.shape == (h, w):
                sem[m] = target
    return sem


def evaluate_maskrcnn(fold, ckpt, out_dir):
    import torch
    from torch.utils.data import DataLoader

    model = _mrcnn_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                     weights_only=False)["model"])
    model.eval()

    def run(split, thresholds):
        ds, coco = _mrcnn_dataset(fold, split, False)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4,
                        collate_fn=_collate)
        cms = {t: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
               for t in thresholds}
        sev = {t: [] for t in thresholds}
        pid = {t: [] for t in thresholds}
        names = []
        with torch.no_grad():
            for images, targets in dl:
                imgs = [i.to(DEVICE) for i in images]
                with torch.amp.autocast("cuda"):
                    outs = model(imgs)
                for img, tgt, out in zip(images, targets, outs):
                    h, w = img.shape[-2:]
                    info = coco.loadImgs(int(tgt["image_id"].item()))[0]
                    stem = Path(info["file_name"]).stem
                    gt = gt_semantic_from_yolo(stem, h, w)
                    gsev = severity(gt)
                    names.append(info["file_name"])
                    for t in thresholds:
                        pr = _mrcnn_pred_semantic(out, h, w, t)
                        c = confusion(gt, pr)
                        cms[t] += c
                        pid[t].append(c)
                        sev[t].append((gsev, severity(pr)))
        return cms, sev, pid, names

    print("\n  selecting threshold on val...")
    cms, _, _, _ = run("val", SCORE_THRESHOLDS)
    thr = sweep_and_pick({t: metrics_from_cm(cms[t])[0][LESION]
                          for t in SCORE_THRESHOLDS})

    print(f"\n  evaluating held-out group '{fold}'...")
    cms, sev, pid, names = run("test", [thr])
    return write_results(out_dir, fold, "Mask R-CNN", cms[thr], pid[thr],
                         sev[thr], names, threshold=thr)
