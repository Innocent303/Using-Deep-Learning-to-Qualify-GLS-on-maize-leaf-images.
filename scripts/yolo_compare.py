import os
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DATA_ROOT = Path("/content/drive/MyDrive/yolo_seg_800sq")
SEED = int(os.environ.get("GLS_SEED", 1))
RUN_NAME = f"yolo11m_seg_800_seed{SEED}"
WEIGHTS = DATA_ROOT / "runs" / RUN_NAME / "weights" / "best.pt"

SPLIT = "test"
OUT_DIR = DATA_ROOT / "runs" / RUN_NAME / f"compare_{SPLIT}"

IMGSZ = 800
MAX_DET = 300

FIXED_THRESHOLD = 0.20
SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

NMS_IOU = 0.7

BACKGROUND, LEAF, LESION = 0, 1, 2
CLASS_NAMES = ["background", "leaf", "GLS_lesion"]
NUM_CLASSES = 3
YOLO_TO_SEMANTIC = {0: LEAF, 1: LESION}

OTHER_METRICS = {
    "U-Net": Path(f"/content/drive/MyDrive/unet/runs/phase3/eval_test/metrics.json"),
    "Mask R-CNN": Path("/content/drive/MyDrive/coco_split_800sq_aligned/runs/maskrcnn")
                  / "compare_test" / "maskrcnn_semantic_metrics.json",
}


def gt_semantic_from_label(label_path, h, w):
    sem = Image.new("L", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(sem)
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cid = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if len(coords) < 6 or len(coords) % 2:
                continue
            rows.append((cid, coords))

    for target in (LEAF, LESION):
        for cid, coords in rows:
            if YOLO_TO_SEMANTIC.get(cid) != target:
                continue
            pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=target)
    return np.array(sem, dtype=np.uint8)


def pred_semantic(result, h, w, score_thr):
    sem = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.boxes) == 0:
        return sem

    masks = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for target in (LEAF, LESION):
        for i in range(len(scores)):
            if scores[i] < score_thr:
                continue
            if YOLO_TO_SEMANTIC.get(int(classes[i])) != target:
                continue
            m = masks[i]
            if m.shape != (h, w):
                m = np.array(Image.fromarray((m > 0.5).astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            else:
                m = m > 0.5
            sem[m] = target
    return sem


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


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics missing:  !pip install -q ultralytics")

    if not WEIGHTS.exists():
        sys.exit(f"No weights at {WEIGHTS}")

    img_dir = DATA_ROOT / SPLIT / "images"
    lbl_dir = DATA_ROOT / SPLIT / "labels"
    if not img_dir.is_dir():
        sys.exit(f"Not found: {img_dir}")

    images = sorted(p for p in img_dir.iterdir()
                    if p.is_file() and not p.name.startswith("."))
    print(f"Evaluating {len(images)} image(s) from '{SPLIT}'")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))

    cms = {t: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
           for t in SCORE_THRESHOLDS}
    sev = {t: [] for t in SCORE_THRESHOLDS}
    per_img_dice = {t: [] for t in SCORE_THRESHOLDS}
    names = []

    for n, img_path in enumerate(images, 1):
        with Image.open(img_path) as im:
            w, h = im.size

        gt = gt_semantic_from_label(lbl_dir / f"{img_path.stem}.txt", h, w)
        gt_sev = severity(gt)
        names.append(img_path.name)

        res = model.predict(str(img_path), imgsz=IMGSZ, conf=min(SCORE_THRESHOLDS),
                            iou=NMS_IOU, max_det=MAX_DET, retina_masks=True,
                            verbose=False)[0]

        for t in SCORE_THRESHOLDS:
            pr = pred_semantic(res, h, w, t)
            c = confusion(gt, pr)
            cms[t] += c
            per_img_dice[t].append(c)
            sev[t].append((gt_sev, severity(pr)))

        if n % 20 == 0:
            print(f"  {n}/{len(images)}...")

    print("\nSCORE THRESHOLD SWEEP")
    print(f"{'thr':>6} {'mean Dice':>11} {'leaf':>8} {'lesion':>8} "
          f"{'sev MAE':>9} {'sev bias':>9}")
    print("-" * 56)
    rows, tuned_t, best = {}, None, -1.0
    for t in SCORE_THRESHOLDS:
        dice, iou, prec, rec = metrics_from_cm(cms[t])
        pairs = np.array(sev[t], dtype=float)
        ok = np.isfinite(pairs).all(axis=1)
        err = pairs[ok, 1] - pairs[ok, 0] if ok.sum() else np.array([np.nan])
        mae, bias = float(np.mean(np.abs(err))), float(np.mean(err))
        rows[t] = (dice, iou, prec, rec, mae, bias)
        print(f"{t:>6.2f} {np.nanmean(dice):>11.4f} {dice[LEAF]:>8.4f} "
              f"{dice[LESION]:>8.4f} {mae:>9.2f} {bias:>+9.2f}")
        if np.isfinite(dice[LESION]) and dice[LESION] > best:
            best, tuned_t = dice[LESION], t

    if FIXED_THRESHOLD is not None:
        best_t = FIXED_THRESHOLD
        print(f"\nUsing FIXED threshold {best_t:.2f} (selected on val).")
        if best_t != tuned_t:
            print(f"  (best on THIS split would be {tuned_t:.2f} — "
                  "the gap is the optimism avoided.)")
    else:
        best_t = tuned_t
        print(f"\nBest lesion Dice at threshold {best_t:.2f}")
        if SPLIT == "test":
            print("  !! tuned ON TEST — set FIXED_THRESHOLD and re-run.")

    dice, iou, prec, rec, mae, bias = rows[best_t]
    cm = cms[best_t]

    print(f"\nYOLO-seg as semantic segmentation  (threshold {best_t:.2f})")
    print(f"{'class':>12} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 46)
    for c, nm in enumerate(CLASS_NAMES):
        print(f"{nm:>12} {dice[c]:>8.4f} {iou[c]:>8.4f} {prec[c]:>8.4f} {rec[c]:>8.4f}")
    print(f"{'mean':>12} {np.nanmean(dice):>8.4f} {np.nanmean(iou):>8.4f}")

    print("\nCONFUSION MATRIX (rows = truth, % of true class)")
    wdt = 13
    print(" " * (wdt + 2) + "".join(f"{nm:>{wdt}}" for nm in CLASS_NAMES))
    for i, nm in enumerate(CLASS_NAMES):
        r = cm[i].sum()
        print(f"{nm:>{wdt}}  " + "".join(
            f"{(100*cm[i,j]/r if r else 0):>{wdt-1}.1f}%" for j in range(NUM_CLASSES)))

    pairs = np.array(sev[best_t], dtype=float)
    ok = np.isfinite(pairs).all(axis=1)
    t_s, p_s = pairs[ok, 0], pairs[ok, 1]
    err = p_s - t_s
    r = float(np.corrcoef(t_s, p_s)[0, 1]) if ok.sum() > 1 else float("nan")

    print(f"\nGLS SEVERITY")
    print(f"  images:    {int(ok.sum())}")
    print(f"  true mean: {t_s.mean():.2f}%   predicted mean: {p_s.mean():.2f}%")
    print(f"  MAE:       {np.mean(np.abs(err)):.2f} pp")
    print(f"  bias:      {np.mean(err):+.2f} pp")
    print(f"  Pearson r: {r:.4f}")
    for wpp in (1, 2, 5):
        print(f"  within +/-{wpp} pp: {100*np.mean(np.abs(err) <= wpp):.1f}%")

    result = {
        "split": SPLIT, "model": "YOLO11m-seg", "score_threshold": best_t,
        "confusion_matrix": cm.tolist(),
        "dice": dict(zip(CLASS_NAMES, dice.tolist())),
        "iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "mean_dice": float(np.nanmean(dice)),
        "mean_iou": float(np.nanmean(iou)),
        "severity": {"mae_pp": float(np.mean(np.abs(err))),
                     "bias_pp": float(np.mean(err)),
                     "pearson_r": r, "n_images": int(ok.sum())},
    }
    (OUT_DIR / "yolo_semantic_metrics.json").write_text(json.dumps(result, indent=2))

    with open(OUT_DIR / "per_image.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["image", "model", "split", "mean_dice", "mean_iou"]
                     + [f"dice_{c}" for c in CLASS_NAMES]
                     + [f"iou_{c}" for c in CLASS_NAMES]
                     + ["true_lesion_px", "pred_lesion_px", "true_leaf_px",
                        "has_lesion", "severity_true_pct", "severity_pred_pct",
                        "severity_error_pp"])
        for nm, c, (gs, ps) in zip(names, per_img_dice[best_t], sev[best_t]):
            c = np.asarray(c)
            d_i, i_i, _, _ = metrics_from_cm(c)
            e = ps - gs if np.isfinite([gs, ps]).all() else float("nan")

            def fm(v):
                return "" if (v is None or (isinstance(v, float) and v != v)) \
                    else f"{v:.6f}"

            wtr.writerow([nm, "YOLO11m-seg", SPLIT,
                          fm(float(np.nanmean(d_i))), fm(float(np.nanmean(i_i)))]
                         + [fm(x) for x in d_i] + [fm(x) for x in i_i]
                         + [int(c[LESION].sum()), int(c[:, LESION].sum()),
                            int(c[LEAF].sum()), int(c[LESION].sum() > 0),
                            fm(gs), fm(ps), fm(e)])

    others = {}
    for label, path in OTHER_METRICS.items():
        if path.exists():
            others[label] = json.loads(path.read_text())
        else:
            print(f"\n(missing {label} metrics: {path})")

    if others:
        print("\n" + "=" * 74)
        print(f"THREE-WAY COMPARISON — {SPLIT} split, all at 800x800")
        print("=" * 74)
        labels = list(others) + ["YOLO11m-seg"]
        allm = dict(others); allm["YOLO11m-seg"] = result

        hdr = f"{'metric':>22}" + "".join(f"{l:>16}" for l in labels)
        print(hdr); print("-" * len(hdr))

        def line(name, getter, fmt="{:>16.4f}"):
            vals = []
            for l in labels:
                try:
                    vals.append(getter(allm[l]))
                except (KeyError, TypeError):
                    vals.append(float("nan"))
            print(f"{name:>22}" + "".join(fmt.format(v) for v in vals))

        line("mean Dice", lambda m: m["mean_dice"])
        line("mean IoU", lambda m: m["mean_iou"])
        for nm in CLASS_NAMES:
            line(f"Dice {nm}", lambda m, k=nm: m["dice"][k])
        line("severity MAE (pp)", lambda m: m["severity"]["mae_pp"])
        line("severity bias (pp)", lambda m: m["severity"]["bias_pp"])
        line("severity r", lambda m: m["severity"]["pearson_r"])

        print("\nNotes:")
        print("  - Initialisation differs: the scratch U-Net uses no pretrained")
        print("    weights; the ImageNet U-Net, Mask R-CNN and YOLO all do.")
        print("  - All three fed identical 800x800 square inputs.")
        print("  - Differences under ~0.03 Dice are not meaningful at n=65;")
        print("    run significance_test.py on the per_image.csv files.")

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
