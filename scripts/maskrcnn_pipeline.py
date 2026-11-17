import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskutils
except ImportError:
    sys.exit("pycocotools missing:  !pip install -q pycocotools")


ROOT = Path("/content/drive/MyDrive/coco_split_800sq_aligned")

NUM_CLASSES = 3
CLASS_NAMES = {1: "leaf", 2: "lesion"}

BOX_DETECTIONS_PER_IMG = 300
RPN_POST_NMS_TOP_N_TRAIN = 2000
RPN_POST_NMS_TOP_N_TEST = 2000

PRETRAINED = True

IMG_SIZE_MIN = 800
IMG_SIZE_MAX = 800
SEED = int(os.environ.get("GLS_SEED", 1))

RUN_NAME = f"maskrcnn_seed{SEED}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OVERFIT_N = 4
OVERFIT_ITERS = 60
OVERFIT_LR = 5e-4


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform(augment=True):
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


def eval_transform():
    return train_transform(augment=False)


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max()) + 1, float(ys.max()) + 1]


class CocoInstanceDataset(Dataset):

    def __init__(self, img_dir, ann_file, train=True):
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.imgs.keys())
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.loadImgs(img_id)[0]
        path = self.img_dir / info["file_name"]
        img = Image.open(path).convert("RGB")
        W, H = img.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            m = self.coco.annToMask(a)
            if m.sum() == 0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(a["category_id"])
            masks.append(m)
            areas.append(a.get("area", (x2 - x1) * (y2 - y1)))
            iscrowd.append(a.get("iscrowd", 0))

        img_np = np.array(img)

        if boxes:
            tf = train_transform(augment=self.train) if self.train else eval_transform()
            result = tf(image=img_np, bboxes=boxes, labels=labels, masks=masks)

            transformed_masks = result["masks"]
            transformed_labels = result["labels"]

            kept_boxes, kept_masks, kept_labels, kept_areas, kept_crowd = \
                [], [], [], [], []
            for mask, lbl in zip(transformed_masks, transformed_labels):
                box = mask_to_box(mask)
                if box is None:
                    continue
                kept_boxes.append(box)
                kept_masks.append(mask)
                kept_labels.append(lbl)
                kept_areas.append((box[2] - box[0]) * (box[3] - box[1]))
                kept_crowd.append(0)

            img_t = torchvision.transforms.functional.to_tensor(result["image"])
            if kept_boxes:
                boxes_t = torch.as_tensor(kept_boxes, dtype=torch.float32)
                masks_t = torch.as_tensor(np.stack(kept_masks), dtype=torch.uint8)
                labels_t = torch.as_tensor(kept_labels, dtype=torch.int64)
                areas_t = torch.as_tensor(kept_areas, dtype=torch.float32)
                crowd_t = torch.as_tensor(kept_crowd, dtype=torch.int64)
            else:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                areas_t = torch.zeros((0,), dtype=torch.float32)
                crowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            img_t = torchvision.transforms.functional.to_tensor(img)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            crowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t, "labels": labels_t, "masks": masks_t,
            "image_id": torch.tensor([img_id]), "area": areas_t,
            "iscrowd": crowd_t,
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = "DEFAULT" if pretrained else None
    backbone_weights = "DEFAULT" if pretrained else None

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=backbone_weights,
        num_classes=91 if pretrained else num_classes,
        min_size=IMG_SIZE_MIN,
        max_size=IMG_SIZE_MAX,
        box_detections_per_img=BOX_DETECTIONS_PER_IMG,
        rpn_post_nms_top_n_train=RPN_POST_NMS_TOP_N_TRAIN,
        rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
    )

    if pretrained:
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)

        in_f_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_f_mask, 256, num_classes)

    return model


def overfit_test():
    seed_everything()
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pretrained weights: {PRETRAINED}")
    if not PRETRAINED:
        print("  !! Training Mask R-CNN from scratch on a small dataset usually")
        print("     fails to converge. See the note in the config block.")

    ann = ROOT / "annotations" / "instances_train.json"
    if not ann.exists():
        sys.exit(f"Not found: {ann}")

    ds = CocoInstanceDataset(ROOT / "train", ann, train=False)
    print(f"\ntrain images: {len(ds)}")

    subset = torch.utils.data.Subset(ds, list(range(OVERFIT_N)))
    dl = DataLoader(subset, batch_size=2, shuffle=True, num_workers=2,
                    collate_fn=collate_fn)

    imgs, targets = next(iter(dl))
    print(f"\nbatch of {len(imgs)}:")
    for k, (im, t) in enumerate(zip(imgs, targets)):
        lbls = t["labels"].tolist()
        counts = {CLASS_NAMES.get(c, c): lbls.count(c) for c in sorted(set(lbls))}
        print(f"  image {tuple(im.shape)}  {len(lbls)} instances  {counts}")
        print(f"    boxes {tuple(t['boxes'].shape)} {t['boxes'].dtype}   "
              f"masks {tuple(t['masks'].shape)} {t['masks'].dtype}")
        assert t["masks"].shape[-2:] == im.shape[-2:], "mask size != image size"
        assert t["boxes"].shape[0] == t["masks"].shape[0] == len(lbls), \
            "boxes/masks/labels count mismatch"
        if t["boxes"].numel():
            assert (t["boxes"][:, 2] > t["boxes"][:, 0]).all(), "degenerate box"
            assert (t["boxes"][:, 3] > t["boxes"][:, 1]).all(), "degenerate box"
            assert t["labels"].max() < NUM_CLASSES, \
                f"label {t['labels'].max()} >= NUM_CLASSES {NUM_CLASSES}"

    model = build_model().to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\ntrainable parameters: {n_par/1e6:.1f}M")

    with torch.no_grad():
        probe = imgs[0]
        tl, _ = model.transform([probe.to(DEVICE)], None)
        h, w = probe.shape[-2:]
        th, tw = tl.tensors.shape[-2:]
        print(f"internal transform: {w}x{h} -> {tw}x{th} "
              f"(padded to /32 multiples)")
        scale = model.transform.min_size[0] / min(h, w)
        if abs(scale - 1.0) > 0.01:
            print(f"  !! rescaling by {scale:.3f} — set IMG_SIZE_MIN/MAX to match "
                  f"your image size for a fair comparison")
        else:
            print("  no rescaling: input matches the U-Net's 800x800")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OVERFIT_LR, weight_decay=1e-4)

    model.train()
    it = 0
    first_loss = None
    print()
    while it < OVERFIT_ITERS:
        for imgs, targets in dl:
            imgs = [i.to(DEVICE) for i in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if first_loss is None:
                first_loss = loss.item()

            if not torch.isfinite(loss):
                print(f"iter {it}: loss is {loss.item()} — stopping.")
                print("  Usually a degenerate box or an empty target.")
                return

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            it += 1
            if it % 10 == 0 or it == 1:
                parts = "  ".join(f"{k.replace('loss_', '')} {v.item():.3f}"
                                  for k, v in loss_dict.items())
                print(f"iter {it:>3}/{OVERFIT_ITERS}  total {loss.item():.4f}   {parts}")
            if it >= OVERFIT_ITERS:
                break

    final = loss.item()
    final_parts = {k: v.item() for k, v in loss_dict.items()}
    obj = final_parts.get("loss_objectness", 1.0)
    cls = final_parts.get("loss_classifier", 1.0)
    drop = first_loss / final if final > 0 else float("inf")

    print("\n" + "=" * 58)
    print(f"loss {first_loss:.3f} -> {final:.3f}  ({drop:.1f}x reduction)")
    print(f"objectness {obj:.4f}   classifier {cls:.4f}")

    checks = {
        "RPN learned objectness (< 0.05)": obj < 0.05,
        "classifier separated classes (< 0.15)": cls < 0.15,
        "total loss fell at least 3x": drop >= 3.0,
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    model.eval()
    with torch.no_grad():
        img, tgt = ds[0]
        out = model([img.to(DEVICE)])[0]
    keep = out["scores"] > 0.5
    n_gt, n_pred = len(tgt["labels"]), int(keep.sum())
    recovered = n_pred / n_gt if n_gt else 0.0

    print(f"\nsanity prediction on a memorised image:")
    print(f"  ground truth: {n_gt} instances")
    print(f"  predicted:    {n_pred} instances above score 0.5 "
          f"({len(out['scores'])} raw)  = {100*recovered:.0f}% recovered")
    if len(out["scores"]):
        print(f"  top score: {out['scores'].max().item():.3f}")

    instance_ok = 0.7 <= recovered <= 1.3
    print(f"  [{'PASS' if instance_ok else 'FAIL'}] recovers 70-130% of instances")

    print("\n" + "=" * 58)
    if all(checks.values()) and instance_ok:
        print("PASS — the pipeline works end to end. Move on to full training.")
        print("Note: a residual mask loss of ~0.15-0.20 is the normal floor for")
        print("Mask R-CNN and does NOT indicate a problem.")
    else:
        print("Something is off. Check, in order:")
        print("  1. run more iterations (dense images need longer)")
        print("  2. visualise a mask over its image — is it aligned?")
        print("  3. PRETRAINED = False stalls with high objectness by design")


if __name__ == "__main__":
    overfit_test()
