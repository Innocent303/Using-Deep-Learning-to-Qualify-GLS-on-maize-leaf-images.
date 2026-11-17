import contextlib
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_pipeline():
    try:
        this_file = Path(__file__).resolve()
        here = this_file.parent
    except NameError:
        this_file = None
        here = Path.cwd()

    candidates = [
        here,
        Path.cwd(),
        Path("/content/drive/MyDrive"),
        Path("/content"),
    ]
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        try:
            candidates += [p for p in drive.iterdir()
                           if p.is_dir() and (p / "maskrcnn_pipeline.py").exists()]
        except OSError:
            pass

    wrong_contents = []
    seen = set()
    for c in candidates:
        c = Path(c)
        if c in seen:
            continue
        seen.add(c)
        f = c / "maskrcnn_pipeline.py"
        if not f.exists():
            continue
        if this_file is not None and f.resolve() == this_file:
            continue
        try:
            text = f.read_text()
        except OSError:
            continue
        need_ds = "class " + "CocoInstanceDataset"
        need_fn = "def " + "build_model"
        if need_ds not in text or need_fn not in text:
            wrong_contents.append(f)
            continue
        sys.path.insert(0, str(c))
        return c, wrong_contents
    return None, wrong_contents


_HERE, _wrong = _find_pipeline()
if _HERE is None:
    msg = ["Could not find a valid maskrcnn_pipeline.py."]
    if _wrong:
        msg.append("\nThese files have the right NAME but the wrong CONTENTS:")
        for f in _wrong:
            first = ""
            try:
                for line in f.read_text().splitlines():
                    if line.strip() and not line.startswith("#!"):
                        first = line.strip().strip('"')
                        break
            except OSError:
                pass
            msg.append(f"    {f}   (starts with: {first[:60]!r})")
        msg.append(
            "\nYou likely saved the TRAINING script under the pipeline's name.\n"
            "You need two separate files:\n"
            "    maskrcnn_pipeline.py  - dataset, model, overfit test\n"
            "    maskrcnn_train.py     - this training script")
    else:
        msg.append(f"\nSearched from: {Path.cwd()}\n"
                   "Check with:  !find /content/drive/MyDrive -name maskrcnn_pipeline.py")
    raise RuntimeError("\n".join(msg))

print(f"Loading pipeline from {_HERE}")

from maskrcnn_pipeline import (
    ROOT, NUM_CLASSES, CLASS_NAMES, DEVICE, SEED, RUN_NAME,
    CocoInstanceDataset, build_model, collate_fn, seed_everything,
)

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutils


EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 200
GRAD_CLIP = 5.0
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 10
SCORE_THRESHOLD = 0.05

CACHE_TO_LOCAL = True
LOCAL_CACHE = Path("/content/coco_cache")

OUT_DIR = ROOT / "runs" / RUN_NAME


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


def autocast_ctx():
    try:
        return torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"))


def cache_dataset():
    if not CACHE_TO_LOCAL:
        return ROOT
    def cache_ok():
        if not (LOCAL_CACHE / "annotations").is_dir():
            return False
        for split in ("train", "val", "test"):
            src = ROOT / split
            if src.is_dir():
                d = LOCAL_CACHE / split
                if not d.is_dir() or len(list(d.glob("*"))) < len(list(src.glob("*"))):
                    return False
        return True

    if LOCAL_CACHE.exists() and not cache_ok():
        print(f"Cache at {LOCAL_CACHE} is incomplete — rebuilding")
        shutil.rmtree(LOCAL_CACHE, ignore_errors=True)

    if LOCAL_CACHE.exists():
        print(f"Using existing cache at {LOCAL_CACHE}")
        return LOCAL_CACHE
    print(f"Caching dataset to {LOCAL_CACHE}...")
    t0 = time.time()
    shutil.copytree(ROOT, LOCAL_CACHE,
                    ignore=shutil.ignore_patterns("runs"))
    print(f"  done in {time.time() - t0:.0f}s\n")
    return LOCAL_CACHE


@torch.no_grad()
def evaluate(model, loader, coco_gt):
    model.eval()
    results = []

    for images, targets in loader:
        images = [i.to(DEVICE) for i in images]
        with autocast_ctx():
            outputs = model(images)

        for tgt, out in zip(targets, outputs):
            img_id = int(tgt["image_id"].item())
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            boxes = out["boxes"].float().cpu().numpy()
            masks = (out["masks"].float() > 0.5).cpu().numpy()

            for i in range(len(scores)):
                if scores[i] < SCORE_THRESHOLD:
                    continue
                m = masks[i, 0].astype(np.uint8)
                rle = maskutils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                x1, y1, x2, y2 = boxes[i]
                results.append({
                    "image_id": img_id,
                    "category_id": int(labels[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "segmentation": rle,
                    "score": float(scores[i]),
                })

    if not results:
        print("  no detections above threshold — mAP is 0")
        return np.zeros(12), np.zeros(12), {}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)

    stats = {}
    per_class = {}
    for iou_type in ("bbox", "segm"):
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(coco_gt, coco_dt, iou_type)
            e.evaluate()
            e.accumulate()
            e.summarize()
        stats[iou_type] = e.stats.copy()

        if iou_type == "segm":
            prec = e.eval["precision"]
            for k, cat_id in enumerate(sorted(coco_gt.getCatIds())):
                p = prec[:, :, k, 0, -1]
                p = p[p > -1]
                per_class[cat_id] = float(np.mean(p)) if p.size else float("nan")

    return stats["bbox"], stats["segm"], per_class


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = cache_dataset()

    train_ds = CocoInstanceDataset(
        data_root / "train", data_root / "annotations" / "instances_train.json",
        train=True)
    val_ds = CocoInstanceDataset(
        data_root / "val", data_root / "annotations" / "instances_val.json",
        train=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn,
                        pin_memory=True)

    print(f"\ntrain: {len(train_ds)} images ({len(train_dl)} batches/epoch)")
    print(f"val:   {len(val_ds)} images")
    print(f"epochs {EPOCHS}, batch {BATCH_SIZE}, lr {LR}\n")

    model = build_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()

    total_iters = EPOCHS * len(train_dl)

    def lr_at(it):
        if it < WARMUP_ITERS:
            return LR * (it + 1) / WARMUP_ITERS
        prog = (it - WARMUP_ITERS) / max(1, total_iters - WARMUP_ITERS)
        return 0.5 * LR * (1 + np.cos(np.pi * prog))

    ckpt_last, ckpt_best = OUT_DIR / "last.pt", OUT_DIR / "best.pt"
    log_csv = OUT_DIR / "history.csv"
    start_epoch, best_map, since_best, global_it = 0, -1.0, 0, 0

    if ckpt_last.exists():
        ck = _torch_load(ckpt_last, DEVICE)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_map, since_best = ck["best_map"], ck.get("since_best", 0)
        global_it = ck.get("global_it", start_epoch * len(train_dl))
        print(f"Resumed from epoch {start_epoch} (best segm mAP {best_map:.4f})\n")
    else:
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "bbox_mAP", "segm_mAP",
                 "segm_mAP50", "segm_mAP75"]
                + [f"segm_AP_{n}" for n in CLASS_NAMES.values()])

    stopped_early = False
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0

        for images, targets in train_dl:
            lr = lr_at(global_it)
            for g in opt.param_groups:
                g["lr"] = lr

            images = [i.to(DEVICE) for i in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with autocast_ctx():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                print(f"  non-finite loss at iter {global_it}, skipping batch")
                opt.zero_grad(set_to_none=True)
                global_it += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()
            nb += 1
            global_it += 1

        train_loss = run_loss / max(nb, 1)
        bbox_stats, segm_stats, per_class = evaluate(model, val_dl, val_ds.coco)
        segm_map = float(segm_stats[0])
        dt = time.time() - t0

        pc = "  ".join(
            f"{CLASS_NAMES.get(cid, cid)} {ap:.3f}" for cid, ap in sorted(per_class.items()))
        marker = ""
        if segm_map > best_map:
            best_map, since_best = segm_map, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "segm_map": segm_map, "bbox_map": float(bbox_stats[0]),
                        "per_class": per_class}, ckpt_best)
            marker = "  *best*"
        else:
            since_best += 1
            marker = f"  (no gain {since_best}/{EARLY_STOP_PATIENCE})"

        print(f"epoch {epoch+1:>3}/{EPOCHS}  loss {train_loss:.4f}  "
              f"bbox mAP {bbox_stats[0]:.4f}  segm mAP {segm_map:.4f}  "
              f"[{pc}]  {dt:.0f}s{marker}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{lr:.3e}", f"{train_loss:.6f}",
                 f"{bbox_stats[0]:.6f}", f"{segm_map:.6f}",
                 f"{segm_stats[1]:.6f}", f"{segm_stats[2]:.6f}"]
                + [f"{per_class.get(cid, float('nan')):.6f}" for cid in sorted(CLASS_NAMES)])

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_map": best_map, "since_best": since_best,
                    "global_it": global_it}, ckpt_last)

        if EARLY_STOP_PATIENCE and since_best >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"Best was epoch {epoch + 1 - since_best}, segm mAP {best_map:.4f}.")
            stopped_early = True
            break

    if not stopped_early:
        print(f"\nRan all {EPOCHS} epochs without early stopping — "
              "mAP may still have been improving.")

    print(f"\nBest val segm mAP: {best_map:.4f}")
    best = _torch_load(ckpt_best, "cpu")
    summary = {
        "best_epoch": best["epoch"] + 1,
        "val_segm_mAP": best["segm_map"],
        "val_bbox_mAP": best["bbox_map"],
        "val_segm_AP_per_class": {CLASS_NAMES.get(int(k), k): v
                                  for k, v in best["per_class"].items()},
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(log_csv)))
        ep = [int(r["epoch"]) for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(ep, [float(r["train_loss"]) for r in rows])
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("Loss")
        ax[1].plot(ep, [float(r["segm_mAP"]) for r in rows], label="segm mAP")
        ax[1].plot(ep, [float(r["bbox_mAP"]) for r in rows], label="bbox mAP")
        for n in CLASS_NAMES.values():
            ax[1].plot(ep, [float(r[f"segm_AP_{n}"]) for r in rows], "--", label=f"segm AP {n}")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("AP"); ax[1].legend(fontsize=8)
        ax[1].set_title("Validation AP")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "curves.png", dpi=110)
        print(f"\nCurves: {OUT_DIR / 'curves.png'}")
    except Exception as e:
        print(f"(curves skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
