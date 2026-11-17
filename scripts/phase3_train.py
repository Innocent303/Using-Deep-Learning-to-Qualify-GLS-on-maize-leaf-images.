import csv
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


def _find_module(name):
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    candidates = [here, Path.cwd(), Path("/content/drive/MyDrive"), Path("/content")]
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        try:
            candidates += [p for p in drive.iterdir()
                           if p.is_dir() and (p / f"{name}.py").exists()]
        except OSError:
            pass
    for c in candidates:
        if (Path(c) / f"{name}.py").exists():
            sys.path.insert(0, str(c))
            return Path(c)
    return None


_HERE = _find_module("phase2_pipeline")
if _HERE is None:
    raise RuntimeError(
        "Could not find phase2_pipeline.py.\n"
        "Put it next to this script, then:  %cd /content/drive/MyDrive\n"
        f"Searched from: {Path.cwd()}\n"
        "Check with:  !find /content/drive/MyDrive -name phase2_pipeline.py"
    )
print(f"Loading pipeline from {_HERE}")

from phase2_pipeline import (
    ROOT, NUM_CLASSES, CLASS_NAMES, IMG_SIZE, DEVICE,
    IMAGES_SUBDIR, MASKS_SUBDIR,
    SegDataset, UNet, build_unet, MODEL_TAG, RUN_NAME, PRETRAINED_ENCODER,
    ComboLoss, per_class_dice,
    list_pairs, train_transform, eval_transform, seed_everything,
)


EPOCHS = 150
BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
NUM_WORKERS = 4

CLASS_WEIGHTS = [1.0, 1.0, 2.0]

EARLY_STOP_PATIENCE = 20

CACHE_TO_LOCAL = True
LOCAL_CACHE = Path("/content/unet_cache")

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


def cache_dataset(splits=("train", "val")):
    if not CACHE_TO_LOCAL:
        return ROOT
    def cache_ok():
        for split in ("train", "val", "test"):
            for sub in (IMAGES_SUBDIR, MASKS_SUBDIR):
                src = ROOT / split / sub
                if src.is_dir():
                    d = LOCAL_CACHE / split / sub
                    if not d.is_dir() or len(list(d.glob("*"))) < len(list(src.glob("*"))):
                        return False
        return True

    if LOCAL_CACHE.exists() and not cache_ok():
        print(f"Cache at {LOCAL_CACHE} is incomplete — rebuilding")
        shutil.rmtree(LOCAL_CACHE, ignore_errors=True)

    if LOCAL_CACHE.exists():
        print(f"Using existing cache at {LOCAL_CACHE}")
        return LOCAL_CACHE

    print(f"Caching dataset to {LOCAL_CACHE} (one-off, speeds up every epoch)...")
    t0 = time.time()
    copied = 0
    for split in splits:
        for sub in (IMAGES_SUBDIR, MASKS_SUBDIR):
            src = ROOT / split / sub
            if not src.is_dir():
                raise SystemExit(
                    f"Expected folder does not exist: {src}\n"
                    f"Check IMAGES_SUBDIR / MASKS_SUBDIR in phase2_pipeline.py "
                    f"(currently '{IMAGES_SUBDIR}' and '{MASKS_SUBDIR}')."
                )
            shutil.copytree(src, LOCAL_CACHE / split / sub, dirs_exist_ok=True)
            copied += len(list((LOCAL_CACHE / split / sub).glob("*")))
    print(f"  {copied} files in {time.time() - t0:.0f}s\n")
    return LOCAL_CACHE


def build_loaders(data_root):
    import phase2_pipeline as p2
    original = p2.ROOT
    p2.ROOT = data_root
    try:
        train_pairs = list_pairs("train")
        val_pairs = list_pairs("val")
    finally:
        p2.ROOT = original

    if not train_pairs or not val_pairs:
        raise SystemExit("Empty train or val split — check paths.")

    train_ds = SegDataset(train_pairs, train_transform(IMG_SIZE, augment=True))
    val_ds = SegDataset(val_pairs, eval_transform(IMG_SIZE))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          drop_last=True, persistent_workers=NUM_WORKERS > 0)
    val_dl = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=NUM_WORKERS > 0)
    return train_dl, val_dl, len(train_pairs), len(val_pairs)


def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return LR * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * LR * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES, device=DEVICE)
    card = torch.zeros(NUM_CLASSES, device=DEVICE)

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

        pred = logits.float().argmax(1)
        for c in range(NUM_CLASSES):
            p, t = (pred == c), (y == c)
            inter[c] += (p & t).sum()
            card[c] += p.sum() + t.sum()

    dice = torch.where(card > 0, 2 * inter / card,
                       torch.full_like(card, float("nan")))
    return loss_sum / max(n, 1), dice.cpu().numpy()


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {IMG_SIZE}   batch: {BATCH_SIZE}   epochs: {EPOCHS}\n")

    data_root = cache_dataset()
    train_dl, val_dl, n_train, n_val = build_loaders(data_root)
    print(f"train: {n_train} pairs ({len(train_dl)} batches/epoch)")
    print(f"val:   {n_val} pairs\n")

    model = build_unet().to(DEVICE)
    weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    criterion = ComboLoss(class_weights=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()

    ckpt_last = OUT_DIR / "last.pt"
    ckpt_best = OUT_DIR / "best.pt"
    log_csv = OUT_DIR / "history.csv"

    start_epoch, best_dice, epochs_since_best = 0, -1.0, 0

    if ckpt_last.exists():
        ck = _torch_load(ckpt_last, DEVICE)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_dice = ck["best_dice"]
        epochs_since_best = ck.get("epochs_since_best", 0)
        print(f"Resumed from epoch {start_epoch} (best val Dice {best_dice:.4f})\n")
    else:
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "val_loss", "val_mean_dice"]
                       + [f"val_dice_{c}" for c in CLASS_NAMES])

    for epoch in range(start_epoch, EPOCHS):
        lr = lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)
        val_loss, val_dice = validate(model, val_dl, criterion)
        mean_dice = float(np.nanmean(val_dice))
        dt = time.time() - t0

        per_class = "  ".join(f"{n} {d:.3f}" for n, d in zip(CLASS_NAMES, val_dice))
        marker = ""
        if mean_dice > best_dice:
            best_dice, epochs_since_best = mean_dice, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice.tolist(), "mean_dice": mean_dice,
                        "img_size": IMG_SIZE, "num_classes": NUM_CLASSES},
                       ckpt_best)
            marker = "  *best*"
        else:
            epochs_since_best += 1

        stale = ""
        if EARLY_STOP_PATIENCE and epochs_since_best > 0:
            stale = f"  (no gain {epochs_since_best}/{EARLY_STOP_PATIENCE})"

        print(f"epoch {epoch+1:>3}/{EPOCHS}  lr {lr:.2e}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"mDice {mean_dice:.4f}  [{per_class}]  {dt:.0f}s{marker}{stale}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{lr:.6e}", f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{mean_dice:.6f}"] + [f"{d:.6f}" for d in val_dice])

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_dice": best_dice,
                    "epochs_since_best": epochs_since_best}, ckpt_last)

        if EARLY_STOP_PATIENCE and epochs_since_best >= EARLY_STOP_PATIENCE:
            best_epoch = epoch + 1 - epochs_since_best
            print(f"\nEarly stop: no improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"Best was epoch {best_epoch} with mean Dice {best_dice:.4f}.")
            stopped_early = True
            break
    else:
        stopped_early = False

    if not stopped_early:
        print(f"\nRan the full {EPOCHS} epochs without early stopping.")
        print("Val Dice may still have been improving — consider raising EPOCHS.")

    print(f"\nBest validation mean Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {ckpt_best}")

    best = _torch_load(ckpt_best, "cpu")
    summary = {"best_epoch": best["epoch"] + 1, "mean_dice": best["mean_dice"],
               "per_class_dice": dict(zip(CLASS_NAMES, best["val_dice"])),
               "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
               "lr": LR, "class_weights": CLASS_WEIGHTS}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(log_csv)))
        ep = [int(r["epoch"]) for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
        ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
        ax[0].set_title("Loss")
        for c in CLASS_NAMES:
            ax[1].plot(ep, [float(r[f"val_dice_{c}"]) for r in rows], label=c)
        ax[1].plot(ep, [float(r["val_mean_dice"]) for r in rows], "k--", label="mean")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Dice"); ax[1].legend()
        ax[1].set_title("Validation Dice")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "curves.png", dpi=110)
        print(f"\nCurves: {OUT_DIR / 'curves.png'}")
    except Exception as e:
        print(f"(curves skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
