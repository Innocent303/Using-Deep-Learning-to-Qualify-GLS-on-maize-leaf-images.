import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


OUT_ROOT = Path("/content/drive/MyDrive/pretrain_exp/pretrain")

IMG = 256
EPOCHS = 10
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
HOLDOUT = 0.10
SPLIT_SEED = 0
EXPECT_TOTAL = 4188
EXPECT_CLASSES = 4
ARMS = ["maize", "control"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png"}


def run_dir(arm, seed):
    return OUT_ROOT / arm / f"seed{seed}"


def is_done(arm, seed):
    return (run_dir(arm, seed) / "DONE").exists()


def status():
    print(f"{OUT_ROOT}")
    print(f"{'arm':<10}" + "".join(f"{'s' + str(s):>8}" for s in range(1, 11)))
    for arm in ARMS:
        cells = []
        for s in range(1, 11):
            if is_done(arm, s):
                try:
                    m = json.loads((run_dir(arm, s) / "metrics.json").read_text())
                    cells.append(f"{m['final_heldout_acc']:>8.3f}")
                except Exception:
                    cells.append(f"{'done':>8}")
            else:
                cells.append(f"{'-':>8}")
        print(f"{arm:<10}" + "".join(cells))
    print("(cells: held-out accuracy of the last epoch)")


def scan_dataset(root):
    root = Path(root)
    if not root.exists():
        sys.exit(f"--data folder not found: {root}")
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len([d for d in dirs if any(f.suffix.lower() in EXTS for f in d.iterdir())]) < 2 \
            and len(dirs) == 1:
        return scan_dataset(dirs[0])
    classes, files = [], []
    for d in dirs:
        fs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)
        if fs:
            classes.append(d.name)
            files.append(fs)
    return root, classes, files


def split_files(files_per_class):
    train, hold = [], []
    for ci, fs in enumerate(files_per_class):
        rng = np.random.default_rng(SPLIT_SEED + ci)
        perm = rng.permutation(len(fs))
        n_hold = int(round(HOLDOUT * len(fs)))
        hold += [(fs[i], ci) for i in perm[:n_hold]]
        train += [(fs[i], ci) for i in perm[n_hold:]]
    return train, hold


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transform(train):
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2
    tf = [A.Resize(IMG, IMG, interpolation=cv2.INTER_AREA)]
    if train:
        tf += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
        ]
    tf += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(tf)


def make_dataset_class():
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import Dataset

    class ClsDS(Dataset):
        def __init__(self, items, train):
            self.items = items
            self.tf = make_transform(train)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            return self.tf(image=np.array(img))["image"], int(y)

    return ClsDS


def worker_init(_worker_id):
    import torch
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
    info = torch.utils.data.get_worker_info()
    tf = getattr(info.dataset, "tf", None) if info else None
    if tf is not None and hasattr(tf, "set_random_seed"):
        try:
            tf.set_random_seed(int(s))
        except Exception:
            pass


def build_classifier(n_classes):
    import torch.nn as nn
    import segmentation_models_pytorch as smp

    unet = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3,
                    classes=n_classes)

    class Clf(nn.Module):
        def __init__(self, encoder, n):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(512, n)

        def forward(self, x):
            f = self.encoder(x)[-1]
            return self.fc(f.mean((2, 3)))

    return Clf(unet.encoder, n_classes)


def train_one(arm, seed, data_root, expect_total, force):
    import math
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    out = run_dir(arm, seed)
    if is_done(arm, seed) and not force:
        print(f"  {arm} seed {seed}: already done, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    root, classes, files = scan_dataset(data_root)
    counts = [len(f) for f in files]
    total = sum(counts)
    print(f"  data: {root}\n  classes ({len(classes)}): "
          + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)) + f"   total {total}")
    if len(classes) != EXPECT_CLASSES:
        sys.exit(f"!! expected {EXPECT_CLASSES} classes, found {len(classes)}")
    if expect_total and total != expect_total:
        sys.exit(f"!! expected {expect_total} images, found {total}. Both arms must have "
                 f"the same image count — fix the data or pass --expect-total.")

    train_items, hold_items = split_files(files)
    print(f"  train {len(train_items)}   held-out {len(hold_items)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("  !! no GPU — this will be very slow")
    ClsDS = make_dataset_class()
    nw = min(4, os.cpu_count() or 2)

    seed_everything(seed)
    model = build_classifier(len(classes)).to(device)

    gen = torch.Generator().manual_seed(seed)
    tr = DataLoader(ClsDS(train_items, True), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=nw, pin_memory=True,
                    worker_init_fn=worker_init, generator=gen)
    ho = DataLoader(ClsDS(hold_items, False), batch_size=64, shuffle=False,
                    num_workers=nw, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    ce = nn.CrossEntropyLoss()

    spe = len(tr)
    total_steps, warm = EPOCHS * spe, spe
    step = 0

    def lr_at(s):
        if s < warm:
            return LR * (s + 1) / warm
        return 0.5 * LR * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))

    hist_path = out / "history.csv"
    with open(hist_path, "w", newline="") as hf:
        hw = csv.writer(hf)
        hw.writerow(["epoch", "lr_end", "train_loss", "train_acc", "heldout_loss", "heldout_acc"])
        t0 = time.time()
        for epoch in range(EPOCHS):
            model.train()
            tl, tc, tn = 0.0, 0, 0
            for x, y in tr:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=use_amp):
                    logits = model(x)
                    loss = ce(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt)
                scaler.update()
                step += 1
                tl += loss.item() * len(y)
                tc += (logits.argmax(1) == y).sum().item()
                tn += len(y)

            model.eval()
            hl, hc, hn = 0.0, 0, 0
            conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
            with torch.no_grad():
                for x, y in ho:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device, enabled=use_amp):
                        logits = model(x)
                    hl += ce(logits.float(), y).item() * len(y)
                    pred = logits.argmax(1)
                    hc += (pred == y).sum().item()
                    hn += len(y)
                    for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                        conf[t, p] += 1
            row = [epoch + 1, f"{lr_at(step - 1):.3e}", f"{tl / tn:.4f}", f"{tc / tn:.4f}",
                   f"{hl / hn:.4f}", f"{hc / hn:.4f}"]
            hw.writerow(row)
            hf.flush()
            print(f"    epoch {epoch + 1:>2}/{EPOCHS}  train loss {tl / tn:.4f} acc {tc / tn:.3f}"
                  f"   held-out loss {hl / hn:.4f} acc {hc / hn:.3f}"
                  f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)

    enc_state = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
    torch.save(enc_state, out / "encoder.pt")
    import segmentation_models_pytorch as smp
    probe = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    probe.encoder.load_state_dict(torch.load(out / "encoder.pt", map_location="cpu"),
                                  strict=True)
    print("  encoder.pt exported and loads into smp.Unet(resnet34) with strict=True")

    metrics = {
        "arm": arm, "seed": seed, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "img": IMG, "split_seed": SPLIT_SEED,
        "classes": classes, "class_counts": counts, "n_total": total,
        "n_train": len(train_items), "n_heldout": len(hold_items),
        "final_heldout_acc": hc / hn, "final_heldout_loss": hl / hn,
        "heldout_confusion": conf.tolist(),
        "checkpoint": "last epoch", "early_stopping": False,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "DONE").write_text(json.dumps({"seed": seed, "arm": arm,
                                          "heldout_acc": hc / hn}, indent=2))
    print(f"  done: held-out accuracy {hc / hn:.4f}")


def main():
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--data", help="folder containing the class sub-folders")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--expect-total", type=int, default=EXPECT_TOTAL,
                    help="0 disables the image-count check")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    args, _ = ap.parse_known_args()

    OUT_ROOT = Path(args.out)
    if args.status:
        status()
        return
    if not args.arm or not args.data:
        ap.error("--arm and --data are required")

    print(f"{args.arm}: seeds {args.seeds}")
    for seed in args.seeds:
        print("\n" + "=" * 70 + f"\n{args.arm}  seed {seed}\n" + "=" * 70, flush=True)
        try:
            train_one(args.arm, seed, args.data, args.expect_total, args.force)
        except KeyboardInterrupt:
            print("\ninterrupted — finished seeds are kept; re-run to continue")
            return
        except SystemExit:
            raise
        except Exception as e:
            import traceback
            print(f"\n  !! {args.arm} seed {seed} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    status()


if __name__ == "__main__":
    main()
