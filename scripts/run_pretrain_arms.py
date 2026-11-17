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


EXP = Path("/content/drive/MyDrive/pretrain_exp")
DATA_ROOT = Path("/content/drive/MyDrive/unet")
IMAGES_SUBDIR, MASKS_SUBDIR = "images", "labels"
SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}

ARMS = ["scratch", "imagenet", "maize", "control"]
ARM_TITLE = {"scratch": "Scratch", "imagenet": "ImageNet",
             "maize": "ImageNet + maize", "control": "ImageNet + non-plant control"}
SEEDS_DEFAULT = list(range(1, 11))

SIZE = 256
NUM_CLASSES = 3
CLASS_NAMES = ["background", "leaf", "GLS_lesion"]
LESION, LEAF = 2, 1

SELECT_BY = "per_image"
EPOCHS, PATIENCE, WARMUP = 150, 20, 5
BATCH, LR, WEIGHT_DECAY, CLIP = 4, 3e-4, 1e-4, 1.0
CE_WEIGHTS = [1.0, 1.0, 2.0]

PLANNED = [("imagenet", "scratch"), ("maize", "imagenet"),
           ("control", "imagenet"), ("maize", "control")]
RELEVANCE = 0.02

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seg_dir(arm, seed):
    return EXP / "seg" / arm / f"seed{seed}"


def pre_dir(arm, seed):
    return EXP / "pretrain" / arm / f"seed{seed}"


def is_done(arm, seed):
    return (seg_dir(arm, seed) / "DONE").exists()


def status(seeds):
    print(f"{EXP / 'seg'}")
    print(f"{'arm':<10}" + "".join(f"{'s' + str(s):>8}" for s in seeds))
    for arm in ARMS:
        cells = []
        for s in seeds:
            if is_done(arm, s):
                try:
                    d = json.loads((seg_dir(arm, s) / "DONE").read_text())
                    cells.append(f"{d['mean_per_image_lesion_dice']:>8.4f}")
                except Exception:
                    cells.append(f"{'done':>8}")
            else:
                cells.append(f"{'-':>8}")
        print(f"{arm:<10}" + "".join(cells))
    print("(cells: test per-image mean lesion Dice at 256 x 256)")


def list_pairs(split):
    d = DATA_ROOT / SPLIT_DIRS[split]
    imgs = {p.stem.lower(): p for p in sorted((d / IMAGES_SUBDIR).glob("*"))
            if p.is_file() and not p.name.startswith(".")}
    masks = {p.stem.lower(): p for p in sorted((d / MASKS_SUBDIR).glob("*"))
             if p.is_file() and not p.name.startswith(".")}
    keys = sorted(set(imgs) & set(masks))
    return [(k, imgs[k], masks[k]) for k in keys]


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transform(augment):
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2
    tf = [A.Resize(SIZE, SIZE, interpolation=cv2.INTER_AREA)]
    if augment:
        tf += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
               A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25,
                                  border_mode=0, p=0.7),
               A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
               A.HueSaturationValue(10, 15, 10, p=0.3)]
    tf += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(tf)


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


def make_dataset_class():
    from PIL import Image, ImageOps
    from torch.utils.data import Dataset

    class SegDS(Dataset):

        def __init__(self, pairs, mode):
            self.pairs, self.mode = pairs, mode
            self.tf = make_transform(augment=(mode == "train"))

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            stem, ip, mp = self.pairs[i]
            img = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
            mask = Image.open(mp)
            if mask.mode != "L":
                mask = mask.convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            img, mask = np.array(img), np.array(mask)
            out = self.tf(image=img, mask=mask)
            if self.mode == "test":
                return out["image"], out["mask"].long(), stem
            return out["image"], out["mask"].long()

    return SegDS


def build_model(arm, seed):
    import torch
    import segmentation_models_pytorch as smp
    seed_everything(seed)
    model = smp.Unet("resnet34", encoder_weights=None if arm == "scratch" else "imagenet",
                     in_channels=3, classes=NUM_CLASSES)
    if arm in ("maize", "control"):
        p = pre_dir(arm, seed) / "encoder.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} — run pretrain_cls.py --arm {arm} --seeds {seed} first")
        model.encoder.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
    return model


def train_one(arm, seed, out):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    SegDS = make_dataset_class()
    nw = min(4, os.cpu_count() or 2)

    model = build_model(arm, seed).to(device)

    gen = torch.Generator().manual_seed(seed)
    tr = DataLoader(SegDS(list_pairs("train"), "train"), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=nw, pin_memory=True,
                    worker_init_fn=worker_init, generator=gen)
    va = DataLoader(SegDS(list_pairs("val"), "val"), batch_size=BATCH, shuffle=False,
                    num_workers=nw, pin_memory=True)

    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=device))

    def dice_loss(logits, target, smooth=1.0):
        probs = F.softmax(logits, 1)
        oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        card = probs.sum(dims) + oh.sum(dims)
        return 1 - ((2 * inter + smooth) / (card + smooth)).mean()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best, since, best_epoch, epochs_run = -1.0, 0, 0, 0
    with open(out / "history.csv", "w", newline="") as hf:
        hw = csv.writer(hf)
        hw.writerow(["epoch", "lr", "train_loss", "val_lesion_dice_per_image", "val_select"]
                    + [f"val_dice_{c}_pooled" for c in CLASS_NAMES])
        for epoch in range(EPOCHS):
            lr = LR * (epoch + 1) / WARMUP if epoch < WARMUP else \
                0.5 * LR * (1 + np.cos(np.pi * (epoch - WARMUP) / max(1, EPOCHS - WARMUP)))
            for g in opt.param_groups:
                g["lr"] = lr

            model.train()
            run_loss, nb = 0.0, 0
            for x, y in tr:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=use_amp):
                    logits = model(x)
                    loss = 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item()
                nb += 1
            train_loss = run_loss / max(nb, 1)

            model.eval()
            inter, card = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
            per_image = []
            with torch.no_grad():
                for x, y in va:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device, enabled=use_amp):
                        pred = model(x).float().argmax(1)
                    for c in range(NUM_CLASSES):
                        p, t = (pred == c), (y == c)
                        inter[c] += (p & t).sum().item()
                        card[c] += p.sum().item() + t.sum().item()
                    for i in range(len(y)):
                        t, p = (y[i] == LESION), (pred[i] == LESION)
                        gt = t.sum().item()
                        if gt > 0:
                            per_image.append(2.0 * (t & p).sum().item() / (gt + p.sum().item()))
            dice = np.where(card > 0, 2 * inter / np.maximum(card, 1), np.nan)
            pi_dice = float(np.mean(per_image)) if per_image else float("nan")
            sel = pi_dice if SELECT_BY == "per_image" else float(dice[LESION])
            if sel != sel:
                sel = -1.0
            hw.writerow([epoch + 1, f"{lr:.6e}", f"{train_loss:.6f}",
                         f"{pi_dice:.6f}" if pi_dice == pi_dice else "", f"{sel:.6f}"]
                        + [f"{d:.6f}" if d == d else "" for d in dice])
            hf.flush()
            epochs_run = epoch + 1

            if sel > best:
                best, since, best_epoch = sel, 0, epoch + 1
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "val_lesion_dice": float(sel), "select_by": SELECT_BY},
                           out / "best.pt")
                tag = " *best*"
            else:
                since += 1
                tag = f" ({since}/{PATIENCE})"
            if epoch % 5 == 0 or tag == " *best*":
                print(f"    epoch {epoch + 1:>3}  loss {train_loss:.4f}  val lesion Dice "
                      f"({SELECT_BY}) {sel:.4f}{tag}", flush=True)
            if since >= PATIENCE:
                print(f"    early stop at epoch {epoch + 1}, best {best:.4f} (epoch {best_epoch})")
                break
    return {"best_epoch": best_epoch, "epochs_run": epochs_run,
            "best_val_lesion_dice": best, "selected_by": SELECT_BY}


def evaluate(arm, seed, out):
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SegDS = make_dataset_class()
    model = build_model(arm, seed).to(device)
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    dl = DataLoader(SegDS(list_pairs("test"), "test"), batch_size=1, shuffle=False,
                    num_workers=min(4, os.cpu_count() or 2))
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    with torch.no_grad():
        for x, y, stem in dl:
            pred = model(x.to(device)).float().argmax(1)[0].cpu().numpy()
            ref = y[0].numpy()
            conf += np.bincount(ref.ravel() * NUM_CLASSES + pred.ravel(),
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            t, p = ref == LESION, pred == LESION
            gt_px, pr_px = int(t.sum()), int(p.sum())
            inter = int(np.logical_and(t, p).sum())
            dice = 2.0 * inter / (gt_px + pr_px) if (gt_px + pr_px) > 0 else float("nan")
            leaf_gt = int((ref == LEAF).sum()) + gt_px
            leaf_pr = int((pred == LEAF).sum()) + pr_px
            rows.append({"stem": stem[0], "has_lesion": int(gt_px > 0),
                         "dice_lesion": dice, "lesion_px_gt": gt_px, "lesion_px_pred": pr_px,
                         "severity_gt": 100.0 * gt_px / leaf_gt if leaf_gt else float("nan"),
                         "severity_pred": 100.0 * pr_px / leaf_pr if leaf_pr else float("nan")})

    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "confusion.json").write_text(json.dumps(
        {"classes": CLASS_NAMES, "rows": "reference", "cols": "predicted",
         "counts": conf.tolist(), "resolution": SIZE}))

    d = np.array([r["dice_lesion"] for r in rows if r["has_lesion"]])
    pooled = 2.0 * conf[LESION, LESION] / (conf[LESION].sum() + conf[:, LESION].sum())
    m = {"arm": arm, "seed": seed, "n_test_images": len(rows),
         "n_with_lesion": int(len(d)), "mean_per_image_lesion_dice": float(np.nanmean(d)),
         "pooled_lesion_dice": float(pooled), "resolution": SIZE}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


def preflight():
    pairs = {s: list_pairs(s) for s in SPLIT_DIRS}
    print("segmentation data:", DATA_ROOT)
    for s, p in pairs.items():
        print(f"   {s:<6}{len(p):>5} image/mask pairs")
        if not p:
            sys.exit(f"!! no pairs for '{s}' under {DATA_ROOT / SPLIT_DIRS[s]} — "
                     f"check DATA_ROOT / SPLIT_DIRS / IMAGES_SUBDIR / MASKS_SUBDIR")
    from PIL import Image
    bad = set()
    for _, _, mp in pairs["train"][:5]:
        bad |= set(np.unique(np.array(Image.open(mp).convert("L")))) - {0, 1, 2}
    if bad:
        sys.exit(f"!! masks contain values {sorted(bad)} — expected only 0/1/2")


def run(arms, seeds, force):
    preflight()
    todo = [(a, s) for s in seeds for a in arms if force or not is_done(a, s)]
    if not todo:
        print("Nothing to do.")
        return
    print(f"\n{len(todo)} run(s) queued (seed-major, so each seed completes all arms):")
    for a, s in todo:
        print(f"   seed {s:<3} {a}")
    t_all = time.time()
    for i, (arm, seed) in enumerate(todo, 1):
        print("\n" + "=" * 70 + f"\n[{i}/{len(todo)}]  {arm}  seed {seed}\n" + "=" * 70, flush=True)
        out = seg_dir(arm, seed)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            info = train_one(arm, seed, out)
            m = evaluate(arm, seed, out)
            (out / "DONE").write_text(json.dumps(
                {**m, **info, "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
            print(f"\n  test per-image mean lesion Dice {m['mean_per_image_lesion_dice']:.4f}  "
                  f"(pooled {m['pooled_lesion_dice']:.4f})   {(time.time() - t0) / 60:.1f} min")
        except KeyboardInterrupt:
            print("\ninterrupted — finished runs are kept; re-run to continue")
            return
        except FileNotFoundError as e:
            print(f"\n  -- skipped: {e}")
        except Exception as e:
            import traceback
            print(f"\n  !! {arm} seed {seed} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nTotal {(time.time() - t_all) / 60:.0f} min")
    status(seeds)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nemenyi_p(mean_rank_a, mean_rank_b, k, n):
    from scipy import stats
    q = abs(mean_rank_a - mean_rank_b) / np.sqrt(k * (k + 1) / (12.0 * n))
    try:
        return float(stats.studentized_range.sf(q, k, np.inf))
    except Exception:
        return float(stats.studentized_range.sf(q, k, 1e7))


def seed_dice(arm, seed):
    import pandas as pd
    p = seg_dir(arm, seed) / "per_image.csv"
    if not (seg_dir(arm, seed) / "DONE").exists() or not p.exists():
        return None
    df = pd.read_csv(p)
    return float(df.loc[df["has_lesion"] == 1, "dice_lesion"].mean())


def summarize(seeds, posthoc_anyway, relevance):
    import pandas as pd
    from scipy import stats

    EXP.mkdir(parents=True, exist_ok=True)
    table = {a: {s: seed_dice(a, s) for s in seeds} for a in ARMS}
    df = pd.DataFrame(table)
    df.index.name = "seed"
    print("Per-image mean lesion Dice per seed (test set, 256 x 256):")
    print(df.round(4).to_string(na_rep="-"))
    df.to_csv(EXP / "arms_per_seed_dice.csv")

    full = df.dropna()
    n, k = len(full), len(ARMS)
    print(f"\nseeds with all four arms: {n}")
    if n < 3:
        print("Need at least 3 complete seeds for the Friedman test; stopping here.")
        return
    if n < 8:
        print(f"  (only {n} complete seeds: the exact Wilcoxon test cannot reach p<0.05 "
              f"below 6 seeds and has little power below ~8)")

    print("\narm                           mean     sd")
    for a in ARMS:
        print(f"  {ARM_TITLE[a]:<28}{full[a].mean():>7.4f}{full[a].std(ddof=1):>8.4f}")

    fr = stats.friedmanchisquare(*[full[a].to_numpy() for a in ARMS])
    ranks = np.vstack([stats.rankdata(-row) for row in full[ARMS].to_numpy()])
    mean_rank = dict(zip(ARMS, ranks.mean(0)))
    print(f"\nFriedman (seed-blocked): chi2({k - 1}) = {fr.statistic:.3f}, p = {fr.pvalue:.4g}")
    print("mean ranks (1 = best): " + ", ".join(f"{a} {mean_rank[a]:.2f}" for a in ARMS))

    if fr.pvalue >= 0.05 and not posthoc_anyway:
        print("Omnibus not significant -> post-hoc comparisons are not reported "
              "(--posthoc-anyway shows them as exploratory).")
    else:
        if fr.pvalue >= 0.05:
            print("!! omnibus NOT significant — everything below is exploratory.")
        rows = []
        pairs = [(a, b) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]]
        for a, b in pairs:
            if (b, a) in PLANNED or ((a, b) not in PLANNED):
                a, b = b, a
            x, y = full[a].to_numpy(), full[b].to_numpy()
            diff = x - y
            try:
                wp = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                wp = float("nan")
            t = stats.ttest_rel(x, y)
            ci = t.confidence_interval(0.95)
            planned = (a, b) in PLANNED
            rows.append({"contrast": f"{ARM_TITLE[a]} - {ARM_TITLE[b]}", "a": a, "b": b,
                         "planned": planned, "mean_diff": float(diff.mean()),
                         "ci_low": float(ci.low), "ci_high": float(ci.high),
                         "nemenyi_p": nemenyi_p(mean_rank[a], mean_rank[b], k, n),
                         "wilcoxon_p": wp})
        pl = [r for r in rows if r["planned"]]
        hp = holm([r["wilcoxon_p"] for r in pl])
        for r, h in zip(pl, hp):
            r["wilcoxon_p_holm"] = float(h)
        for r in rows:
            r.setdefault("wilcoxon_p_holm", float("nan"))
            r["relevant"] = bool(abs(r["mean_diff"]) > relevance and
                                 (r["ci_low"] > 0 or r["ci_high"] < 0))
        out = pd.DataFrame(rows)
        out.to_csv(EXP / "arms_pairwise.csv", index=False)
        print(f"\nPairwise (positive = first arm better). Planned contrasts carry the Holm "
              f"correction (over {len(pl)}); the other pairs are descriptive.")
        print(f"Relevance rule: |mean diff| > {relevance} AND 95% CI excludes 0 "
              f"(threshold is ad hoc).")
        print(f"{'contrast':<56}{'diff':>8}{'95% CI':>20}{'Nemenyi':>9}{'Wilcox':>9}"
              f"{'Holm':>8}  rel")
        for r in rows:
            print(f"{r['contrast'] + (' *' if r['planned'] else ''):<56}{r['mean_diff']:>+8.4f}"
                  f"{'[%+.4f, %+.4f]' % (r['ci_low'], r['ci_high']):>20}"
                  f"{r['nemenyi_p']:>9.4f}{r['wilcoxon_p']:>9.4f}"
                  f"{r['wilcoxon_p_holm']:>8.4f}  {'yes' if r['relevant'] else 'no'}")
        print("  * planned contrast")

    pretrain_summary()
    confusion_summary(seeds)


def pretrain_summary():
    print("\nPretraining sanity check (held-out accuracy, last epoch):")
    for arm in ("maize", "control"):
        accs = []
        for s in SEEDS_DEFAULT:
            p = pre_dir(arm, s) / "metrics.json"
            if p.exists():
                accs.append(json.loads(p.read_text())["final_heldout_acc"])
        if accs:
            print(f"  {arm:<8} n={len(accs)}  mean {np.mean(accs):.4f}  "
                  f"min {np.min(accs):.4f}  max {np.max(accs):.4f}")
        else:
            print(f"  {arm:<8} no pretraining metrics found")


def confusion_summary(seeds):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, sd, nused = {}, {}, {}
    for a in ARMS:
        mats = []
        for s in seeds:
            p = seg_dir(a, s) / "confusion.json"
            if not ((seg_dir(a, s) / "DONE").exists() and p.exists()):
                continue
            c = np.array(json.loads(p.read_text())["counts"], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mats.append(c / c.sum(1, keepdims=True))
        if mats:
            m = np.stack(mats)
            mean[a] = np.nanmean(m, 0)
            sd[a] = np.nanstd(m, 0, ddof=1) if len(m) > 1 \
                else np.zeros((NUM_CLASSES, NUM_CLASSES))
            nused[a] = len(m)
    if not mean:
        print("\nNo confusion matrices found.")
        return

    print("\nAverage confusion matrix per arm (row-normalised per seed, then averaged;"
          "\nrows = reference class, cols = predicted; pixels pooled over the test set —"
          "\ndescriptive, the per-image Dice carries the inference):")
    long_rows = []
    for a in mean:
        print(f"\n  {ARM_TITLE[a]}   (mean over {nused[a]} seeds; ± = sd across seeds)")
        print(f"  {'':<12}" + "".join(f"{c:>18}" for c in CLASS_NAMES))
        for i, rc in enumerate(CLASS_NAMES):
            print(f"  {rc:<12}" + "".join(f"{mean[a][i, j]:>10.4f} ±{sd[a][i, j]:.4f}"
                                          for j in range(NUM_CLASSES)))
            for j, pc in enumerate(CLASS_NAMES):
                long_rows.append({"arm": a, "reference": rc, "predicted": pc,
                                  "mean": mean[a][i, j], "sd": sd[a][i, j], "n_seeds": nused[a]})
    pd.DataFrame(long_rows).to_csv(EXP / "avg_confusion_matrices.csv", index=False)

    for a, b in [("imagenet", "scratch"), ("maize", "imagenet"), ("control", "imagenet")]:
        if a in mean and b in mean:
            d = mean[a] - mean[b]
            print(f"\n  difference  {ARM_TITLE[a]} - {ARM_TITLE[b]}  (positive = a assigns more)")
            for i, rc in enumerate(CLASS_NAMES):
                print(f"  {rc:<12}" + "".join(f"{d[i, j]:>+12.4f}" for j in range(NUM_CLASSES)))

    arms = [a for a in ARMS if a in mean]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms) + 0.8, 4.3),
                             facecolor="white", squeeze=False)
    for pi, (ax, a) in enumerate(zip(axes[0], arms)):
        im = ax.imshow(mean[a], cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = mean[a][i, j]
                ax.text(j, i, f"{v:.3f}\n±{sd[a][i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if v > 0.5 else "#1f2937")
        ax.set_xticks(range(NUM_CLASSES), ["bg", "leaf", "lesion"], fontsize=9)
        ax.set_yticks(range(NUM_CLASSES), ["bg", "leaf", "lesion"], fontsize=9)
        if pi > 0:
            ax.tick_params(labelleft=False, left=False)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_title(f"{ARM_TITLE[a]}\n(n = {nused[a]} seeds)", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0][0].set_ylabel("Reference", fontsize=10)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.75, pad=0.02)
    cb.set_label("share of reference pixels", fontsize=9)
    fig.suptitle("Average pixel confusion matrix, test set (row-normalised, mean over seeds)",
                 fontsize=11)
    fig.savefig(EXP / "avg_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {EXP / 'avg_confusion_matrices.png'}, avg_confusion_matrices.csv, "
          f"arms_per_seed_dice.csv, arms_pairwise.csv")


def main():
    global EXP, DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--exp", default=str(EXP), help="experiment root (pretrain/ and seg/ live here)")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--posthoc-anyway", action="store_true",
                    help="show post-hoc tests even if the Friedman test is not significant")
    ap.add_argument("--relevance", type=float, default=RELEVANCE)
    args, _ = ap.parse_known_args()

    EXP, DATA_ROOT = Path(args.exp), Path(args.data_root)
    if args.status:
        status(args.seeds)
    elif args.summarize:
        summarize(args.seeds, args.posthoc_anyway, args.relevance)
    else:
        run(args.arm, args.seeds, args.force)


if __name__ == "__main__":
    main()
