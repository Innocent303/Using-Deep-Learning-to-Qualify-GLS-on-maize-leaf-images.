import csv
import json
import sys
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
    list_pairs, eval_transform, seed_everything,
)


CLASS_NAMES = ["background", "leaf", "GLS_lesion"]

assert len(CLASS_NAMES) == NUM_CLASSES, (
    f"CLASS_NAMES has {len(CLASS_NAMES)} entries but NUM_CLASSES is {NUM_CLASSES}"
)

SPLIT = "test"

LEAF_CLASS = 1
LESION_CLASS = 2
_RUN_DIR = ROOT / "runs" / RUN_NAME
CHECKPOINT = _RUN_DIR / "best.pt"
OUT_DIR = _RUN_DIR / f"eval_{SPLIT}"
BATCH_SIZE = 4
NUM_WORKERS = 4
N_WORST = 8


def autocast_ctx():
    try:
        return torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"))


@torch.no_grad()
def evaluate(model, pairs):
    ds = SegDataset(pairs, eval_transform(IMG_SIZE))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image = []
    severity = []
    idx = 0

    model.eval()
    for x, y in dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast_ctx():
            logits = model(x)
        pred = logits.float().argmax(1)

        t_flat = y.view(-1)
        p_flat = pred.view(-1)
        k = t_flat * NUM_CLASSES + p_flat
        counts = torch.bincount(k, minlength=NUM_CLASSES ** 2)
        cm += counts.view(NUM_CLASSES, NUM_CLASSES).cpu().numpy()

        for b in range(x.size(0)):
            cm_i = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            tf = y[b].view(-1)
            pf = pred[b].view(-1)
            k = tf * NUM_CLASSES + pf
            cm_i += torch.bincount(k, minlength=NUM_CLASSES ** 2).view(
                NUM_CLASSES, NUM_CLASSES).cpu().numpy()
            dices, _, _, _ = metrics_from_cm(cm_i)
            per_image.append((idx, float(np.nanmean(dices)), list(dices), cm_i))

            t_les = float((y[b] == LESION_CLASS).sum())
            t_leaf = float((y[b] == LEAF_CLASS).sum())
            p_les = float((pred[b] == LESION_CLASS).sum())
            p_leaf = float((pred[b] == LEAF_CLASS).sum())
            t_tot, p_tot = t_les + t_leaf, p_les + p_leaf
            severity.append((
                idx,
                100.0 * t_les / t_tot if t_tot > 0 else float("nan"),
                100.0 * p_les / p_tot if p_tot > 0 else float("nan"),
            ))
            idx += 1

    return cm, per_image, severity


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    pred_tot = cm.sum(axis=0).astype(np.float64)
    true_tot = cm.sum(axis=1).astype(np.float64)
    fp = pred_tot - tp
    fn = true_tot - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, np.nan)
        recall = np.where(true_tot > 0, tp / true_tot, np.nan)
        dice = np.where((pred_tot + true_tot) > 0, 2 * tp / (pred_tot + true_tot), np.nan)
        iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
    return dice, iou, precision, recall


def print_cm(cm):
    w = max(9, max(len(n) for n in CLASS_NAMES) + 2)
    tot = cm.sum()

    print("\nCONFUSION MATRIX — pixel counts")
    print("rows = ground truth, columns = prediction\n")
    head = " " * (w + 2) + "".join(f"{n:>{w}}" for n in CLASS_NAMES)
    print(head)
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:>{w}}  " + "".join(f"{cm[i, j]:>{w},}" for j in range(NUM_CLASSES)))

    print("\nROW-NORMALISED (recall): of all TRUE class X pixels, "
          "what fraction went where")
    print(head)
    for i, n in enumerate(CLASS_NAMES):
        row = cm[i].sum()
        cells = "".join(f"{(100*cm[i, j]/row if row else 0):>{w-1}.1f}%"
                        for j in range(NUM_CLASSES))
        print(f"{n:>{w}}  " + cells)

    print("\nCOLUMN-NORMALISED (precision): of all pixels PREDICTED class X, "
          "what fraction were correct")
    print(head)
    for i, n in enumerate(CLASS_NAMES):
        cells = "".join(f"{(100*cm[i, j]/cm[:, j].sum() if cm[:, j].sum() else 0):>{w-1}.1f}%"
                        for j in range(NUM_CLASSES))
        print(f"{n:>{w}}  " + cells)

    print(f"\nOverall pixel accuracy: {100*np.diag(cm).sum()/tot:.2f}%")


def diagnose(cm, dice):
    print("\n" + "=" * 60)
    print("READING THE MATRIX")
    worst = int(np.nanargmin(dice))
    print(f"\nWeakest class: {CLASS_NAMES[worst]} (Dice {dice[worst]:.3f})")

    row = cm[worst].astype(float)
    if row.sum() > 0:
        conf = [(j, row[j] / row.sum()) for j in range(NUM_CLASSES) if j != worst]
        conf.sort(key=lambda t: -t[1])
        j, frac = conf[0]
        print(f"  Missed pixels mostly become '{CLASS_NAMES[j]}' "
              f"({100*frac:.1f}% of its true pixels)")

    col = cm[:, worst].astype(float)
    if col.sum() > 0:
        false_pos = [(i, col[i] / col.sum()) for i in range(NUM_CLASSES) if i != worst]
        false_pos.sort(key=lambda t: -t[1])
        i, frac = false_pos[0]
        print(f"  False positives mostly come from '{CLASS_NAMES[i]}' "
              f"({100*frac:.1f}% of predictions for this class)")

    tp = cm[worst, worst]
    fn = cm[worst].sum() - tp
    fp = cm[:, worst].sum() - tp
    print(f"\n  TP {tp:,}   FN {fn:,}   FP {fp:,}")
    if fn > fp * 1.5:
        print("  -> UNDER-predicting: the model is too conservative here.")
        print("     Raise this class's weight, or try Tversky loss with beta > alpha.")
    elif fp > fn * 1.5:
        print("  -> OVER-predicting: too eager, bleeding into other classes.")
        print("     Lower its weight, or check for annotation noise.")
    else:
        print("  -> Balanced errors: likely boundary imprecision, not bias.")
        print("     Higher resolution or more training usually helps most.")


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT} — run Phase 3 first.")

    ck = _torch_load(CHECKPOINT, DEVICE)
    model = build_unet().to(DEVICE)
    model.load_state_dict(ck["model"])
    print(f"Loaded {CHECKPOINT.name} from epoch {ck.get('epoch', '?') + 1}, "
          f"training mean Dice {ck.get('mean_dice', float('nan')):.4f}")

    pairs = list_pairs(SPLIT)
    if not pairs:
        raise SystemExit(f"No pairs found in split '{SPLIT}'.")
    print(f"Evaluating {len(pairs)} image(s) from '{SPLIT}' at {IMG_SIZE}px\n")

    cm, per_image, severity = evaluate(model, pairs)
    dice, iou, precision, recall = metrics_from_cm(cm)

    sev_idx = [i for i, t, p in severity]
    sev_true = np.array([t for _, t, _ in severity])
    sev_pred = np.array([p for _, _, p in severity])
    ok = np.isfinite(sev_true) & np.isfinite(sev_pred)

    print("\n" + "=" * 62)
    print("DISEASE SEVERITY  —  lesion / (lesion + leaf) * 100")
    print("=" * 62)
    if ok.sum() < 2:
        print("Not enough images with leaf tissue to assess severity.")
        sev_stats = {}
    else:
        t, p = sev_true[ok], sev_pred[ok]
        err = p - t
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r = float(np.corrcoef(t, p)[0, 1])
        ss_res = float(np.sum((p - t) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        print(f"  images assessed:        {ok.sum()}")
        print(f"  true severity range:    {t.min():.2f}% to {t.max():.2f}%  "
              f"(mean {t.mean():.2f}%)")
        print(f"  predicted range:        {p.min():.2f}% to {p.max():.2f}%  "
              f"(mean {p.mean():.2f}%)")
        print(f"\n  MAE:                    {mae:.2f} percentage points")
        print(f"  RMSE:                   {rmse:.2f} pp")
        print(f"  bias (pred - true):     {bias:+.2f} pp")
        print(f"  Pearson r:              {r:.4f}")
        print(f"  R^2 vs 1:1 line:        {r2:.4f}")

        within = [1, 2, 5]
        print()
        for w in within:
            print(f"  within +/-{w} pp:          "
                  f"{100*np.mean(np.abs(err) <= w):.1f}% of images")

        if bias < -0.5:
            print(f"\n  -> Systematically UNDER-estimates severity by {abs(bias):.2f} pp.")
            print("     Consistent with under-predicting the lesion class.")
        elif bias > 0.5:
            print(f"\n  -> Systematically OVER-estimates severity by {bias:.2f} pp.")
        else:
            print(f"\n  -> Essentially unbiased ({bias:+.2f} pp). Errors are random,")
            print("     so mean severity across many leaves stays accurate.")

        if r > 0.9 and mae > 3:
            print("     High correlation with high MAE suggests a scale offset —")
            print("     a linear calibration on val could reduce the error.")

        sev_stats = {"mae_pp": mae, "rmse_pp": rmse, "bias_pp": bias,
                     "pearson_r": r, "r2_vs_identity": r2,
                     "n_images": int(ok.sum()),
                     "mean_true_pct": float(t.mean()),
                     "mean_pred_pct": float(p.mean())}

        worst_sev = sorted(
            [(sev_idx[i], sev_true[i], sev_pred[i]) for i in range(len(severity))
             if ok[i]], key=lambda r_: -abs(r_[2] - r_[1]))[:5]
        print("\n  Largest severity errors:")
        for i, tv, pv in worst_sev:
            print(f"    {pairs[i][0].name:<24} true {tv:6.2f}%  "
                  f"pred {pv:6.2f}%  ({pv - tv:+.2f} pp)")

    print("\n" + "=" * 62)

    print("PER-CLASS METRICS (over all pixels in the split)")
    print(f"{'class':>12} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'pixels':>14}")
    print("-" * 62)
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:>12} {dice[c]:>8.4f} {iou[c]:>8.4f} "
              f"{precision[c]:>8.4f} {recall[c]:>8.4f} {cm[c].sum():>14,}")
    print("-" * 62)
    print(f"{'mean':>12} {np.nanmean(dice):>8.4f} {np.nanmean(iou):>8.4f}")

    print_cm(cm)
    diagnose(cm, dice)

    per_image.sort(key=lambda t: t[1])
    LEAF_I, LESION_I = 1, 2
    sev_map = {i: (t, p) for i, t, p in severity}
    csv_path = OUT_DIR / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "model", "split", "mean_dice", "mean_iou"]
                   + [f"dice_{n}" for n in CLASS_NAMES]
                   + [f"iou_{n}" for n in CLASS_NAMES]
                   + ["true_lesion_px", "pred_lesion_px", "true_leaf_px",
                      "has_lesion", "severity_true_pct", "severity_pred_pct",
                      "severity_error_pp"])
        label = ("U-Net (scratch)" if PRETRAINED_ENCODER is None
                 else "U-Net (ImageNet)")
        for i, m, ds, cm_i in per_image:
            st, sp = sev_map.get(i, (float("nan"), float("nan")))
            d_i, i_i, _, _ = metrics_from_cm(cm_i)

            def fm(v):
                return "" if (v is None or (isinstance(v, float) and v != v)) \
                    else f"{v:.6f}"

            w.writerow([pairs[i][0].name, label, SPLIT,
                        fm(float(np.nanmean(d_i))), fm(float(np.nanmean(i_i)))]
                       + [fm(x) for x in d_i] + [fm(x) for x in i_i]
                       + [int(cm_i[LESION_I].sum()), int(cm_i[:, LESION_I].sum()),
                          int(cm_i[LEAF_I].sum()), int(cm_i[LESION_I].sum() > 0),
                          fm(st), fm(sp), fm(sp - st)])

    print(f"\nWORST {N_WORST} IMAGES")
    for i, m, ds, _cm in per_image[:N_WORST]:
        detail = "  ".join(f"{n} {d:.3f}" for n, d in zip(CLASS_NAMES, ds))
        print(f"  {pairs[i][0].name:<24} mean {m:.3f}   [{detail}]")

    scores = np.array([m for _, m, _, _ in per_image])
    print(f"\nPer-image mean Dice: median {np.nanmedian(scores):.3f}, "
          f"worst {np.nanmin(scores):.3f}, best {np.nanmax(scores):.3f}")
    n_bad = int((scores < 0.5).sum())
    if n_bad:
        print(f"  {n_bad} image(s) below 0.5 — inspect these for bad annotations.")

    json.dump(
        {"split": SPLIT, "confusion_matrix": cm.tolist(),
         "dice": dict(zip(CLASS_NAMES, dice.tolist())),
         "iou": dict(zip(CLASS_NAMES, iou.tolist())),
         "precision": dict(zip(CLASS_NAMES, precision.tolist())),
         "recall": dict(zip(CLASS_NAMES, recall.tolist())),
         "mean_dice": float(np.nanmean(dice)),
         "mean_iou": float(np.nanmean(iou)),
         "severity": sev_stats,
         "pixel_accuracy": float(np.diag(cm).sum() / cm.sum())},
        open(OUT_DIR / "metrics.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps

        fig, ax = plt.subplots(figsize=(6, 5))
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix ({SPLIT}, recall-normalised)")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{100*norm[i,j]:.1f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=10)
        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
        plt.close()

        true_px = cm.sum(axis=1).astype(float)
        pred_px = cm.sum(axis=0).astype(float)
        total = true_px.sum()

        fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 4.8))
        xpos = np.arange(NUM_CLASSES)
        width = 0.38
        colours = ["#888888", "#2e8b57", "#c1440e"]

        axl.bar(xpos - width / 2, true_px, width, label="ground truth",
                color=colours, edgecolor="black", linewidth=0.6)
        axl.bar(xpos + width / 2, pred_px, width, label="predicted",
                color=colours, alpha=0.45, edgecolor="black",
                linewidth=0.6, hatch="//")
        axl.set_yscale("log")
        axl.set_xticks(xpos, CLASS_NAMES)
        axl.set_ylabel("pixels (log scale)")
        axl.set_title(f"Class imbalance — {SPLIT} split")
        axl.legend()
        for i in range(NUM_CLASSES):
            axl.text(i - width / 2, true_px[i], f" {100*true_px[i]/total:.1f}%",
                     ha="center", va="bottom", fontsize=8, rotation=90)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(true_px > 0, pred_px / true_px, np.nan)
        bars = axr.bar(xpos, ratio, 0.55, color=colours,
                       edgecolor="black", linewidth=0.6)
        axr.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axr.set_xticks(xpos, CLASS_NAMES)
        axr.set_ylabel("predicted pixels / true pixels")
        axr.set_title("Volume bias  (1.0 = correct amount)")
        top = float(np.nanmax(ratio)) if np.isfinite(ratio).any() else 1.0
        axr.set_ylim(0, max(1.4, top * 1.25))
        for b, r in zip(bars, ratio):
            axr.text(b.get_x() + b.get_width() / 2, r,
                     f"{r:.2f}x", ha="center", va="bottom", fontsize=9)
        axr.text(0.02, 0.95, "below 1.0 = under-predicting\nabove 1.0 = over-predicting",
                 transform=axr.transAxes, fontsize=8, va="top", color="#444444")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "class_balance.png", dpi=120)
        plt.close()

        print("\nPIXEL COUNTS")
        print(f"{'class':>12} {'true':>14} {'% of total':>11} "
              f"{'predicted':>14} {'pred/true':>10}")
        print("-" * 66)
        for c, name in enumerate(CLASS_NAMES):
            print(f"{name:>12} {int(true_px[c]):>14,} "
                  f"{100*true_px[c]/total:>10.2f}% {int(pred_px[c]):>14,} "
                  f"{ratio[c]:>9.2f}x")
        rarest = int(np.argmin(true_px))
        print(f"\nImbalance ratio (largest : smallest class) = "
              f"{true_px.max()/max(true_px.min(), 1):.0f} : 1")
        if ratio[rarest] < 0.85:
            print(f"  {CLASS_NAMES[rarest]} is under-predicted by volume "
                  f"({ratio[rarest]:.2f}x) — a classic imbalance symptom.")
        elif ratio[rarest] > 1.15:
            print(f"  {CLASS_NAMES[rarest]} is over-predicted by volume "
                  f"({ratio[rarest]:.2f}x) — class weight may be too high.")

        if ok.sum() >= 2:
            t, p = sev_true[ok], sev_pred[ok]
            err = p - t
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))

            lim = max(t.max(), p.max()) * 1.08 + 0.5
            a1.plot([0, lim], [0, lim], "k--", linewidth=1.2,
                    label="perfect agreement")
            a1.scatter(t, p, s=42, alpha=0.75, color="#c1440e",
                       edgecolor="black", linewidth=0.5)
            a1.set_xlim(0, lim); a1.set_ylim(0, lim)
            a1.set_xlabel("true severity (%)")
            a1.set_ylabel("predicted severity (%)")
            a1.set_title(f"GLS severity agreement — {SPLIT}")
            a1.legend(loc="upper left", fontsize=9)
            a1.text(0.98, 0.04,
                    f"MAE {np.mean(np.abs(err)):.2f} pp\n"
                    f"bias {np.mean(err):+.2f} pp\n"
                    f"r {np.corrcoef(t, p)[0,1]:.3f}",
                    transform=a1.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="#999999"))

            a2.axhline(0, color="black", linestyle="--", linewidth=1.2)
            a2.axhline(np.mean(err), color="#c1440e", linewidth=1,
                       label=f"mean bias {np.mean(err):+.2f} pp")
            sd = np.std(err)
            a2.fill_between([0, lim], np.mean(err) - 1.96 * sd,
                            np.mean(err) + 1.96 * sd, color="#c1440e", alpha=0.12,
                            label="95% of errors")
            a2.scatter(t, err, s=42, alpha=0.75, color="#2e8b57",
                       edgecolor="black", linewidth=0.5)
            a2.set_xlim(0, lim)
            a2.set_xlabel("true severity (%)")
            a2.set_ylabel("error, predicted - true (pp)")
            a2.set_title("Severity error vs disease level")
            a2.legend(fontsize=9)

            plt.tight_layout()
            plt.savefig(OUT_DIR / "severity_agreement.png", dpi=120)
            plt.close()

        picks = per_image[:N_WORST]
        fig, axes = plt.subplots(3, len(picks), figsize=(3.2 * len(picks), 9.5))
        if len(picks) == 1:
            axes = axes.reshape(3, 1)
        ds_eval = SegDataset(pairs, eval_transform(IMG_SIZE))
        for col, (i, m, *_r) in enumerate(picks):
            img = ImageOps.exif_transpose(Image.open(pairs[i][0])).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x, y = ds_eval[i]
            with torch.no_grad(), autocast_ctx():
                pr = model(x.unsqueeze(0).to(DEVICE)).float().argmax(1)[0].cpu().numpy()
            axes[0, col].imshow(img); axes[0, col].set_title(
                f"{pairs[i][0].name}\nDice {m:.3f}", fontsize=8)
            axes[1, col].imshow(y.numpy(), cmap="tab10", vmin=0, vmax=9)
            axes[1, col].set_title("ground truth", fontsize=8)
            axes[2, col].imshow(pr, cmap="tab10", vmin=0, vmax=9)
            axes[2, col].set_title("prediction", fontsize=8)
            for r in range(3):
                axes[r, col].axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "worst_cases.png", dpi=100)
        plt.close()
        print(f"\nFigures: {OUT_DIR}")
    except Exception as e:
        print(f"(figures skipped: {type(e).__name__}: {e})")

    print(f"Per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
