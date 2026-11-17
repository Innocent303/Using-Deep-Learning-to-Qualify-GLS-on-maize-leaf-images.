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


_HERE = _find_module("maskrcnn_pipeline")
if _HERE is None:
    raise RuntimeError(
        "Could not find maskrcnn_pipeline.py.\n"
        "Put it next to this script, then:  %cd /content/drive/MyDrive\n"
        f"Searched from: {Path.cwd()}\n"
        "Check with:  !find /content/drive/MyDrive -name maskrcnn_pipeline.py"
    )
print(f"Loading pipeline from {_HERE}")

from maskrcnn_pipeline import (
    ROOT, DEVICE, RUN_NAME, CocoInstanceDataset, build_model, collate_fn,
    seed_everything,
)
from pycocotools import mask as maskutils


SPLIT = "test"

FIXED_THRESHOLD = 0.60

CHECKPOINT = ROOT / "runs" / RUN_NAME / "best.pt"
OUT_DIR = ROOT / "runs" / RUN_NAME / f"compare_{SPLIT}"

UNET_METRICS = Path("/content/drive/MyDrive/unet/runs/phase3") / f"eval_{SPLIT}" / "metrics.json"

BACKGROUND, LEAF, LESION = 0, 1, 2
CLASS_NAMES = ["background", "leaf", "GLS_lesion"]
NUM_CLASSES = 3

CAT_TO_SEMANTIC = {1: LEAF, 2: LESION}

SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

MASK_BINARISE = 0.5
BATCH_SIZE = 4
NUM_WORKERS = 4


def autocast_ctx():
    try:
        return torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"))


def gt_semantic_map(coco, img_id, h, w):
    sem = np.zeros((h, w), dtype=np.uint8)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    for target_class in (LEAF, LESION):
        for a in anns:
            if CAT_TO_SEMANTIC.get(a["category_id"]) != target_class:
                continue
            m = coco.annToMask(a)
            if m.shape != (h, w):
                continue
            sem[m > 0] = target_class
    return sem


def pred_semantic_map(out, h, w, score_thr):
    sem = np.zeros((h, w), dtype=np.uint8)
    scores = out["scores"].float().cpu().numpy()
    labels = out["labels"].cpu().numpy()
    masks = (out["masks"].float() > MASK_BINARISE).cpu().numpy()

    for target_class in (LEAF, LESION):
        for i in range(len(scores)):
            if scores[i] < score_thr:
                continue
            if CAT_TO_SEMANTIC.get(int(labels[i])) != target_class:
                continue
            m = masks[i, 0]
            if m.shape != (h, w):
                continue
            sem[m] = target_class
    return sem


def confusion(gt, pred, n=NUM_CLASSES):
    k = gt.astype(np.int64) * n + pred.astype(np.int64)
    return np.bincount(k.ravel(), minlength=n * n).reshape(n, n)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    pred_tot, true_tot = cm.sum(0).astype(np.float64), cm.sum(1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(pred_tot + true_tot > 0, 2 * tp / (pred_tot + true_tot), np.nan)
        iou = np.where(tp + (pred_tot - tp) + (true_tot - tp) > 0,
                       tp / (tp + (pred_tot - tp) + (true_tot - tp)), np.nan)
        prec = np.where(pred_tot > 0, tp / pred_tot, np.nan)
        rec = np.where(true_tot > 0, tp / true_tot, np.nan)
    return dice, iou, prec, rec


def severity(sem):
    les = int((sem == LESION).sum())
    leaf = int((sem == LEAF).sum())
    tot = les + leaf
    return 100.0 * les / tot if tot > 0 else float("nan")


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT} — train first.")

    ann_file = ROOT / "annotations" / f"instances_{SPLIT}.json"
    ds = CocoInstanceDataset(ROOT / SPLIT, ann_file, train=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate_fn)

    ck = _torch_load(CHECKPOINT, DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded epoch {ck.get('epoch', -1) + 1}, "
          f"val segm mAP {ck.get('segm_map', float('nan')):.4f}")
    print(f"Evaluating {len(ds)} image(s) from '{SPLIT}'\n")

    cms = {t: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
           for t in SCORE_THRESHOLDS}
    sev = {t: [] for t in SCORE_THRESHOLDS}
    per_image = {t: [] for t in SCORE_THRESHOLDS}
    missing_leaf = {t: 0 for t in SCORE_THRESHOLDS}
    names = []

    with torch.no_grad():
        for images, targets in dl:
            imgs_d = [i.to(DEVICE) for i in images]
            with autocast_ctx():
                outputs = model(imgs_d)

            for img, tgt, out in zip(images, targets, outputs):
                h, w = img.shape[-2:]
                img_id = int(tgt["image_id"].item())
                names.append(ds.coco.loadImgs(img_id)[0]["file_name"])

                gt = gt_semantic_map(ds.coco, img_id, h, w)
                gt_sev = severity(gt)

                for t in SCORE_THRESHOLDS:
                    pr = pred_semantic_map(out, h, w, t)
                    cms[t] += confusion(gt, pr)
                    p_sev = severity(pr)
                    sev[t].append((gt_sev, p_sev))
                    per_image[t].append(confusion(gt, pr))
                    if (pr == LEAF).sum() == 0 and (pr == LESION).sum() == 0:
                        missing_leaf[t] += 1

    print("SCORE THRESHOLD SWEEP")
    print(f"{'thr':>6} {'mean Dice':>11} {'leaf':>8} {'lesion':>8} "
          f"{'sev MAE':>9} {'sev bias':>9} {'no plant':>9}")
    print("-" * 68)

    tuned_t, best_score = None, -1.0
    rows = {}
    for t in SCORE_THRESHOLDS:
        dice, iou, prec, rec = metrics_from_cm(cms[t])
        pairs = np.array(sev[t], dtype=float)
        ok = np.isfinite(pairs).all(axis=1)
        if ok.sum():
            err = pairs[ok, 1] - pairs[ok, 0]
            mae, bias = float(np.mean(np.abs(err))), float(np.mean(err))
        else:
            mae = bias = float("nan")

        score = dice[LESION]
        rows[t] = (dice, iou, prec, rec, mae, bias)
        print(f"{t:>6.2f} {np.nanmean(dice):>11.4f} {dice[LEAF]:>8.4f} "
              f"{dice[LESION]:>8.4f} {mae:>9.2f} {bias:>+9.2f} "
              f"{missing_leaf[t]:>9}")
        if np.isfinite(score) and score > best_score:
            best_score, tuned_t = score, t

    if FIXED_THRESHOLD is not None:
        if FIXED_THRESHOLD not in rows:
            raise SystemExit(
                f"FIXED_THRESHOLD {FIXED_THRESHOLD} is not in SCORE_THRESHOLDS "
                f"{SCORE_THRESHOLDS} — add it to the sweep list.")
        best_t = FIXED_THRESHOLD
        print(f"\nUsing FIXED threshold {best_t:.2f} (selected on val).")
        print(f"  Sweep above is shown for information only; the reported")
        print(f"  numbers use {best_t:.2f} and were not tuned on this split.")
        if best_t != tuned_t:
            print(f"  (best on THIS split would have been {tuned_t:.2f} — "
                  f"the gap is the optimism you just avoided.)")
    else:
        best_t = tuned_t
        print(f"\nBest lesion Dice at score threshold {best_t:.2f}")
        if SPLIT == "test":
            print("  !! WARNING: threshold was tuned ON THE TEST SPLIT.")
            print("     This inflates the result. Set FIXED_THRESHOLD to the")
            print("     value chosen on val and re-run.")
        else:
            print("  (chosen on this split — keep it fixed for test)")

    dice, iou, prec, rec, mae, bias = rows[best_t]
    cm = cms[best_t]

    print(f"\nMASK R-CNN as semantic segmentation  (threshold {best_t:.2f})")
    print(f"{'class':>12} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 46)
    for c, n in enumerate(CLASS_NAMES):
        print(f"{n:>12} {dice[c]:>8.4f} {iou[c]:>8.4f} {prec[c]:>8.4f} {rec[c]:>8.4f}")
    print(f"{'mean':>12} {np.nanmean(dice):>8.4f} {np.nanmean(iou):>8.4f}")

    print("\nCONFUSION MATRIX (rows = truth, cols = prediction, % of true class)")
    w = 13
    print(" " * (w + 2) + "".join(f"{n:>{w}}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        r = cm[i].sum()
        print(f"{n:>{w}}  " + "".join(
            f"{(100*cm[i,j]/r if r else 0):>{w-1}.1f}%" for j in range(NUM_CLASSES)))

    pairs = np.array(sev[best_t], dtype=float)
    ok = np.isfinite(pairs).all(axis=1)
    t_s, p_s = pairs[ok, 0], pairs[ok, 1]
    err = p_s - t_s
    r = float(np.corrcoef(t_s, p_s)[0, 1]) if ok.sum() > 1 else float("nan")

    print(f"\nGLS SEVERITY  lesion/(lesion+leaf)*100")
    print(f"  images:      {int(ok.sum())}")
    print(f"  true mean:   {t_s.mean():.2f}%    predicted mean: {p_s.mean():.2f}%")
    print(f"  MAE:         {np.mean(np.abs(err)):.2f} pp")
    print(f"  bias:        {np.mean(err):+.2f} pp")
    print(f"  Pearson r:   {r:.4f}")
    for wpp in (1, 2, 5):
        print(f"  within +/-{wpp} pp: {100*np.mean(np.abs(err) <= wpp):.1f}%")

    csv_path = OUT_DIR / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "model", "split", "mean_dice", "mean_iou"]
                   + [f"dice_{c}" for c in CLASS_NAMES]
                   + [f"iou_{c}" for c in CLASS_NAMES]
                   + ["true_lesion_px", "pred_lesion_px", "true_leaf_px",
                      "has_lesion", "severity_true_pct", "severity_pred_pct",
                      "severity_error_pp"])
        for nm, c, (gs, ps) in zip(names, per_image[best_t], sev[best_t]):
            c = np.asarray(c)
            d_i, i_i, _, _ = metrics_from_cm(c)
            e = ps - gs if np.isfinite([gs, ps]).all() else float("nan")

            def fm(v):
                return "" if (v is None or (isinstance(v, float) and v != v)) \
                    else f"{v:.6f}"

            w.writerow([nm, "Mask R-CNN", SPLIT,
                        fm(float(np.nanmean(d_i))), fm(float(np.nanmean(i_i)))]
                       + [fm(x) for x in d_i] + [fm(x) for x in i_i]
                       + [int(c[LESION].sum()), int(c[:, LESION].sum()),
                          int(c[LEAF].sum()), int(c[LESION].sum() > 0),
                          fm(gs), fm(ps), fm(e)])

    print(f"Per-image: {OUT_DIR / 'per_image.csv'}")

    result = {
        "split": SPLIT, "score_threshold": best_t,
        "confusion_matrix": cm.tolist(),
        "dice": dict(zip(CLASS_NAMES, dice.tolist())),
        "iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "mean_dice": float(np.nanmean(dice)),
        "mean_iou": float(np.nanmean(iou)),
        "severity": {"mae_pp": float(np.mean(np.abs(err))),
                     "bias_pp": float(np.mean(err)),
                     "pearson_r": r, "n_images": int(ok.sum())},
        "instance_segm_mAP": ck.get("segm_map"),
    }
    (OUT_DIR / "maskrcnn_semantic_metrics.json").write_text(json.dumps(result, indent=2))

    if not UNET_METRICS.exists():
        print(f"\n(no U-Net metrics at {UNET_METRICS} — run phase4_evaluate.py "
              f"on '{SPLIT}' for the comparison)")
    else:
        u = json.loads(UNET_METRICS.read_text())
        print("\n" + "=" * 66)
        print(f"HEAD TO HEAD  —  {SPLIT} split, both at 800x800")
        print("=" * 66)
        print(f"{'metric':>22} {'U-Net':>11} {'Mask R-CNN':>12} {'diff':>10}")
        print("-" * 58)

        def cmp(label, a, b, inv=False):
            if a is None or b is None or not np.isfinite([a, b]).all():
                return
            d = float(b) - float(a)
            better = (d < 0) if inv else (d > 0)
            tag = "  MRCNN" if better else ("  U-Net" if d != 0 else "")
            print(f"{label:>22} {float(a):>11.4f} {float(b):>12.4f} "
                  f"{d:>+9.4f}{tag}")

        sev_res = result["severity"]
        cmp("mean Dice", u.get("mean_dice"), result["mean_dice"])
        cmp("mean IoU", u.get("mean_iou"), result["mean_iou"])
        for n in CLASS_NAMES:
            cmp(f"Dice {n}", u.get("dice", {}).get(n), result["dice"][n])
        us = u.get("severity") or {}
        if us:
            cmp("severity MAE (pp)", us.get("mae_pp"), sev_res["mae_pp"], inv=True)
            cmp("severity |bias| (pp)", abs(us.get("bias_pp", np.nan)),
                abs(sev_res["bias_pp"]), inv=True)
            cmp("severity r", us.get("pearson_r"), sev_res["pearson_r"])
        print("\n  (This table is a convenience only. The authoritative "
              "comparison is\n   three_way_test.py, which works from per-image "
              "results and is paired.)")

        print("\nNotes for reporting:")
        print(f"  - The 'U-Net' column above is whichever run UNET_METRICS points")
        print(f"    at: {UNET_METRICS}")
        print("    Mask R-CNN uses a COCO-pretrained backbone. Check whether the")
        print("    U-Net run being compared is the scratch or the ImageNet variant.")
        print("  - Both were fed identical 800x800 square inputs.")
        print(f"  - Mask R-CNN instance segm mAP on this split: "
              f"{ck.get('segm_map', float('nan')):.4f} (no U-Net equivalent).")
        print(f"  - Score threshold {best_t:.2f} was tuned on this split.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
        ts = SCORE_THRESHOLDS
        ax[0].plot(ts, [rows[t][0][LESION] for t in ts], "o-", label="lesion")
        ax[0].plot(ts, [rows[t][0][LEAF] for t in ts], "s-", label="leaf")
        ax[0].axvline(best_t, ls="--", c="k", lw=1)
        ax[0].set_xlabel("score threshold"); ax[0].set_ylabel("Dice")
        ax[0].set_title("Dice vs score threshold"); ax[0].legend()

        ax[1].plot(ts, [rows[t][4] for t in ts], "o-", color="#c1440e")
        ax[1].axvline(best_t, ls="--", c="k", lw=1)
        ax[1].set_xlabel("score threshold"); ax[1].set_ylabel("severity MAE (pp)")
        ax[1].set_title("Severity error vs threshold")

        lim = max(t_s.max(), p_s.max()) * 1.08 + 0.5
        ax[2].plot([0, lim], [0, lim], "k--", lw=1.2)
        ax[2].scatter(t_s, p_s, s=40, alpha=0.75, color="#2e8b57",
                      edgecolor="black", lw=0.5)
        ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim)
        ax[2].set_xlabel("true severity (%)"); ax[2].set_ylabel("predicted (%)")
        ax[2].set_title(f"Mask R-CNN severity (thr {best_t:.2f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison.png", dpi=120)
        plt.close()

        fig2, cax = plt.subplots(figsize=(6.2, 5.4))
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im2 = cax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        cax.set_xticks(range(NUM_CLASSES))
        cax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        cax.set_yticks(range(NUM_CLASSES))
        cax.set_yticklabels(CLASS_NAMES)
        cax.set_xlabel("predicted"); cax.set_ylabel("true")
        cax.set_title(f"Mask R-CNN — {SPLIT} split (recall-normalised)\n"
                      f"lesion Dice {dice[LESION]:.3f}   threshold {best_t:.2f}",
                      fontsize=11)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                colour = "white" if norm[i, j] > 0.5 else "#222222"
                cax.text(j, i, f"{100*norm[i,j]:.1f}%", ha="center", va="center",
                         fontsize=11, color=colour,
                         fontweight="bold" if i == j else "normal")
                cax.text(j, i + 0.30, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=6.5, color=colour, alpha=0.75)
            cax.add_patch(mpatches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#333333", lw=1.4))
        fig2.colorbar(im2, fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=140)
        plt.close()

        print(f"\nFigures: {OUT_DIR / 'comparison.png'}")
        print(f"         {OUT_DIR / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"(figures skipped: {type(e).__name__}: {e})")

    print(f"Metrics: {OUT_DIR / 'maskrcnn_semantic_metrics.json'}")


if __name__ == "__main__":
    main()
