import csv
import itertools
import sys
from pathlib import Path

import numpy as np


SPLIT = "test"

SOURCES = {
    "U-Net (scratch)": Path(
        f"/content/drive/MyDrive/unet/runs/phase3/eval_{SPLIT}/per_image.csv"),
    "U-Net (ImageNet)": Path(
        "/content/drive/MyDrive/unet/runs/phase3_unet_resnet34_imagenet")
        / f"eval_{SPLIT}" / "per_image.csv",
    "Mask R-CNN": Path("/content/drive/MyDrive/coco_split_800sq_aligned/runs/maskrcnn")
                  / f"compare_{SPLIT}" / "per_image.csv",
    "YOLO11m-seg": Path("/content/drive/MyDrive/yolo_seg_800sq/runs/yolo11m_seg_800")
                   / f"compare_{SPLIT}" / "per_image.csv",
}

METRICS = [
    ("dice_GLS_lesion", "lesion Dice", False, False),
    ("severity_error_pp", "severity |error| (pp)", True, True),
    ("mean_dice", "mean Dice (all classes)", False, False),
]

EXCLUDE_NO_LESION = True

ALPHA = 0.05
N_BOOT = 10000
SEED = 0


def load(path):
    if not path.exists():
        return None
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = r.get("image") or r.get("file")
            if key:
                rows[Path(key).stem.lower()] = r
    return rows


def paired_p(d):
    n = len(d)
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return 0.0 if np.mean(d) != 0 else 1.0
    t = float(np.mean(d)) / (sd / np.sqrt(n))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        from math import erfc, sqrt
        return float(erfc(abs(t) / sqrt(2)))


def bootstrap_ci(d, seed=SEED):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def friedman(matrix):
    if matrix.shape[1] < 3:
        return None, None
    try:
        from scipy import stats
        return stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    except ImportError:
        return None, None


def main():
    loaded, missing = {}, []
    for name, path in SOURCES.items():
        d = load(path)
        if d is None:
            missing.append(f"{name}: {path}")
        else:
            loaded[name] = d

    if missing:
        print("Missing per-image files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun phase6_test.py, compare_models.py and yolo_compare.py first.")
        if len(loaded) < 2:
            sys.exit(1)
        print(f"\nContinuing with {len(loaded)} model(s).\n")

    names = list(loaded)
    shared = sorted(set.intersection(*(set(loaded[n]) for n in names)))
    print(f"Models: {', '.join(names)}")
    print(f"Paired on {len(shared)} image(s) common to all\n")

    sizes = [len(loaded[n]) for n in names]
    if shared and min(sizes) and len(shared) < 0.8 * min(sizes):
        print("!" * 72)
        print("WARNING: the models were evaluated on largely DIFFERENT images.")
        print(f"  smallest per-model set: {min(sizes)}   shared: {len(shared)}")
        print("  This usually means the datasets were split independently, so")
        print("  each model has its own test set. A paired comparison on the")
        print("  overlap is unreliable, and an image held out by one model may")
        print("  have been TRAINING data for another.")
        print("  Run check_splits.py before trusting anything below.")
        print("!" * 72 + "\n")

    if len(shared) < 10:
        sys.exit("Too few paired images.")

    for col, label, lower_better, take_abs in METRICS:
        print("=" * 72)
        print(f"{label}   (n = {len(shared)})")
        print("=" * 72)

        vals = {}
        skip = False
        for nm in names:
            try:
                v = np.array([float(loaded[nm][k][col]) for k in shared])
            except KeyError:
                print(f"  column '{col}' missing for {nm}, skipping metric\n")
                skip = True
                break
            vals[nm] = np.abs(v) if take_abs else v
        if skip:
            continue

        ok = np.all([np.isfinite(vals[nm]) for nm in names], axis=0)
        if EXCLUDE_NO_LESION and "lesion" in col:
            has = np.array([
                all(str(loaded[nm][k].get("has_lesion", "1")).strip() in ("1", "1.0")
                    for nm in names) for k in shared])
            n_drop = int((~has & ok).sum())
            if n_drop:
                print(f"  excluding {n_drop} image(s) with no annotated lesion")
            ok = ok & has
        for nm in names:
            vals[nm] = vals[nm][ok]
        n = int(ok.sum())

        print(f"  {'model':>14} {'mean':>9} {'sd':>9}")
        ranked = sorted(names, key=lambda nm: vals[nm].mean(), reverse=not lower_better)
        for nm in ranked:
            print(f"  {nm:>14} {vals[nm].mean():>9.4f} {vals[nm].std(ddof=1):>9.4f}")
        print(f"  best: {ranked[0]}")

        matrix = np.column_stack([vals[nm] for nm in names])
        stat, p_om = friedman(matrix)
        if p_om is not None:
            verdict = "differences exist" if p_om < ALPHA else "NO detectable difference"
            print(f"\n  Friedman omnibus: chi2 = {stat:.3f}, p = {p_om:.4f}  "
                  f"-> {verdict}")
        else:
            print("\n  (scipy unavailable — omnibus test skipped)")

        pairs = list(itertools.combinations(names, 2))
        raw_p, diffs, cis = [], [], []
        for a, b in pairs:
            d = vals[b] - vals[a]
            raw_p.append(paired_p(d))
            diffs.append(float(np.mean(d)))
            cis.append(bootstrap_ci(d))

        adj_p = holm(raw_p)

        print(f"\n  {'comparison':>28} {'diff':>10} {'95% CI':>22} "
              f"{'p':>8} {'p adj':>8}")
        print("  " + "-" * 80)
        for (a, b), d, (lo, hi), p, pa in zip(pairs, diffs, cis, raw_p, adj_p):
            sig = "*" if pa < ALPHA and not (lo <= 0 <= hi) else " "
            print(f"  {f'{b} - {a}':>28} {d:>+10.4f} "
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22} {p:>8.4f} {pa:>8.4f} {sig}")

        if not any(pa < ALPHA for pa in adj_p):
            print(f"\n  -> No pair differs significantly after correction.")
            print(f"     At n={n} these models are indistinguishable on {label}.")
            sds = [np.std(vals[b] - vals[a], ddof=1) for a, b in pairs]
            md = max(abs(d) for d in diffs) if diffs else 0
            if md > 0:
                need = int(np.ceil(((1.96 + 0.84) * max(sds) / md) ** 2))
                print(f"     Detecting the largest observed gap ({md:+.4f}) would")
                print(f"     need roughly {need} paired images.")
        else:
            print(f"\n  -> Significant pair(s) marked *")
        print()

    print("=" * 72)
    print("Reporting notes")
    print("=" * 72)
    print("  - p adj is Holm-Bonferroni across the 3 pairwise tests per metric.")
    print("  - A non-significant result is NOT proof of equivalence; with 65")
    print("    images only fairly large differences are detectable.")
    print("  - Report the CI alongside the point estimate, not just the p-value.")
    print("  - Initialisation differs between models. 'U-Net (scratch)' uses no")
    print("    pretrained weights; 'U-Net (ImageNet)', Mask R-CNN and YOLO all")
    print("    use pretrained backbones. Comparing the two U-Net variants")
    print("    isolates the effect of pretraining, since they share an otherwise")
    print("    identical training setup.")


if __name__ == "__main__":
    main()
