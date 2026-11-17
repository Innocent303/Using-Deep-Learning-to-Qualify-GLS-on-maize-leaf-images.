import sys
from pathlib import Path

import pandas as pd


DRIVE = Path("/content/drive/MyDrive")
SEEDS = list(range(1, 11))

MODELS = {
    "U-Net (scratch)": (
        lambda s: DRIVE / "unet" / "runs" / f"unet_scratch_seed{s}",
        "eval_test/per_image.csv"),
    "U-Net (ImageNet)": (
        lambda s: DRIVE / "unet" / "runs" / f"unet_resnet34_imagenet_seed{s}",
        "eval_test/per_image.csv"),
    "Mask R-CNN": (
        lambda s: DRIVE / "coco_split_800sq_aligned" / "runs" / f"maskrcnn_seed{s}",
        "compare_test/per_image.csv"),
    "YOLO11m-seg": (
        lambda s: DRIVE / "yolo_seg_800sq" / "runs" / f"yolo11m_seg_800_seed{s}",
        "compare_test/per_image.csv"),
}

OUT_LONG = DRIVE / "standard_long.csv"
OUT_MEANS = DRIVE / "standard_seed_means.csv"

OUT_A = DRIVE / "partA_architectures.csv"
OUT_B = DRIVE / "partB_pretraining.csv"

PART_A_MODELS = ["U-Net (ImageNet)", "Mask R-CNN", "YOLO11m-seg"]
PART_B_MODELS = ["U-Net (scratch)", "U-Net (ImageNet)"]


def main():
    frames, missing = [], []

    for label, (path_fn, rel) in MODELS.items():
        for s in SEEDS:
            p = path_fn(s) / rel
            if not p.exists():
                missing.append(f"{label} seed {s}")
                continue
            t = pd.read_csv(p)
            if "image" not in t.columns and "file" in t.columns:
                t = t.rename(columns={"file": "image"})
            t["model"] = label
            t["seed"] = s
            frames.append(t)

    if not frames:
        sys.exit("No per-image files found. Check the paths in MODELS.")

    long = pd.concat(frames, ignore_index=True)

    if "dice_GLS_lesion" not in long.columns:
        sys.exit("No dice_GLS_lesion column — the runs predate the per-class "
                 "output and must be re-evaluated.")

    front = [c for c in ["image", "model", "seed"] if c in long.columns]
    long = long[front + [c for c in long.columns if c not in front]]
    long = long.sort_values(["model", "seed", "image"])
    long.to_csv(OUT_LONG, index=False)

    print(f"wrote {OUT_LONG}")
    print(f"  {len(long)} rows   "
          f"{long['image'].nunique()} images x {long['model'].nunique()} models "
          f"x {long['seed'].nunique()} seeds")
    if missing:
        print(f"\n  {len(missing)} run(s) not found:")
        for m in missing[:12]:
            print(f"    {m}")
        if len(missing) > 12:
            print(f"    ... and {len(missing) - 12} more")

    has_lesion = long["has_lesion"] == 1 if "has_lesion" in long.columns \
        else pd.Series(True, index=long.index)
    les = long[has_lesion]

    g = les.groupby(["model", "seed"])
    means = g.agg(
        n_images=("image", "nunique"),
        lesion_dice=("dice_GLS_lesion", "mean"),
        lesion_iou=("iou_GLS_lesion", "mean"),
        mean_dice=("mean_dice", "mean"),
        sev_abs_err=("severity_error_pp", lambda x: x.abs().mean()),
        sev_bias=("severity_error_pp", "mean"),
    ).reset_index()

    if {"severity_true_pct", "severity_pred_pct"} <= set(les.columns):
        r = (les.groupby(["model", "seed"])
                .apply(lambda d: d["severity_true_pct"].corr(d["severity_pred_pct"]),
                       include_groups=False)
                .reset_index(name="sev_r"))
        means = means.merge(r, on=["model", "seed"], how="left")

    n_nolesion = long[~has_lesion]["image"].nunique()
    means.to_csv(OUT_MEANS, index=False)
    print(f"\nwrote {OUT_MEANS}")
    print(f"  {len(means)} rows (one per model per seed)")
    print(f"  {n_nolesion} image(s) with no annotated lesion excluded "
          f"from the lesion means")

    print("\n" + "=" * 70)
    print("SEED-LEVEL SUMMARY  (this is what the tests compare)")
    print("=" * 70)
    print(f"{'model':<20}{'seeds':>7}{'lesion Dice':>14}{'sd':>9}"
          f"{'sev MAE':>10}{'sd':>8}")
    print("-" * 68)
    for m, d in means.groupby("model"):
        print(f"{m:<20}{len(d):>7}{d['lesion_dice'].mean():>14.4f}"
              f"{d['lesion_dice'].std():>9.4f}"
              f"{d['sev_abs_err'].mean():>10.2f}{d['sev_abs_err'].std():>8.2f}")

    a = means[means["model"].isin(PART_A_MODELS)].copy()
    have_a = [m for m in PART_A_MODELS if m in set(a["model"])]
    if len(have_a) < len(PART_A_MODELS):
        print(f"\n  !! Part A is missing: "
              f"{set(PART_A_MODELS) - set(have_a)}")
    counts_a = a.groupby("seed")["model"].nunique()
    keep_a = counts_a[counts_a == len(have_a)].index
    dropped_a = sorted(set(a["seed"]) - set(keep_a))
    a = a[a["seed"].isin(keep_a)].sort_values(["model", "seed"])
    a.to_csv(OUT_A, index=False)

    b = means[means["model"].isin(PART_B_MODELS)].copy()
    counts_b = b.groupby("seed")["model"].nunique()
    keep_b = counts_b[counts_b == 2].index
    dropped_b = sorted(set(b["seed"]) - set(keep_b))
    b = b[b["seed"].isin(keep_b)].sort_values(["model", "seed"])
    b.to_csv(OUT_B, index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS FILES")
    print("=" * 70)
    print(f"  {OUT_A.name:<28} {len(keep_a):>2} seeds x {len(have_a)} models"
          f"  ({len(a)} rows)")
    print(f"      {', '.join(have_a)}")
    if dropped_a:
        print(f"      seeds excluded for incompleteness: {dropped_a}")
    print(f"  {OUT_B.name:<28} {len(keep_b):>2} paired seeds"
          f"  ({len(b)} rows)")
    print(f"      {' vs '.join(PART_B_MODELS)}")
    if dropped_b:
        print(f"      seeds excluded (only one variant present): {dropped_b}")

    if len(keep_b) < 6:
        print(f"\n  !! Only {len(keep_b)} paired seeds for Part B. The Wilcoxon")
        print( "     signed-rank test cannot return p < 0.05 with fewer than 6")
        print( "     pairs, whatever the size of the effect. Report the")
        print( "     difference and its interval rather than a test result.")

    print("\nSeeds available per model:")
    piv = means.pivot_table(index="seed", columns="model",
                            values="lesion_dice", aggfunc="size").fillna(0)
    print(piv.astype(int).to_string())

    complete = piv[(piv > 0).all(axis=1)].index.tolist()
    print(f"\nSeeds present for EVERY model: {complete}")
    print("Friedman needs complete blocks, so an analysis across all four")
    print("models can only use these seeds. Analysing the three fully "
          "replicated\nmodels separately keeps all ten.")


if __name__ == "__main__":
    main()
