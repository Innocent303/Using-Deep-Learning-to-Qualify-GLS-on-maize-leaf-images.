import json
import sys
from pathlib import Path

import pandas as pd


DRIVE = Path("/content/drive/MyDrive")
SEEDS = list(range(1, 11))
RUNS_FOR = {s: DRIVE / f"logo_runs_seed{s}" for s in SEEDS}
GROUPS_CSV = DRIVE / "logo_groups.csv"

OUT_LONG = DRIVE / "logo_long.csv"
OUT_FOLD = DRIVE / "logo_fold_summary.csv"

MODELS = ["unet", "maskrcnn", "yolo"]
FOLDS = ["baynesfield_2009", "cedara_2012", "cedara_2019", "delmas"]


def main():
    present = {s: r for s, r in RUNS_FOR.items() if r.is_dir()}
    if not present:
        sys.exit(f"No seed directories found under {DRIVE}")
    print(f"seed directories found: {sorted(present)}\n")

    rows, fold_rows, missing = [], [], []

    for seed, RUNS in present.items():
      for m in MODELS:
        for f in FOLDS:
            d = RUNS / m / f
            pi, mj = d / "per_image.csv", d / "metrics.json"
            if not pi.exists():
                missing.append(f"seed{seed}/{m}/{f}")
                continue

            t = pd.read_csv(pi)
            if "image" not in t.columns and "file" in t.columns:
                t = t.rename(columns={"file": "image"})
            t["model_dir"] = m
            t["seed"] = seed
            if "fold" not in t.columns:
                t["fold"] = f
            if "model" not in t.columns:
                t["model"] = m
            rows.append(t)

            if mj.exists():
                j = json.loads(mj.read_text())
                fold_rows.append({
                    "model": j.get("model", m),
                    "model_dir": m,
                    "seed": seed,
                    "fold": f,
                    "n_images": j.get("n_images"),
                    "score_threshold": j.get("score_threshold"),
                    "mean_dice": j.get("mean_dice"),
                    "mean_iou": j.get("mean_iou"),
                    **{f"dice_{k}": v for k, v in (j.get("dice") or {}).items()},
                    **{f"iou_{k}": v for k, v in (j.get("iou") or {}).items()},
                    **{f"sev_{k}": v for k, v in (j.get("severity") or {}).items()},
                })

    if missing:
        print(f"missing {len(missing)} run(s): {', '.join(missing)}")
        print("(the table below covers only completed runs)\n")
    if not rows:
        sys.exit("Nothing to collect.")

    long = pd.concat(rows, ignore_index=True)

    if GROUPS_CSV.exists():
        g = pd.read_csv(GROUPS_CSV, dtype={"stem": str})
        long["stem"] = long["image"].astype(str).str.split(".").str[0]
        long = long.merge(g[["stem", "group", "gls_instance_count"]],
                          on="stem", how="left")
        long = long.rename(columns={"gls_instance_count": "n_lesions_annotated"})

    front = [c for c in ["image", "stem", "model", "model_dir", "seed",
                         "fold", "group"] if c in long.columns]
    rest = [c for c in long.columns if c not in front]
    long = long[front + rest].sort_values(["seed", "fold", "image", "model"])

    long.to_csv(OUT_LONG, index=False)
    print(f"wrote {OUT_LONG}")
    print(f"  {len(long)} rows  ({long['image'].nunique()} images "
          f"x {long['model'].nunique()} models)")

    if fold_rows:
        fs = pd.DataFrame(fold_rows).sort_values(["model", "fold"])
        fs.to_csv(OUT_FOLD, index=False)
        print(f"wrote {OUT_FOLD}  ({len(fs)} rows)")

    print("\n" + "=" * 66)
    print("CHECKS")
    print("=" * 66)
    if "seed" in long.columns:
        n_seeds = long["seed"].nunique()
        print(f"seeds represented: {n_seeds}  {sorted(long['seed'].unique())}")
        per_seed = long.groupby("seed")["image"].nunique()
        if per_seed.nunique() != 1:
            print("  !! seeds cover different numbers of images:")
            print(per_seed.to_string())
        else:
            print(f"  each seed covers {per_seed.iloc[0]} images\n")

    per_model = long.groupby("model")["image"].nunique()
    print("images scored per model:")
    for k, v in per_model.items():
        print(f"  {k:<20} {v}")
    if per_model.nunique() != 1:
        print("  !! models cover different image sets — comparisons are not paired")
    else:
        print("  all models cover the same images (paired)")

    if "group" in long.columns and "fold" in long.columns:
        mismatch = (long["group"] != long["fold"]).sum()
        print(f"\nrows where group != fold: {mismatch} "
              f"(should be 0 — an image is only scored when its group is held out)")

    if "has_lesion" in long.columns:
        n0 = long[long["has_lesion"] == 0]["image"].nunique()
        print(f"\nimages with NO annotated lesion: {n0}")
        if n0:
            print("  lesion Dice is degenerate for these — filter on has_lesion")

    if "dice_GLS_lesion" in long.columns:
        sub = long[long.get("has_lesion", 1) == 1]
        print("\n" + "=" * 66)
        print("LESION DICE — mean per fold, averaged over seeds")
        print("=" * 66)
        piv = sub.pivot_table(index="fold", columns="model",
                              values="dice_GLS_lesion", aggfunc="mean")
        print(piv.round(4).to_string())

        if "seed" in sub.columns and sub["seed"].nunique() > 1:
            print("\nSEED-TO-SEED VARIABILITY (per model, over fold means)")
            print(f"{'model':<20}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}{'n seeds':>9}")
            print("-" * 65)
            per = (sub.groupby(["model", "seed"])["dice_GLS_lesion"]
                      .mean().reset_index())
            for mdl, g in per.groupby("model"):
                v = g["dice_GLS_lesion"]
                print(f"{mdl:<20}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}"
                      f"{v.min():>9.4f}{v.max():>9.4f}{len(v):>9}")
            print("\nThis sd is the run-to-run variability. Differences between")
            print("models smaller than about 2x it should not be called real.")
        print("\n(these are per-image means; the fold summary holds the "
              "pixel-accumulated values, which are the ones to report)")

    print("\n" + "=" * 66)
    print("R")
    print("=" * 66)
    print(f'''
d <- read.csv("logo_long.csv", stringsAsFactors = TRUE)

# lesion Dice is undefined for images with no annotated lesion
d <- subset(d, has_lesion == 1)

d$model <- relevel(factor(d$model), ref = "U-Net (ImageNet)")
d$image <- factor(d$image)
d$group <- factor(d$group)

library(lme4); library(lmerTest); library(emmeans)

# (1|image) pairs the models on the same picture
# (1|group) stops images from one site counting as independent observations
m <- lmer(dice_GLS_lesion ~ model + (1|image) + (1|group), data = d)
summary(m)
emmeans(m, pairwise ~ model, adjust = "tukey")   # report estimates + CIs

# does the ranking hold at every site?
m2 <- lmer(dice_GLS_lesion ~ model * group + (1|image), data = d)
anova(m2)                                        # look at the interaction term

# severity error, absolute
d$sev_abs_err <- abs(d$severity_error_pp)
ms <- lmer(sev_abs_err ~ model + (1|image) + (1|group), data = d)
emmeans(ms, pairwise ~ model, adjust = "tukey")

# check residuals before trusting any of the above
plot(m); qqnorm(resid(m)); qqline(resid(m))
'''.strip())


if __name__ == "__main__":
    main()
