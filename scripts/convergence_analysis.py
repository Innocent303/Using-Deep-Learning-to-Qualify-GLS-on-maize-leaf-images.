import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXP = Path("/content/drive/MyDrive/pretrain_exp")
ARMS = ["scratch", "imagenet", "maize", "control"]
TITLE = {"scratch": "Scratch", "imagenet": "ImageNet", "maize": "ImageNet + maize",
         "control": "ImageNet + non-plant control"}
PLANNED = [("imagenet", "scratch"), ("maize", "imagenet"),
           ("control", "imagenet"), ("maize", "control")]
EPOCHS = 150
CURVE = "val_lesion_dice_per_image"

METRICS = [
    ("auc", "AUC (mean Dice, epochs 1..K)", True, "PRIMARY"),
    ("t90own", "epochs to 90% of own plateau", False, "secondary"),
    ("tcommon", "epochs to a common Dice level", False, "secondary"),
    ("plateau", "plateau (best moving avg)", True, "context"),
]

COLORS = {"scratch": "#2a78d6", "imagenet": "#eb6834", "maize": "#1baf7a", "control": "#eda100"}
MARKERS = {"scratch": "o", "imagenet": "s", "maize": "^", "control": "D"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(exp, seeds):
    runs, notes = {}, []
    for arm in ARMS:
        for s in seeds:
            d = exp / "seg" / arm / f"seed{s}"
            if not ((d / "DONE").exists() and (d / "history.csv").exists()):
                continue
            try:
                h = pd.read_csv(d / "history.csv")
                done = json.loads((d / "DONE").read_text())
                raw = h[CURVE].to_numpy(float)
            except Exception as e:
                notes.append(f"{arm} seed {s}: unreadable ({type(e).__name__}) - skipped")
                continue
            if np.isnan(raw).any():
                notes.append(f"{arm} seed {s}: {int(np.isnan(raw).sum())} NaN epoch(s) filled forward")
                raw = pd.Series(raw).ffill().fillna(0.0).to_numpy()
            runs[(arm, s)] = {"raw": raw, "epochs_run": len(raw),
                              "best_epoch": int(done.get("best_epoch", int(np.argmax(raw)) + 1))}
    complete = [s for s in seeds if all((a, s) in runs for a in ARMS)]
    dropped = [s for s in seeds if s not in complete and any((a, s) in runs for a in ARMS)]
    if dropped:
        notes.append(f"seeds not complete for all 4 arms, left out: {dropped}")
    return runs, complete, notes


def smooth(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def first_reach(sm, level):
    idx = np.flatnonzero(sm >= level)
    return (int(idx[0]) + 1, False) if len(idx) else (len(sm), True)


def compute_metrics(runs, seeds, k, w, frac):
    t_common_all = min(r["epochs_run"] for r in runs.values())
    if k > t_common_all:
        print(f"!! K={k} exceeds the shortest run ({t_common_all} epochs); using K={t_common_all}")
        k = t_common_all
    plateaus = [smooth(r["raw"], w).max() for r in runs.values()]
    level = frac * float(np.median(plateaus))

    rows = []
    for (arm, s), r in sorted(runs.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1])):
        if s not in seeds:
            continue
        raw = r["raw"]
        sm = smooth(raw, w)
        plateau = float(sm.max())
        t_own, _ = first_reach(sm, frac * plateau)
        t_com, cens = first_reach(sm, level)
        best = float(raw.max())
        rows.append({"arm": arm, "seed": s, "epochs_run": r["epochs_run"],
                     "best_epoch": r["best_epoch"], "epochs_within_001_of_best": int((raw >= best - 0.01).sum()),
                     "auc": float(raw[:k].mean()), "t90own": t_own, "tcommon": t_com,
                     "tcommon_censored": cens, "plateau": plateau, "best_val_dice": best})
    return pd.DataFrame(rows), k, level, t_common_all


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def analyse(df, key, label, higher, role, seeds, posthoc_anyway, relevance):
    mat = df.pivot(index="seed", columns="arm", values=key).loc[seeds, ARMS]
    n = len(mat)
    print(f"\n--- {label}   [{role}] ---")
    print(f"  {'arm':<30}{'mean':>9}{'sd':>8}")
    for a in ARMS:
        print(f"  {TITLE[a]:<30}{mat[a].mean():>9.3f}{mat[a].std(ddof=1):>8.3f}")
    out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, p = stats.friedmanchisquare(*[mat[a].to_numpy() for a in ARMS])
    except Exception:
        print("  Friedman not computable (all values identical?)")
        return out
    sign = -1 if higher else 1
    ranks = np.apply_along_axis(stats.rankdata, 1, sign * mat.to_numpy()).mean(0)
    print(f"  Friedman chi2(3) = {chi2:.3f}, p = {p:.4f}   mean ranks (1 = best): "
          + ", ".join(f"{a} {r:.2f}" for a, r in zip(ARMS, ranks)))
    out.append({"measure": key, "contrast": "Friedman", "stat": chi2, "p": p})
    if p >= 0.05 and not posthoc_anyway:
        print("  Omnibus not significant -> post-hoc not reported (--posthoc-anyway for exploratory).")
        return out
    tag = "" if p < 0.05 else "   [EXPLORATORY - omnibus not significant]"
    print(f"  planned contrasts, a - b (positive = a higher){tag}")
    pv, rows = [], []
    for a, b in PLANNED:
        d = (mat[a] - mat[b]).to_numpy()
        try:
            pw = stats.wilcoxon(mat[a].to_numpy(), mat[b].to_numpy()).pvalue
        except ValueError:
            pw = 1.0
        se = d.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        pv.append(pw)
        rows.append((a, b, d.mean(), d.mean() - h, d.mean() + h, pw))
    adj = holm(pv)
    print(f"    {'a - b':<28}{'diff':>8}{'95% CI':>20}{'p':>8}{'p(Holm)':>9}")
    for (a, b, m, lo, hi, pw), pa in zip(rows, adj):
        flag = ""
        if key in ("auc", "plateau") and abs(m) > relevance and (lo > 0 or hi < 0):
            flag = "  <- relevant"
        print(f"    {a + ' - ' + b:<28}{m:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{pw:>8.4f}{pa:>9.4f}{flag}")
        out.append({"measure": key, "contrast": f"{a} - {b}", "stat": m, "ci_lo": lo,
                    "ci_hi": hi, "p": pw, "p_holm": pa})
    return out


def mean_curves(runs, seeds, w, t_common):
    rows = []
    for arm in ARMS:
        sm = np.array([smooth(runs[(arm, s)]["raw"], w)[:t_common] for s in seeds])
        bsf = []
        for s in seeds:
            b = np.maximum.accumulate(runs[(arm, s)]["raw"])
            bsf.append(np.concatenate([b, np.full(EPOCHS - len(b), b[-1])]))
        bsf = np.array(bsf)
        live = np.array([[runs[(arm, s)]["epochs_run"] > e for s in seeds] for e in range(EPOCHS)]).sum(1)
        for e in range(EPOCHS):
            row = {"arm": arm, "epoch": e + 1, "n_runs_still_training": int(live[e]),
                   "best_so_far_mean": bsf[:, e].mean(),
                   "best_so_far_se": bsf[:, e].std(ddof=1) / np.sqrt(len(seeds))}
            if e < t_common:
                row["val_dice_smoothed_mean"] = sm[:, e].mean()
                row["val_dice_smoothed_se"] = sm[:, e].std(ddof=1) / np.sqrt(len(seeds))
            rows.append(row)
    return pd.DataFrame(rows)


def make_figure(curves, df, seeds, t_common, k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE,
                           gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=1, zorder=0)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    for arm in ARMS:
        c = curves[(curves.arm == arm) & curves.val_dice_smoothed_mean.notna()]
        ax[0].fill_between(c.epoch, c.val_dice_smoothed_mean - c.val_dice_smoothed_se,
                           c.val_dice_smoothed_mean + c.val_dice_smoothed_se,
                           color=COLORS[arm], alpha=0.10, lw=0, zorder=2)
        ax[0].plot(c.epoch, c.val_dice_smoothed_mean, color=COLORS[arm], lw=2,
                   solid_capstyle="round", zorder=3, label=TITLE[arm])
        ax[0].plot(c.epoch.iloc[-1], c.val_dice_smoothed_mean.iloc[-1], MARKERS[arm], ms=7,
                   color=COLORS[arm], mec=SURFACE, mew=2, zorder=4)
    ax[0].axvline(k, color=INK2, lw=1, ls=":", zorder=1)
    ax[0].annotate(f" K={k}", xy=(k, 1.0), xycoords=("data", "axes fraction"), color=INK2,
                   va="top", fontsize=9)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation lesion Dice (5-epoch moving avg)")
    ax[0].set_title(f"A  How fast it learns (epochs 1-{t_common}, all runs training)",
                    loc="left", fontsize=10.5, color=INK)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")

    for arm in ARMS:
        c = curves[curves.arm == arm]
        ax[1].fill_between(c.epoch, c.best_so_far_mean - c.best_so_far_se,
                           c.best_so_far_mean + c.best_so_far_se,
                           color=COLORS[arm], alpha=0.10, lw=0, zorder=2)
        ax[1].plot(c.epoch, c.best_so_far_mean, color=COLORS[arm], lw=2, solid_capstyle="round", zorder=3)
    ax[1].set_xlim(0, EPOCHS)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("validation lesion Dice of best checkpoint so far")
    ax[1].set_title("B  Level reached (best so far)", loc="left", fontsize=10.5, color=INK)
    fig.subplots_adjust(wspace=0.32)

    piv = df.pivot(index="seed", columns="arm", values="auc").loc[seeds, ARMS]
    x = np.arange(len(ARMS))
    for s in seeds:
        ax[2].plot(x, piv.loc[s].to_numpy(), color=GRID, lw=1, zorder=1)
    for i, arm in enumerate(ARMS):
        ax[2].plot(np.full(len(seeds), i), piv[arm], MARKERS[arm], ms=7, color=COLORS[arm],
                   mec=SURFACE, mew=2, alpha=0.9, zorder=3)
        ax[2].hlines(piv[arm].mean(), i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["Scratch", "ImageNet", "+ maize", "+ control"], fontsize=9)
    ax[2].set_ylabel(f"AUC: mean val Dice, epochs 1-{k}")
    ax[2].set_title("C  Per seed (lines join the same seed)", loc="left", fontsize=10.5, color=INK)

    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    global EXP
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(EXP))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--k", type=int, default=30, help="AUC window: epochs 1..K")
    ap.add_argument("--smooth", type=int, default=5, help="trailing moving-average window")
    ap.add_argument("--frac", type=float, default=0.9, help="fraction of plateau for the time-to-level measures")
    ap.add_argument("--relevance", type=float, default=0.02, help="ad hoc |diff| for AUC / plateau")
    ap.add_argument("--posthoc-anyway", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args, _ = ap.parse_known_args()
    EXP = Path(args.exp)

    runs, seeds, notes = load_runs(EXP, args.seeds)
    for n in notes:
        print("note:", n)
    if len(seeds) < 3:
        sys.exit(f"!! only {len(seeds)} seed(s) complete for all 4 arms - need at least 3")
    out_dir = EXP / "convergence"
    out_dir.mkdir(exist_ok=True)

    df, k, level, t_all = compute_metrics(runs, seeds, args.k, args.smooth, args.frac)
    print(f"{len(seeds)} complete seeds x 4 arms = {len(df)} runs   (seeds {seeds})")
    print(f"shortest run trained {t_all} epochs; AUC uses epochs 1..{k}; common level = "
          f"{level:.4f} Dice ({args.frac:.0%} of the median plateau)")
    n_cens = int(df.tcommon_censored.sum())
    if n_cens:
        print(f"!! {n_cens} run(s) never reached the common level; their time is censored at their last epoch")

    print("\nBest epoch is not a speed measure (shown for reference)")
    print(f"  {'arm':<30}{'best epoch, median [min-max]':>30}{'epochs within 0.01 of best':>29}")
    for a in ARMS:
        d = df[df.arm == a]
        be = f"{d.best_epoch.median():.0f} [{d.best_epoch.min()}-{d.best_epoch.max()}]"
        wd = d.epochs_within_001_of_best
        fl = f"{wd.median():.0f} [{wd.min()}-{wd.max()}]"
        print(f"  {TITLE[a]:<30}{be:>30}{fl:>29}")
    print("  (many epochs within 0.01 of the best = a flat peak, so its argmax is unstable)")

    print("\n" + "=" * 78)
    print("ARM COMPARISONS (Friedman, seed-blocked; paired by seed)")
    print("=" * 78)
    stat_rows = []
    for key, label, higher, role in METRICS:
        lab = label.replace("K", str(k)) if key == "auc" else label
        stat_rows += analyse(df, key, lab, higher, role, seeds, args.posthoc_anyway, args.relevance)

    curves = mean_curves(runs, seeds, args.smooth, t_all)
    df.to_csv(out_dir / "convergence_per_run.csv", index=False)
    curves.to_csv(out_dir / "convergence_mean_curves.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out_dir / "convergence_stats.csv", index=False)
    if not args.no_plot:
        png = out_dir / "convergence_curves.png"
        make_figure(curves, df, seeds, t_all, k, png)
        print(f"\nfigure: {png}")
    print(f"tables: {out_dir}")
    print("\nReading it: H5 is supported if AUC (and the time measures) differ between arms while the")
    print("plateau does not. AUC is the pre-specified primary; the time measures are secondary.")


if __name__ == "__main__":
    main()
