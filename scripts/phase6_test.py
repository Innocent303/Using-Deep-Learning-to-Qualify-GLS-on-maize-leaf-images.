import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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


_HERE = _find_module("phase4_evaluate")
if _HERE is None:
    raise RuntimeError(
        "Could not find phase4_evaluate.py.\n"
        "Put it next to this script, then:  %cd /content/drive/MyDrive\n"
        f"Searched from: {Path.cwd()}\n"
        "Check with:  !find /content/drive/MyDrive -name phase4_evaluate.py"
    )

import phase4_evaluate as p4


TEST_SPLIT = "test"

SKIP_CONFIRM = False


def checkpoint_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    ckpt = p4.CHECKPOINT
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt} — run Phase 3 first.")

    root = p4.ROOT
    images_subdir = getattr(p4, "IMAGES_SUBDIR", "images")
    test_images = root / TEST_SPLIT / images_subdir
    if not test_images.is_dir():
        raise SystemExit(f"No test split found at {test_images}\n"
                         "Run resplit_dataset.py first.")

    ledger_path = root / "runs" / "phase3" / "test_runs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    ck_id = checkpoint_id(ckpt)

    print("=" * 64)
    print("PHASE 6 — HELD-OUT TEST EVALUATION")
    print("=" * 64)
    print(f"checkpoint: {ckpt.name}  (id {ck_id})")

    if ledger:
        print(f"\n!! This test set has already been evaluated {len(ledger)} time(s):")
        for r in ledger:
            same = "  <-- same model" if r["checkpoint_id"] == ck_id else ""
            print(f"     {r['timestamp']}  model {r['checkpoint_id']}  "
                  f"mDice {r['mean_dice']:.4f}{same}")
        print("\n   Repeated evaluation on the same test set makes the result")
        print("   progressively more optimistic. Report the FIRST run's number,")
        print("   and say how many times the set was used.")

    if not SKIP_CONFIRM:
        print("\nThis should be your final model. Continue?")
        try:
            if input("type 'yes' to run the test evaluation: ").strip().lower() != "yes":
                print("Cancelled — nothing was evaluated.")
                return
        except EOFError:
            print("\nNon-interactive session. Set SKIP_CONFIRM = True to run.")
            return

    p4.SPLIT = TEST_SPLIT
    p4.OUT_DIR = root / "runs" / "phase3" / f"eval_{TEST_SPLIT}"
    p4.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    p4.main()

    test_metrics_path = p4.OUT_DIR / "metrics.json"
    if not test_metrics_path.exists():
        print("\n(no metrics.json produced — skipping ledger and comparison)")
        return
    test_m = json.loads(test_metrics_path.read_text())

    ledger.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": ckpt.name,
        "checkpoint_id": ck_id,
        "mean_dice": test_m["mean_dice"],
        "run_number": len(ledger) + 1,
    })
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))

    val_metrics_path = root / "runs" / "phase3" / "eval_val" / "metrics.json"
    if not val_metrics_path.exists():
        print(f"\n(no val metrics at {val_metrics_path} — run phase4_evaluate.py "
              "on val to get a comparison)")
        return
    val_m = json.loads(val_metrics_path.read_text())

    print("\n" + "=" * 64)
    print("VALIDATION vs TEST")
    print("=" * 64)
    print(f"{'metric':>22} {'val':>10} {'test':>10} {'diff':>10}")
    print("-" * 56)

    def row(label, v, t, fmt="{:>10.4f}", pp=False):
        d = t - v
        suffix = " pp" if pp else ""
        print(f"{label:>22} {fmt.format(v)} {fmt.format(t)} "
              f"{d:>+9.4f}{suffix}")
        return d

    row("mean Dice", val_m["mean_dice"], test_m["mean_dice"])
    row("mean IoU", val_m["mean_iou"], test_m["mean_iou"])
    row("pixel accuracy", val_m["pixel_accuracy"], test_m["pixel_accuracy"])
    print()
    for name in val_m["dice"]:
        if name in test_m["dice"]:
            row(f"Dice {name}", val_m["dice"][name], test_m["dice"][name])

    vs, ts = val_m.get("severity") or {}, test_m.get("severity") or {}
    if vs and ts:
        print()
        row("severity MAE (pp)", vs["mae_pp"], ts["mae_pp"])
        row("severity bias (pp)", vs["bias_pp"], ts["bias_pp"])
        row("severity r", vs["pearson_r"], ts["pearson_r"])

    gap = val_m["mean_dice"] - test_m["mean_dice"]
    print("\n" + "-" * 56)
    if abs(gap) < 0.02:
        print(f"Test tracks val closely (gap {gap:+.4f}). The val set was a")
        print("reliable guide, and the test number should generalise.")
    elif gap > 0.05:
        print(f"Test is {gap:.4f} BELOW val — the model was tuned to val more")
        print("than it generalises. Report the test number as the honest result.")
    elif gap < -0.02:
        print(f"Test is {abs(gap):.4f} ABOVE val. Usually means the test split")
        print("happens to be easier — small splits (65 images) are noisy.")
    else:
        print(f"Modest gap ({gap:+.4f}) — normal for a {65}-image test split.")

    print(f"\nTest run #{len(ledger)} recorded in {ledger_path.name}")
    print(f"Outputs: {p4.OUT_DIR}")

    print("\n" + "=" * 64)
    print("REPORT THESE NUMBERS")
    print("=" * 64)
    print(f"  mean Dice        {test_m['mean_dice']:.4f}")
    print(f"  mean IoU         {test_m['mean_iou']:.4f}")
    for name, d in test_m["dice"].items():
        print(f"  Dice {name:<14} {d:.4f}")
    if ts:
        print(f"  severity MAE     {ts['mae_pp']:.2f} pp")
        print(f"  severity bias    {ts['bias_pp']:+.2f} pp")
        print(f"  severity r       {ts['pearson_r']:.4f}")
    print(f"\n  n = {test_m.get('severity', {}).get('n_images', '?')} test images, "
          f"evaluated at {getattr(p4, 'IMG_SIZE', '?')}px")


if __name__ == "__main__":
    main()
