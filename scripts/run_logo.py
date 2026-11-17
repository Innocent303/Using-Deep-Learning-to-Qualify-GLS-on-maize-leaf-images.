import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DRIVE = Path("/content/drive/MyDrive")
LOGO = DRIVE / "logo"
POOL = DRIVE / "logo_pool"
SEED = int(os.environ.get("GLS_SEED", 1))

RUNS = DRIVE / f"logo_runs_seed{SEED}"

FOLDS = ["baynesfield_2009", "cedara_2012", "cedara_2019", "delmas"]
MODELS = ["unet", "maskrcnn", "yolo"]

ORDER = [(m, f) for m in ["yolo", "unet", "maskrcnn"] for f in FOLDS]

IMGSZ = 800

UNET_EPOCHS, UNET_PATIENCE, UNET_BATCH, UNET_LR = 150, 20, 4, 3e-4
UNET_ENCODER = "resnet34"

MRCNN_EPOCHS, MRCNN_PATIENCE, MRCNN_BATCH, MRCNN_LR = 50, 20, 4, 1e-4

YOLO_MODEL, YOLO_EPOCHS, YOLO_PATIENCE, YOLO_BATCH = "yolo11m-seg.pt", 150, 20, 16

SCORE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run_dir(model, fold):
    return RUNS / model / fold


def is_done(model, fold):
    return (run_dir(model, fold) / "DONE").exists()


def mark_done(model, fold, info):
    d = run_dir(model, fold)
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text(json.dumps(
        {"finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "seed": SEED, **info}, indent=2))


def status():
    print(f"seed {SEED}   {RUNS}")
    print(f"{'model':<12}" + "".join(f"{f[:14]:>16}" for f in FOLDS))
    print("-" * (12 + 16 * len(FOLDS)))
    n_done = 0
    for m in MODELS:
        cells = []
        for f in FOLDS:
            if is_done(m, f):
                n_done += 1
                try:
                    d = json.loads((run_dir(m, f) / "DONE").read_text())
                    cells.append(f"{d.get('lesion_dice', float('nan')):>16.4f}")
                except Exception:
                    cells.append(f"{'done':>16}")
            else:
                cells.append(f"{'-':>16}")
        print(f"{m:<12}" + "".join(cells))
    print(f"\n{n_done}/{len(MODELS)*len(FOLDS)} runs complete "
          f"(cells show held-out lesion Dice)")


def sh(cmd, cwd=None):
    print(f"\n$ {cmd}\n", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd)
    return p.returncode


def run_yolo(fold, force=False):
    from ultralytics import YOLO
    import numpy as np

    d = run_dir("yolo", fold)
    d.mkdir(parents=True, exist_ok=True)
    yaml = LOGO / "yolo" / f"fold_{fold}.yaml"
    if not yaml.exists():
        raise SystemExit(f"missing {yaml}")

    weights = d / "train" / "weights" / "best.pt"
    if weights.exists() and not force:
        print(f"  weights already present, skipping training")
    else:
        model = YOLO(YOLO_MODEL)
        model.train(data=str(yaml), epochs=YOLO_EPOCHS, imgsz=IMGSZ,
                    batch=YOLO_BATCH, patience=YOLO_PATIENCE, seed=SEED,
                    device=0, workers=8, project=str(d), name="train",
                    exist_ok=True, max_det=300, plots=True, val=True,
                    close_mosaic=10)

    import logo_eval
    logo_eval.SEED = SEED
    from logo_eval import evaluate_yolo
    return evaluate_yolo(fold, weights, d)


def run_unet(fold, force=False):
    import logo_eval
    logo_eval.SEED = SEED
    from logo_eval import train_unet_fold, evaluate_unet
    d = run_dir("unet", fold)
    d.mkdir(parents=True, exist_ok=True)
    ckpt = d / "best.pt"
    if ckpt.exists() and not force:
        print("  checkpoint already present, skipping training")
    else:
        train_unet_fold(fold, d)
    return evaluate_unet(fold, ckpt, d)


def run_maskrcnn(fold, force=False):
    import logo_eval
    logo_eval.SEED = SEED
    from logo_eval import train_maskrcnn_fold, evaluate_maskrcnn
    d = run_dir("maskrcnn", fold)
    d.mkdir(parents=True, exist_ok=True)
    ckpt = d / "best.pt"
    if ckpt.exists() and not force:
        print("  checkpoint already present, skipping training")
    else:
        train_maskrcnn_fold(fold, d)
    return evaluate_maskrcnn(fold, ckpt, d)


RUNNERS = {"yolo": run_yolo, "unet": run_unet, "maskrcnn": run_maskrcnn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, help="run only this architecture")
    ap.add_argument("--fold", choices=FOLDS, help="run only this fold")
    ap.add_argument("--force", action="store_true", help="retrain even if done")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)

    if args.status:
        status()
        return

    todo = [(m, f) for m, f in ORDER
            if (args.model is None or m == args.model)
            and (args.fold is None or f == args.fold)
            and (args.force or not is_done(m, f))]

    if not todo:
        print("Nothing to do — every requested run is already complete.")
        status()
        return

    print(f"{len(todo)} run(s) queued:")
    for m, f in todo:
        print(f"   {m:<10} {f}")

    t_start = time.time()
    for i, (m, f) in enumerate(todo, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(todo)}]  {m}  |  held-out group: {f}")
        print("=" * 70, flush=True)
        t0 = time.time()
        try:
            info = RUNNERS[m](f, force=args.force)
            mark_done(m, f, info or {})
            print(f"\n  done in {(time.time()-t0)/60:.1f} min")
        except KeyboardInterrupt:
            print("\ninterrupted — completed runs are preserved, re-run to continue")
            return
        except Exception as e:
            print(f"\n  !! {m}/{f} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("  continuing with the next run")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.0f} min")
    status()


if __name__ == "__main__":
    main()
