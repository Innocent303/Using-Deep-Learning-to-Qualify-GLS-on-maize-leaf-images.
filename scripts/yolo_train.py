import os
import json
import shutil
import sys
import time
from pathlib import Path


DATA_YAML = Path("/content/drive/MyDrive/yolo_seg_800sq/data.yaml")

MODEL = "yolo11m-seg.pt"

IMGSZ = 800
EPOCHS = 150
BATCH = 16
PATIENCE = 20
SEED = int(os.environ.get("GLS_SEED", 1))
MAX_DET = 300
WORKERS = 8
DEVICE = 0

PROJECT = Path("/content/drive/MyDrive/yolo_seg_800sq/runs")
NAME = f"yolo11m_seg_800_seed{SEED}"

CACHE_TO_LOCAL = True
LOCAL_CACHE = Path("/content/yolo_cache")


def cache_dataset(yaml_path: Path) -> Path:
    if not CACHE_TO_LOCAL:
        return yaml_path

    src_root = yaml_path.parent

    def cache_ok():
        if not (LOCAL_CACHE / "data.yaml").exists():
            return False
        for split in ("train", "val", "test"):
            d = LOCAL_CACHE / split / "images"
            src = src_root / split / "images"
            if src.is_dir():
                if not d.is_dir():
                    return False
                if len(list(d.glob("*"))) < len(list(src.glob("*"))):
                    return False
        return True

    if LOCAL_CACHE.exists() and not cache_ok():
        print(f"Cache at {LOCAL_CACHE} is incomplete — rebuilding")
        shutil.rmtree(LOCAL_CACHE, ignore_errors=True)

    if LOCAL_CACHE.exists():
        print(f"Using existing cache at {LOCAL_CACHE}")
    else:
        print(f"Caching dataset to {LOCAL_CACHE}...")
        t0 = time.time()
        shutil.copytree(src_root, LOCAL_CACHE,
                        ignore=shutil.ignore_patterns("runs", "*.pt"))
        print(f"  done in {time.time() - t0:.0f}s")

    local_yaml = LOCAL_CACHE / "data.yaml"
    lines = []
    for line in local_yaml.read_text().splitlines():
        lines.append(f"path: {LOCAL_CACHE}" if line.strip().startswith("path:")
                     else line)
    local_yaml.write_text("\n".join(lines) + "\n")
    return local_yaml


def main():
    try:
        from ultralytics import YOLO
        import ultralytics
    except ImportError:
        sys.exit("ultralytics missing:  !pip install -q ultralytics")

    print(f"ultralytics {ultralytics.__version__}")

    if not DATA_YAML.exists():
        sys.exit(f"Not found: {DATA_YAML}\nRun yolo_resize_square.py first.")

    print(f"\ndata.yaml:\n{DATA_YAML.read_text()}")
    data_yaml = cache_dataset(DATA_YAML)

    pretrained = MODEL.endswith(".pt")
    print(f"\nmodel: {MODEL}  ({'PRETRAINED (COCO)' if pretrained else 'FROM SCRATCH'})")
    print(f"imgsz {IMGSZ}, batch {BATCH}, epochs {EPOCHS}, patience {PATIENCE}\n")

    model = YOLO(MODEL)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        max_det=MAX_DET,
        plots=True,
        val=True,
        close_mosaic=10,
    )

    run_dir = PROJECT / NAME
    best = run_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}")

    print("\n" + "=" * 60)
    print("VALIDATION (val split)")
    print("=" * 60)
    m = YOLO(str(best))
    metrics = m.val(data=str(data_yaml), split="val", imgsz=IMGSZ,
                    max_det=MAX_DET, project=str(PROJECT), name=f"{NAME}_val",
                    exist_ok=True, plots=True)

    names = metrics.names if hasattr(metrics, "names") else {}
    summary = {
        "model": MODEL,
        "pretrained": pretrained,
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "batch": BATCH,
        "seed": SEED,
        "box_mAP50_95": float(metrics.box.map),
        "box_mAP50": float(metrics.box.map50),
        "mask_mAP50_95": float(metrics.seg.map),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP75": float(metrics.seg.map75),
    }

    print(f"\n  box  mAP50-95 {metrics.box.map:.4f}   mAP50 {metrics.box.map50:.4f}")
    print(f"  mask mAP50-95 {metrics.seg.map:.4f}   mAP50 {metrics.seg.map50:.4f}"
          f"   mAP75 {metrics.seg.map75:.4f}")

    try:
        per_class = {}
        for i, ap in enumerate(metrics.seg.maps):
            nm = names.get(i, f"class{i}")
            per_class[nm] = float(ap)
            print(f"    mask AP {nm:<10} {ap:.4f}")
        summary["mask_AP_per_class"] = per_class
    except Exception as e:
        print(f"  (per-class AP unavailable: {type(e).__name__})")

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {out}")
    print(f"curves and plots: {run_dir}")

    print("\nNext: convert YOLO instance predictions to semantic maps and")
    print("compare lesion Dice + severity against the other two models.")
    print("Do NOT run on the test split until the comparison is set up.")


if __name__ == "__main__":
    main()
