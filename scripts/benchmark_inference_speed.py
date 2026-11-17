import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
for _c in [_HERE, Path.cwd(), Path("/content/drive/MyDrive"), Path("/content")]:
    if (Path(_c) / "logo_eval.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("Could not find logo_eval.py - put this script in the same "
             "folder, or run from /content/drive/MyDrive.")

import torch
from PIL import Image, ImageOps

import logo_eval as le

DEFAULT_SEED = 1
N_DEFAULT = 30
WARMUP_DEFAULT = 5
FOLD_FOR_STEMS = "delmas"
OUT_CSV = le.DRIVE / "runs" / "diagnostics" / "inference_speed.csv"

MODEL_DIRS = ["unet", "maskrcnn", "yolo"]
MODEL_LABELS = {"unet": "U-Net (ImageNet)", "maskrcnn": "Mask R-CNN",
                "yolo": "YOLO11m-seg"}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    sync()
    return time.perf_counter()


def run_dir(seed, model, fold):
    return le.DRIVE / f"logo_runs_seed{seed}" / model / fold


def load_threshold(model, fold, seed):
    import json
    p = run_dir(seed, model, fold) / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("score_threshold")
    except Exception:
        return None


def get_model(model_dir, seed, fold):
    if model_dir == "unet":
        ckpt = run_dir(seed, "unet", fold) / "best.pt"
        if not ckpt.exists():
            return None
        m = le._unet_model().to(le.DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=le.DEVICE,
                                     weights_only=False)["model"])
        m.eval()
        return m

    if model_dir == "maskrcnn":
        ckpt = run_dir(seed, "maskrcnn", fold) / "best.pt"
        if not ckpt.exists():
            return None
        m = le._mrcnn_model().to(le.DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=le.DEVICE,
                                     weights_only=False)["model"])
        m.eval()
        return m

    if model_dir == "yolo":
        ckpt = run_dir(seed, "yolo", fold) / "train" / "weights" / "best.pt"
        if not ckpt.exists():
            ckpt = run_dir(seed, "yolo", fold) / "weights" / "best.pt"
        if not ckpt.exists():
            return None
        from ultralytics import YOLO
        return YOLO(str(ckpt))

    raise ValueError(model_dir)


def bench_unet(model, stem):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    t0 = now()
    img = ImageOps.exif_transpose(Image.open(le.image_path(stem))).convert("RGB")
    w, h = img.size
    tf = A.Compose([A.Resize(le.IMGSZ, le.IMGSZ),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)), ToTensorV2()])
    x = tf(image=np.array(img))["image"].unsqueeze(0).to(le.DEVICE)
    t1 = now()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        logits = model(x)
    t2 = now()

    pred = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    if pred.shape != (h, w):
        pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))
    t3 = now()

    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def bench_maskrcnn(model, stem, thr):
    import torchvision

    t0 = now()
    img = Image.open(le.image_path(stem)).convert("RGB")
    w, h = img.size
    t = torchvision.transforms.functional.to_tensor(img).to(le.DEVICE)
    t1 = now()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        out = model([t])[0]
    t2 = now()

    _ = le._mrcnn_pred_semantic(out, h, w, thr if thr is not None else 0.5)
    t3 = now()

    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def bench_yolo(model, stem, thr):
    t0 = now()
    res = model.predict(str(le.image_path(stem)), imgsz=le.IMGSZ,
                        conf=min(le.SCORE_THRESHOLDS), iou=0.7, max_det=300,
                        retina_masks=True, verbose=False)[0]
    t1 = now()

    img = Image.open(le.image_path(stem))
    w, h = img.size
    _ = le._yolo_pred_semantic(res, h, w, thr if thr is not None else 0.2)
    t2 = now()

    speed = getattr(res, "speed", {})
    pre = speed.get("preprocess", float("nan"))
    inf = speed.get("inference", float("nan"))
    post_ultra = speed.get("postprocess", float("nan"))

    return (t1 - t0) * 1000, 0.0, (t2 - t1) * 1000, (pre, inf, post_ultra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! No GPU detected. Inference-speed numbers from a CPU runtime")
        print("   are not comparable to a GPU deployment and should not be")
        print("   reported as the project's inference-speed figures. Switch")
        print("   to a GPU runtime (Runtime > Change runtime type) and rerun.")
        print()
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    stems = le.fold_stems(FOLD_FOR_STEMS, "test")
    need = args.warmup + args.n
    if len(stems) < need:
        sys.exit(f"Only {len(stems)} images available in '{FOLD_FOR_STEMS}' "
                 f"test set; need {need} (warmup {args.warmup} + n {args.n}). "
                 f"Reduce --n or --warmup.")
    stems = stems[:need]

    rows = []
    for model_dir in MODEL_DIRS:
        label = MODEL_LABELS[model_dir]
        print(f"\n{label}: loading checkpoint (seed {args.seed})...")
        model = get_model(model_dir, args.seed, FOLD_FOR_STEMS)
        if model is None:
            print(f"  !! no checkpoint found, skipping")
            continue
        thr = load_threshold(model_dir, FOLD_FOR_STEMS, args.seed)

        print(f"  warmup ({args.warmup} images, discarded)...")
        for stem in stems[:args.warmup]:
            try:
                if model_dir == "unet":
                    bench_unet(model, stem)
                elif model_dir == "maskrcnn":
                    bench_maskrcnn(model, stem, thr)
                else:
                    bench_yolo(model, stem, thr)
            except Exception as e:
                print(f"    warmup failed on {stem}: {e}")

        print(f"  timing ({args.n} images)...")
        for stem in stems[args.warmup:]:
            try:
                if model_dir == "unet":
                    pre, inf, post = bench_unet(model, stem)
                elif model_dir == "maskrcnn":
                    pre, inf, post = bench_maskrcnn(model, stem, thr)
                else:
                    pre, inf, post, _ultra = bench_yolo(model, stem, thr)
                rows.append({"model": label, "stem": stem,
                            "preprocess_ms": pre, "inference_ms": inf,
                            "postprocess_ms": post,
                            "total_ms": pre + inf + post})
            except Exception as e:
                print(f"    failed on {stem}: {type(e).__name__}: {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        sys.exit("No timings collected.")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} timed images)")

    print("\n" + "=" * 78)
    print("INFERENCE SPEED  (ms per image, batch size 1)")
    print("=" * 78)
    summary = df.groupby("model").agg(
        n=("total_ms", "count"),
        preprocess_ms=("preprocess_ms", "mean"),
        inference_ms=("inference_ms", "mean"),
        postprocess_ms=("postprocess_ms", "mean"),
        mean_total_ms=("total_ms", "mean"),
        sd_total_ms=("total_ms", "std"),
        median_total_ms=("total_ms", "median"),
    ).round(2)
    summary["images_per_sec"] = (1000 / summary["mean_total_ms"]).round(2)
    print(summary.to_string())

    print("\n" + "=" * 78)
    print("NOTES")
    print("=" * 78)
    print("  - Batch size 1, matching how every model was actually run for")
    print("    the accuracy results in this project (one image at a time).")
    print(f"  - First {args.warmup} images discarded as GPU warmup.")
    print("  - YOLO's preprocess and inference are combined into one figure")
    print("    (Ultralytics does not expose them separately via the public")
    print("    predict() API); postprocess is the semantic-map conversion,")
    print("    comparable to the other two architectures.")
    print("  - All timings are from a single seed's checkpoint; inference")
    print("    speed is a property of the architecture, not the trained")
    print("    weights, so this does not need to be repeated across seeds.")
    print("  - Report the GPU model alongside these figures - they are only")
    print("    meaningful relative to each other on the SAME hardware.")


if __name__ == "__main__":
    main()
