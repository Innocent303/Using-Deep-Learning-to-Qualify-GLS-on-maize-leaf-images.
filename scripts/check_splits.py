import json
import sys
from pathlib import Path


DATASETS = {
    "U-Net": {
        "kind": "folders",
        "root": Path("/content/drive/MyDrive/unet"),
        "pattern": "{split}/images",
    },
    "Mask R-CNN": {
        "kind": "coco",
        "root": Path("/content/drive/MyDrive/coco_split_800sq_aligned"),
    },
    "YOLO": {
        "kind": "folders",
        "root": Path("/content/drive/MyDrive/yolo_seg_800sq"),
        "pattern": "{split}/images",
    },
}

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def stems_folders(root, pattern, split):
    d = root / pattern.format(split=split)
    if not d.is_dir():
        return None
    return {p.stem.lower() for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            and not p.name.startswith(".")}


def stems_coco(root, split):
    f = root / "annotations" / f"instances_{split}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {Path(im["file_name"]).stem.lower() for im in d.get("images", [])}


def main():
    data = {}
    for name, cfg in DATASETS.items():
        per_split = {}
        for s in SPLITS:
            if cfg["kind"] == "coco":
                st = stems_coco(cfg["root"], s)
            else:
                st = stems_folders(cfg["root"], cfg["pattern"], s)
            per_split[s] = st
        data[name] = per_split

    print("SPLIT SIZES")
    print(f"{'model':>14}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'total':>10}")
    print("-" * 56)
    for name, per in data.items():
        sizes = [len(per[s]) if per[s] is not None else -1 for s in SPLITS]
        tot = sum(x for x in sizes if x >= 0)
        cells = "".join(f"{(x if x >= 0 else 'n/a'):>10}" for x in sizes)
        print(f"{name:>14}{cells}{tot:>10}")

    names = [n for n in data if all(data[n][s] is not None for s in SPLITS)]
    if len(names) < 2:
        sys.exit("\nNeed at least two readable datasets.")

    print("\n\nARE THE SPLITS IDENTICAL?")
    print("-" * 56)
    identical = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            same = all(data[a][s] == data[b][s] for s in SPLITS)
            print(f"  {a} vs {b}: {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                identical = False
                for s in SPLITS:
                    sa, sb = data[a][s], data[b][s]
                    inter = len(sa & sb)
                    exp = len(sa) * len(sb) / max(
                        len(set().union(*(data[a][x] for x in SPLITS))), 1)
                    print(f"      {s:>5}: {inter} shared of {len(sa)}/{len(sb)}"
                          f"   (chance would give ~{exp:.0f})")

    if identical:
        print("\nAll splits match. Paired cross-model comparison is valid.")
        return

    print("\n\nLEAKAGE CHECK — is one model's TEST set another's TRAINING set?")
    print("-" * 56)
    worst = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            bad = data[a]["test"] & data[b]["train"]
            if bad:
                worst = max(worst, len(bad))
                print(f"  {len(bad):>3} of {a}'s {len(data[a]['test'])} test images "
                      f"are in {b}'s TRAINING set")

    common_test = set.intersection(*(data[n]["test"] for n in names))
    print(f"\n  images in the test set of ALL models: {len(common_test)}")

    print("\n" + "=" * 62)
    print("VERDICT")
    print("=" * 62)
    print("  The splits differ, so a paired comparison across models is not")
    print("  valid as it stands:")
    print(f"    - only {len(common_test)} image(s) are held out by every model")
    print(f"    - up to {worst} of one model's test images were TRAINING data")
    print("      for another")
    print("\n  Options:")
    print("    1. Rebuild all three datasets from ONE canonical split, then")
    print("       retrain. Correct, but costs a retrain of each model.")
    print("    2. Report each model on its own test set and compare UNPAIRED,")
    print("       stating clearly that the splits differ.")
    print("    3. Compare on the common held-out images only — valid but")
    print(f"       n={len(common_test)} is too small to conclude much.")


if __name__ == "__main__":
    main()
