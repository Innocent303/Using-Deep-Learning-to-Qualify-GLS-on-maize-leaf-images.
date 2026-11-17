import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


DRIVE = Path("/content/drive/MyDrive")

MANIFEST = DRIVE / "logo_groups.csv"
OUT = DRIVE / "logo"

VAL_FRACTION = 0.15
SEED = 42

UNET_ROOT = DRIVE / "unet"
COCO_ROOT = DRIVE / "coco_split_800sq_aligned"
YOLO_ROOT = DRIVE / "yolo_seg_800sq"

POOL = DRIVE / "logo_pool"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_file(roots_patterns, stem, exts):
    for root, pattern in roots_patterns:
        for split in SPLITS:
            d = root / pattern.format(split=split)
            if not d.is_dir():
                continue
            for ext in exts:
                p = d / f"{stem}{ext}"
                if p.exists():
                    return p
            for p in d.iterdir():
                if p.is_file() and p.stem.lower() == stem.lower() \
                        and p.suffix.lower() in exts:
                    return p
    return None


def assign_folds(df, groups):
    import numpy as np
    folds = {}
    for held in groups:
        rng = np.random.default_rng(abs(hash((SEED, held))) % (2**32))
        split_of = {}
        pool_rows = []
        for _, r in df.iterrows():
            if r["group"] == held:
                split_of[r["stem"]] = "test"
            else:
                pool_rows.append(r)

        pool = pd.DataFrame(pool_rows)
        for g, sub in pool.groupby("group"):
            stems = sorted(sub["stem"].astype(str))
            rng.shuffle(stems)
            n_val = int(round(len(stems) * VAL_FRACTION))
            for s in stems[:n_val]:
                split_of[s] = "val"
            for s in stems[n_val:]:
                split_of[s] = "train"
        folds[held] = split_of
    return folds


def main():
    if not MANIFEST.exists():
        sys.exit(f"Missing {MANIFEST} — run build_group_manifest.py first.")

    df = pd.read_csv(MANIFEST, dtype={"stem": str})
    df["stem"] = df["stem"].astype(str)
    groups = sorted(df["group"].unique())
    print(f"manifest: {len(df)} images, {len(groups)} groups")
    for g in groups:
        print(f"   {g:<20} {int((df['group'] == g).sum()):>4}")

    folds = assign_folds(df, groups)

    print("\n" + "=" * 60)
    print("FOLDS")
    print("=" * 60)
    print(f"{'hold out':<20}{'train':>7}{'val':>6}{'test':>7}")
    print("-" * 40)
    rows = []
    for held in groups:
        c = Counter(folds[held].values())
        print(f"{held:<20}{c['train']:>7}{c['val']:>6}{c['test']:>7}")
        for stem, sp in folds[held].items():
            rows.append({"fold": held, "stem": stem, "split": sp})
    folds_df = pd.DataFrame(rows).merge(df[["stem", "group"]], on="stem", how="left")

    bad = 0
    for held in groups:
        sub = folds_df[folds_df["fold"] == held]
        test_stems = set(sub[sub["split"] == "test"]["stem"])
        other = set(sub[sub["split"] != "test"]["stem"])
        if test_stems & other:
            print(f"  !! {held}: {len(test_stems & other)} image(s) in test AND train/val")
            bad += 1
        wrong = sub[(sub["split"] == "test") & (sub["group"] != held)]
        if len(wrong):
            print(f"  !! {held}: {len(wrong)} test image(s) not from the held-out group")
            bad += 1
        if len(sub) != len(df):
            print(f"  !! {held}: covers {len(sub)} images, expected {len(df)}")
            bad += 1
    print("\n  fold integrity: " + ("OK" if bad == 0 else f"{bad} PROBLEM(S)"))

    print("\nval composition per fold (should span all training groups):")
    for held in groups:
        sub = folds_df[(folds_df["fold"] == held) & (folds_df["split"] == "val")]
        comp = ", ".join(f"{g} {n}" for g, n in sorted(sub["group"].value_counts().items()))
        print(f"  {held:<20} {comp}")

    OUT.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(OUT / "folds.csv", index=False)
    print(f"\nwrote {OUT / 'folds.csv'}")

    print("\n" + "=" * 60)
    print("BUILDING THE SHARED POOL")
    print("=" * 60)
    import shutil

    pool_img = POOL / "images"
    pool_lbl = POOL / "labels"
    pool_msk = POOL / "masks"
    for d in (pool_img, pool_lbl, pool_msk):
        d.mkdir(parents=True, exist_ok=True)

    img_sources = [(YOLO_ROOT, "{split}/images"), (COCO_ROOT, "{split}"),
                   (UNET_ROOT, "{split}/images")]
    lbl_sources = [(YOLO_ROOT, "{split}/labels")]
    msk_sources = [(UNET_ROOT, "{split}/labels")]

    missing = {"image": [], "label": [], "mask": []}
    for stem in df["stem"]:
        for kind, srcs, dst, exts in [
            ("image", img_sources, pool_img, IMAGE_EXTS),
            ("label", lbl_sources, pool_lbl, {".txt"}),
            ("mask", msk_sources, pool_msk, {".png", ".PNG"}),
        ]:
            existing = list(dst.glob(f"{stem}.*"))
            if existing:
                continue
            src = find_file(srcs, stem, exts)
            if src is None:
                missing[kind].append(stem)
            else:
                shutil.copy2(src, dst / src.name)

    for kind, lst in missing.items():
        n_have = len(df) - len(lst)
        print(f"  {kind:<6}: {n_have}/{len(df)} in pool"
              + (f"   MISSING {len(lst)}" if lst else ""))
        for s in lst[:6]:
            print(f"      missing {kind} for {s}")

    if missing["image"]:
        sys.exit("\nCannot continue without every image. Check the source paths.")

    yolo_dir = OUT / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    have_labels = not missing["label"]
    if have_labels:
        ext_of = {p.stem: p.suffix for p in pool_img.iterdir() if p.is_file()}
        for held in groups:
            for sp in SPLITS:
                stems = [s for s, v in folds[held].items() if v == sp]
                lines = [str(pool_img / f"{s}{ext_of[s]}") for s in sorted(stems)]
                (yolo_dir / f"fold_{held}_{sp}.txt").write_text("\n".join(lines) + "\n")
            (yolo_dir / f"fold_{held}.yaml").write_text(
                f"# LOGO fold — held-out group: {held}\n"
                f"path: {POOL}\n"
                f"train: {yolo_dir / f'fold_{held}_train.txt'}\n"
                f"val: {yolo_dir / f'fold_{held}_val.txt'}\n"
                f"test: {yolo_dir / f'fold_{held}_test.txt'}\n"
                f"nc: 2\n"
                f"names:\n  0: leaf\n  1: lesion\n")
        print(f"\nYOLO: {len(groups)*3} list(s) + {len(groups)} yaml(s) in {yolo_dir}")
    else:
        print("\nYOLO: skipped (labels missing from the pool)")

    unet_dir = OUT / "unet"
    unet_dir.mkdir(exist_ok=True)
    for held in groups:
        for sp in SPLITS:
            stems = sorted(s for s, v in folds[held].items() if v == sp)
            (unet_dir / f"fold_{held}_{sp}.txt").write_text("\n".join(stems) + "\n")
    print(f"U-Net: {len(groups)*3} stem list(s) in {unet_dir}")
    if missing["mask"]:
        print(f"  !! {len(missing['mask'])} mask(s) missing — U-Net folds "
              f"will be incomplete")

    coco_dir = OUT / "coco"
    coco_dir.mkdir(exist_ok=True)

    all_images, all_anns, categories = {}, [], None
    for sp in SPLITS:
        f = COCO_ROOT / "annotations" / f"instances_{sp}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        categories = categories or d["categories"]
        by_id = {im["id"]: im for im in d["images"]}
        for im in d["images"]:
            all_images[Path(im["file_name"]).stem] = im
        for a in d["annotations"]:
            if a["image_id"] in by_id:
                all_anns.append((Path(by_id[a["image_id"]]["file_name"]).stem, a))

    if not all_images:
        print(f"\nCOCO: skipped — no annotations found under {COCO_ROOT}")
    else:
        anns_by_stem = {}
        for stem, a in all_anns:
            anns_by_stem.setdefault(stem, []).append(a)
        print(f"\nCOCO source: {len(all_images)} images, {len(all_anns)} annotations")

        absent = [s for s in df["stem"] if s not in all_images]
        if absent:
            print(f"  !! {len(absent)} manifest image(s) absent from the COCO "
                  f"annotations: {absent[:6]}")

        for held in groups:
            for sp in SPLITS:
                stems = sorted(s for s, v in folds[held].items()
                               if v == sp and s in all_images)
                imgs, anns = [], []
                nid, aid = 1, 1
                for s in stems:
                    rec = dict(all_images[s])
                    rec["id"] = nid
                    imgs.append(rec)
                    for a in anns_by_stem.get(s, []):
                        na = dict(a)
                        na["id"] = aid
                        na["image_id"] = nid
                        anns.append(na)
                        aid += 1
                    nid += 1
                (coco_dir / f"fold_{held}_{sp}.json").write_text(json.dumps({
                    "info": {"description": f"LOGO fold, held-out group {held}",
                             "split": sp},
                    "licenses": [], "images": imgs, "annotations": anns,
                    "categories": categories}))
        print(f"COCO: {len(groups)*3} annotation file(s) in {coco_dir}")

        held = groups[0]
        tot_i = tot_a = 0
        seen = set()
        for sp in SPLITS:
            d = json.loads((coco_dir / f"fold_{held}_{sp}.json").read_text())
            ids = {im["id"] for im in d["images"]}
            assert all(a["image_id"] in ids for a in d["annotations"]), "dangling ann"
            names = {im["file_name"] for im in d["images"]}
            assert not (seen & names), "filename in two splits"
            seen |= names
            tot_i += len(d["images"])
            tot_a += len(d["annotations"])
        print(f"  verified fold '{held}': {tot_i} images, {tot_a} annotations, "
              f"splits disjoint")

    print("\n" + "=" * 60)
    print(f"Pool:    {POOL}")
    print(f"Folds:   {OUT}")
    print(f"Seed {SEED} — re-running reproduces identical folds.")


if __name__ == "__main__":
    main()
