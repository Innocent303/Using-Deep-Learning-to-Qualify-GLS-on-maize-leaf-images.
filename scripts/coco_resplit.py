import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/content/drive/MyDrive/coco")

SOURCES = [
    {"images": ROOT / "train", "json": ROOT / "instances_train.json"},
    {"images": ROOT / "val",   "json": ROOT / "instances_val.json"},
]

OUT = ROOT.parent / "coco_split"

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

COPY_IMAGES = True

DRY_RUN = True


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        sys.exit(f"Missing: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON: {e}")


def main():
    random.seed(SEED)

    datasets = []
    for src in SOURCES:
        d = load(src["json"])
        datasets.append((d, Path(src["images"])))
        print(f"{Path(src['json']).name}: {len(d.get('images', []))} images, "
              f"{len(d.get('annotations', []))} annotations")

    cat_sets = [json.dumps(sorted(((c["id"], c["name"]) for c in d.get("categories", [])))) 
                for d, _ in datasets]
    if len(set(cat_sets)) != 1:
        sys.exit("Categories differ between source files — reconcile them first.")
    categories = datasets[0][0]["categories"]
    print(f"categories: {[(c['id'], c['name']) for c in categories]}\n")

    merged_images = []
    merged_anns = []
    src_path_for = {}
    seen_names = {}
    next_img_id = 1
    next_ann_id = 1
    dup_names = []
    missing_files = []

    for d, img_dir in datasets:
        id_map = {}
        by_old_id = {im["id"]: im for im in d.get("images", [])}

        for im in d.get("images", []):
            fname = Path(im["file_name"]).name

            if fname in seen_names:
                dup_names.append(fname)
                id_map[im["id"]] = seen_names[fname]
                continue

            src_file = img_dir / fname
            if not src_file.exists():
                for _, other_dir in datasets:
                    alt = other_dir / fname
                    if alt.exists():
                        src_file = alt
                        break
            if not src_file.exists():
                missing_files.append(fname)
                continue

            new = dict(im)
            new["id"] = next_img_id
            new["file_name"] = fname
            merged_images.append(new)
            src_path_for[fname] = src_file
            seen_names[fname] = next_img_id
            id_map[im["id"]] = next_img_id
            next_img_id += 1

        for a in d.get("annotations", []):
            if a.get("image_id") not in id_map:
                continue
            new = dict(a)
            new["id"] = next_ann_id
            new["image_id"] = id_map[a["image_id"]]
            merged_anns.append(new)
            next_ann_id += 1

    print(f"merged: {len(merged_images)} images, {len(merged_anns)} annotations")
    if dup_names:
        print(f"  {len(dup_names)} filename(s) listed in both sources — kept once")
    if missing_files:
        print(f"  !! {len(missing_files)} image(s) in JSON but not found on disk:")
        for f in missing_files[:8]:
            print(f"       {f}")

    strays, redundant = [], []
    for _, img_dir in datasets:
        if img_dir.is_dir():
            for p in img_dir.iterdir():
                if not p.is_file() or p.name.startswith("."):
                    continue
                if p.name not in seen_names:
                    strays.append(p)
                elif src_path_for.get(p.name) != p:
                    redundant.append(p)
    if redundant:
        print(f"  {len(redundant)} redundant duplicate copy/copies on disk "
              f"(annotated in the other split) — not copied:")
        for p in redundant[:8]:
            print(f"       {p.parent.name}/{p.name}")
        if len(redundant) > 8:
            print(f"       ... and {len(redundant) - 8} more")
    if strays:
        print(f"  {len(strays)} file(s) on disk with no annotations at all "
              f"— NOT copied:")
        for p in strays[:8]:
            print(f"       {p.parent.name}/{p.name}")
        if len(strays) > 8:
            print(f"       ... and {len(strays) - 8} more")

    n = len(merged_images)
    if n == 0:
        sys.exit("Nothing to split.")

    ids = sorted(im["id"] for im in merged_images)
    random.Random(SEED).shuffle(ids)

    counts = {s: int(n * RATIOS[s]) for s in RATIOS}
    biggest = max(RATIOS, key=lambda s: RATIOS[s])
    counts[biggest] += n - sum(counts.values())

    assign = {}
    i = 0
    for s in RATIOS:
        for iid in ids[i:i + counts[s]]:
            assign[iid] = s
        i += counts[s]

    anns_by_img = defaultdict(list)
    for a in merged_anns:
        anns_by_img[a["image_id"]].append(a)

    cat_name = {c["id"]: c["name"] for c in categories}

    print(f"\n{'split':>8} {'images':>8} {'%':>7} {'anns':>8} " +
          " ".join(f"{cat_name[c['id']]:>10}" for c in categories))
    print("-" * 62)

    per_split = {}
    for s in RATIOS:
        imgs = [im for im in merged_images if assign[im["id"]] == s]
        anns = [a for im in imgs for a in anns_by_img[im["id"]]]
        cc = Counter(a["category_id"] for a in anns)
        per_split[s] = (imgs, anns)
        print(f"{s:>8} {len(imgs):>8} {100*len(imgs)/n:>6.1f}% {len(anns):>8} " +
              " ".join(f"{cc.get(c['id'], 0):>10}" for c in categories))

    per_img_counts = [len(anns_by_img[im["id"]]) for im in merged_images]
    if per_img_counts:
        pc = sorted(per_img_counts)
        print(f"\nannotations per image: min {pc[0]}, median {pc[len(pc)//2]}, "
              f"max {pc[-1]}")
        over = sum(1 for c in per_img_counts if c > 100)
        if over:
            print(f"  !! {over} image(s) have >100 instances. Mask R-CNN's default")
            print("     box_detections_per_img=100 will truncate these at inference.")
            print("     Raise it, or expect capped recall on dense images.")

    empty = sum(1 for im in merged_images if not anns_by_img[im["id"]])
    if empty:
        print(f"  {empty} image(s) have no annotations")

    if DRY_RUN:
        print(f"\nDry run — nothing written. Output would go to {OUT}")
        print("Set DRY_RUN = False to apply.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "annotations").mkdir(exist_ok=True)

    for s, (imgs, anns) in per_split.items():
        split_dir = OUT / s
        split_dir.mkdir(exist_ok=True)

        if COPY_IMAGES:
            for im in imgs:
                dst = split_dir / im["file_name"]
                if not dst.exists():
                    shutil.copy2(src_path_for[im["file_name"]], dst)

        out_json = {
            "info": datasets[0][0].get("info", {}),
            "licenses": datasets[0][0].get("licenses", []),
            "images": imgs,
            "annotations": anns,
            "categories": categories,
        }
        path = OUT / "annotations" / f"instances_{s}.json"
        path.write_text(json.dumps(out_json))
        print(f"wrote {path.name}: {len(imgs)} images, {len(anns)} annotations")

    print("\nVERIFYING OUTPUT")
    ok = True
    all_ids, all_names = set(), set()
    for s in RATIOS:
        d = load(OUT / "annotations" / f"instances_{s}.json")
        iids = [im["id"] for im in d["images"]]
        aids = [a["id"] for a in d["annotations"]]
        names = [im["file_name"] for im in d["images"]]

        if len(set(iids)) != len(iids):
            print(f"  {s}: duplicate image_id!"); ok = False
        if len(set(aids)) != len(aids):
            print(f"  {s}: duplicate annotation id!"); ok = False
        bad = [a["id"] for a in d["annotations"] if a["image_id"] not in set(iids)]
        if bad:
            print(f"  {s}: {len(bad)} annotation(s) reference a missing image!"); ok = False
        if all_ids & set(iids):
            print(f"  {s}: image_id overlaps another split!"); ok = False
        if all_names & set(names):
            print(f"  {s}: filename appears in another split!"); ok = False
        all_ids |= set(iids)
        all_names |= set(names)

        if COPY_IMAGES:
            on_disk = {p.name for p in (OUT / s).iterdir() if p.is_file()}
            miss = set(names) - on_disk
            if miss:
                print(f"  {s}: {len(miss)} image file(s) missing on disk!"); ok = False

    print("  all checks passed" if ok else "  PROBLEMS FOUND — do not use this output")
    print(f"\nOutput: {OUT}")
    print(f"Seed {SEED} — record it to reproduce this split.")


if __name__ == "__main__":
    main()
