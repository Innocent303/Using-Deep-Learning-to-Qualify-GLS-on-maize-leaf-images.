import hashlib
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


SRC = Path("/content/drive/MyDrive/yolo_seg")
DST = Path("/content/drive/MyDrive/yolo_seg_split")

SOURCE_SPLITS = ["train", "val"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

DUPLICATE_KEEP = "more"

CLASS_NAMES = {0: "leaf", 1: "lesion"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DRY_RUN = True


def find_dirs(root, split):
    a = root / split / "images", root / split / "labels"
    if a[0].is_dir():
        return a
    b = root / "images" / split, root / "labels" / split
    if b[0].is_dir():
        return b
    return None, None


def count_instances(path):
    try:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def file_hash(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def main():
    if not SRC.is_dir():
        sys.exit(f"Not a folder: {SRC}")

    samples = {}
    conflicts = []
    unpaired = []

    for split in SOURCE_SPLITS:
        img_dir, lbl_dir = find_dirs(SRC, split)
        if img_dir is None:
            print(f"{split}: not found, skipping")
            continue

        images = {p.stem: p for p in sorted(img_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not p.name.startswith(".")}
        labels = {p.stem: p for p in sorted(lbl_dir.iterdir())
                  if p.is_file() and p.suffix.lower() == ".txt"
                  and not p.name.startswith(".")}

        print(f"{split}: {len(images)} images, {len(labels)} labels")

        for stem in sorted(set(images) | set(labels)):
            if stem not in images or stem not in labels:
                unpaired.append(f"{split}/{stem}")
                continue

            if stem not in samples:
                samples[stem] = (images[stem], labels[stem], split)
                continue

            prev_img, prev_lbl, prev_split = samples[stem]
            same = file_hash(prev_lbl) == file_hash(labels[stem])
            n_prev, n_new = count_instances(prev_lbl), count_instances(labels[stem])
            conflicts.append((stem, prev_split, n_prev, split, n_new, same))

            if same:
                continue
            if DUPLICATE_KEEP == "more" and n_new > n_prev:
                samples[stem] = (images[stem], labels[stem], split)
            elif DUPLICATE_KEEP == split:
                samples[stem] = (images[stem], labels[stem], split)

    print(f"\nunique samples: {len(samples)}")

    if unpaired:
        print(f"\n{len(unpaired)} unpaired file(s) — EXCLUDED:")
        for u in unpaired[:8]:
            print(f"    {u}")
        if len(unpaired) > 8:
            print(f"    ... and {len(unpaired) - 8} more")

    if conflicts:
        identical = sum(1 for c in conflicts if c[5])
        differing = [c for c in conflicts if not c[5]]
        print(f"\n{len(conflicts)} filename(s) present in both source splits:")
        print(f"    {identical} with IDENTICAL labels (safe, kept once)")
        if differing:
            print(f"    {len(differing)} with DIFFERENT labels "
                  f"— resolved by DUPLICATE_KEEP='{DUPLICATE_KEEP}':")
            for stem, sa, na, sb, nb, _ in differing[:10]:
                kept = sb if (DUPLICATE_KEEP == "more" and nb > na) or \
                             DUPLICATE_KEEP == sb else sa
                print(f"        {stem}: {sa}={na} inst, {sb}={nb} inst  -> kept {kept}")
            if len(differing) > 10:
                print(f"        ... and {len(differing) - 10} more")
            print("      Differing labels mean the two copies were annotated")
            print("      separately. Worth checking a couple by hand.")

    if not samples:
        sys.exit("Nothing to split.")

    stems = sorted(samples)
    random.Random(SEED).shuffle(stems)

    n = len(stems)
    counts = {s: int(n * RATIOS[s]) for s in RATIOS}
    biggest = max(RATIOS, key=lambda s: RATIOS[s])
    counts[biggest] += n - sum(counts.values())

    assign, i = {}, 0
    for s in RATIOS:
        for stem in stems[i:i + counts[s]]:
            assign[stem] = s
        i += counts[s]

    print(f"\n{'split':>8} {'images':>8} {'%':>7} {'instances':>11} " +
          " ".join(f"{nm:>9}" for nm in CLASS_NAMES.values()))
    print("-" * 62)
    for s in RATIOS:
        members = [k for k in stems if assign[k] == s]
        cc = Counter()
        for k in members:
            try:
                for line in samples[k][1].read_text().splitlines():
                    if line.strip():
                        cc[int(float(line.split()[0]))] += 1
            except (OSError, ValueError):
                pass
        print(f"{s:>8} {len(members):>8} {100*len(members)/n:>6.1f}% "
              f"{sum(cc.values()):>11} " +
              " ".join(f"{cc.get(c, 0):>9}" for c in sorted(CLASS_NAMES)))

    if DRY_RUN:
        print(f"\nDry run — nothing written. Output would go to {DST}")
        print("Set DRY_RUN = False to apply.")
        return

    for s in RATIOS:
        (DST / s / "images").mkdir(parents=True, exist_ok=True)
        (DST / s / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    for stem in stems:
        img, lbl, _ = samples[stem]
        s = assign[stem]
        shutil.copy2(img, DST / s / "images" / img.name)
        shutil.copy2(lbl, DST / s / "labels" / lbl.name)
        copied += 2

    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items()))
    (DST / "data.yaml").write_text(
        f"path: {DST}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    print(f"\ncopied {copied} file(s)")
    print(f"wrote {DST / 'data.yaml'}")

    print("\nVERIFYING")
    ok = True
    seen = set()
    for s in RATIOS:
        imgs = {p.stem for p in (DST / s / "images").iterdir() if p.is_file()}
        lbls = {p.stem for p in (DST / s / "labels").iterdir() if p.is_file()}
        if imgs != lbls:
            print(f"  {s}: image/label mismatch "
                  f"({len(imgs - lbls)} unpaired images, "
                  f"{len(lbls - imgs)} unpaired labels)")
            ok = False
        if seen & imgs:
            print(f"  {s}: {len(seen & imgs)} filename(s) also in another split!")
            ok = False
        seen |= imgs
        print(f"  {s:>6}: {len(imgs)} images, {len(lbls)} labels")

    if len(seen) != n:
        print(f"  total {len(seen)} != expected {n}")
        ok = False

    print("  all checks passed" if ok else "  PROBLEMS FOUND")
    print(f"\nOutput: {DST}")
    print(f"Seed {SEED} — same as the U-Net and COCO splits.")


if __name__ == "__main__":
    main()
