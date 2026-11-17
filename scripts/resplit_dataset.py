import os
import random
import shutil
import sys
from pathlib import Path


ROOT = "/content/drive/MyDrive/unet"

SPLITS = ["train", "val", "test"]
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "labels"

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

SEED = 42

MASK_SUFFIX = ""

DRY_RUN = True


def stem_key(p: Path, suffix: str = "") -> str:
    s = p.stem
    if suffix and s.endswith(suffix):
        s = s[: -len(suffix)]
    return s.lower()


def listdir(folder: Path) -> dict:
    if not folder.is_dir():
        return {}
    return {p for p in sorted(folder.glob("*"))
            if p.is_file() and not p.is_symlink() and not p.name.startswith(".")}


def main():
    root = Path(ROOT)
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}\n"
                 "Is Drive mounted?  from google.colab import drive; drive.mount('/content/drive')")

    total_ratio = sum(RATIOS.values())
    if abs(total_ratio - 1.0) > 1e-6:
        sys.exit(f"RATIOS must sum to 1.0, got {total_ratio}")
    if set(RATIOS) != set(SPLITS):
        sys.exit("RATIOS and SPLITS must cover the same split names.")

    samples = {}
    orphans = []

    for split in SPLITS:
        img_dir = root / split / IMAGES_SUBDIR
        mask_dir = root / split / MASKS_SUBDIR

        images = {stem_key(p): p for p in listdir(img_dir)}
        masks = {stem_key(p, MASK_SUFFIX): p for p in listdir(mask_dir)}

        for k in sorted(set(images) | set(masks)):
            if k in images and k in masks:
                if k in samples:
                    print(f"  ! duplicate sample '{k}' in both "
                          f"{samples[k][2]} and {split} — skipping")
                    continue
                samples[k] = (images[k], masks[k], split)
            elif k in images:
                orphans.append(f"{split}: {images[k].name} has no mask")
            else:
                orphans.append(f"{split}: {masks[k].name} has no image")

    n = len(samples)
    print(f"Found {n} complete image+mask pair(s) across {', '.join(SPLITS)}.")
    if orphans:
        print(f"\n{len(orphans)} unpaired file(s) — these will NOT be moved:")
        for o in orphans[:10]:
            print(f"    {o}")
        if len(orphans) > 10:
            print(f"    ... and {len(orphans) - 10} more")
        print("  Fix these before splitting, or they will sit in the wrong split.")

    if n == 0:
        sys.exit("\nNothing to split.")

    keys = sorted(samples)
    random.Random(SEED).shuffle(keys)

    counts = {s: int(n * RATIOS[s]) for s in SPLITS}
    biggest = max(RATIOS, key=lambda s: RATIOS[s])
    counts[biggest] += n - sum(counts.values())

    print("\nTarget split:")
    for s in SPLITS:
        print(f"  {s:<6} {counts[s]:>5}  ({counts[s] / n:.1%})")

    assignment = {}
    i = 0
    for s in SPLITS:
        for k in keys[i: i + counts[s]]:
            assignment[k] = s
        i += counts[s]

    moves = []
    staying = 0
    for k, target in assignment.items():
        img, mask, current = samples[k]
        if current == target:
            staying += 1
            continue
        moves.append((img, root / target / IMAGES_SUBDIR / img.name))
        moves.append((mask, root / target / MASKS_SUBDIR / mask.name))

    print(f"\n{staying} sample(s) already in the right split.")
    print(f"{len(moves) // 2} sample(s) to move  ({len(moves)} files).")

    collisions = [dst for _, dst in moves if dst.exists()]
    if collisions:
        print(f"\nABORTING — {len(collisions)} destination file(s) already exist:")
        for c in collisions[:10]:
            print(f"    {c}")
        sys.exit("Resolve these first; nothing was moved.")

    if DRY_RUN:
        print("\nExamples of planned moves:")
        for src, dst in moves[:8]:
            print(f"  {src.parent.parent.name}/{src.parent.name}/{src.name}"
                  f"  ->  {dst.parent.parent.name}/{dst.parent.name}/")
        if len(moves) > 8:
            print(f"  ... and {len(moves) - 8} more")
        print("\nDry run — nothing moved. Set DRY_RUN = False to apply.")
        return

    for split in SPLITS:
        (root / split / IMAGES_SUBDIR).mkdir(parents=True, exist_ok=True)
        (root / split / MASKS_SUBDIR).mkdir(parents=True, exist_ok=True)

    moved = 0
    for src, dst in moves:
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except OSError as e:
            print(f"  ! failed to move {src}: {e}", file=sys.stderr)

    print(f"\nMoved {moved} file(s).")

    print("\nFinal counts:")
    for s in SPLITS:
        ni = len(listdir(root / s / IMAGES_SUBDIR))
        nm = len(listdir(root / s / MASKS_SUBDIR))
        flag = "" if ni == nm else "   <-- images and masks disagree!"
        print(f"  {s:<6} images {ni:>5}   masks {nm:>5}{flag}")

    print(f"\nSplit seed was {SEED} — record it to reproduce this split.")


if __name__ == "__main__":
    main()
