import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = "/content/drive/MyDrive/unet"
SPLITS = ["train", "val", "test"]
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "labels"
MASK_SUFFIX = ""

SAMPLE_N = 60

SAVE_OVERLAY = "/content/mask_check.png"
OVERLAY_COUNT = 6


def stem_key(p, suffix=""):
    s = p.stem
    if suffix and s.endswith(suffix):
        s = s[: -len(suffix)]
    return s.lower()


def collect_pairs(root):
    pairs = []
    for split in SPLITS:
        img_dir, mask_dir = root / split / IMAGES_SUBDIR, root / split / MASKS_SUBDIR
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue
        images = {stem_key(p): p for p in sorted(img_dir.glob("*"))
                  if p.is_file() and not p.name.startswith(".")}
        masks = {stem_key(p, MASK_SUFFIX): p for p in sorted(mask_dir.glob("*"))
                 if p.is_file() and not p.name.startswith(".")}
        for k in sorted(set(images) & set(masks)):
            pairs.append((k, images[k], masks[k], split))
    return pairs


def main():
    root = Path(ROOT)
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")

    pairs = collect_pairs(root)
    if not pairs:
        sys.exit("No image/mask pairs found — check ROOT and the subdir names.")
    print(f"{len(pairs)} pair(s) found.\n")

    sample = pairs if SAMPLE_N is None else pairs[:: max(1, len(pairs) // SAMPLE_N)]
    sample = sample[:SAMPLE_N] if SAMPLE_N else sample

    modes = Counter()
    value_counts = Counter()
    per_image_values = Counter()
    rgb_colours = Counter()
    total_px = 0

    print(f"Scanning {len(sample)} mask(s)...\n")

    for k, img_p, mask_p, split in sample:
        try:
            with Image.open(mask_p) as m:
                mode = m.mode
                modes[mode] += 1
                if mode == "P":
                    arr = np.array(m)
                elif mode in ("RGB", "RGBA"):
                    arr = np.array(m.convert("RGB"))
                else:
                    arr = np.array(m.convert("L"))
        except Exception as e:
            print(f"  {k}: cannot read ({type(e).__name__})")
            continue

        if arr.ndim == 3:
            flat = arr.reshape(-1, 3)
            uniq, counts = np.unique(flat, axis=0, return_counts=True)
            for c, n in zip(uniq, counts):
                rgb_colours[tuple(int(x) for x in c)] += int(n)
            for c in uniq:
                per_image_values[tuple(int(x) for x in c)] += 1
            total_px += flat.shape[0]
        else:
            uniq, counts = np.unique(arr, return_counts=True)
            for v, n in zip(uniq, counts):
                value_counts[int(v)] += int(n)
            for v in uniq:
                per_image_values[int(v)] += 1
            total_px += arr.size

    print("Mask file modes:")
    for m, n in modes.most_common():
        label = {"L": "8-bit grayscale", "P": "palette", "RGB": "colour",
                 "RGBA": "colour + alpha", "1": "1-bit binary"}.get(m, m)
        print(f"  {m}  ({label}): {n} file(s)")

    if rgb_colours:
        print(f"\nStored as RGB. {len(rgb_colours)} distinct colour(s) found:")
        print(f"{'colour':>18} {'% of pixels':>12} {'in N images':>12}")
        for c, n in rgb_colours.most_common(20):
            print(f"{str(c):>18} {100 * n / total_px:>11.3f}% {per_image_values[c]:>12}")
        if len(rgb_colours) > 20:
            print(f"  ... and {len(rgb_colours) - 20} more colours")
        print("\n  -> You must map these colours to class indices 0..N-1")
        print("     before training. See COLOUR_TO_CLASS below.")
        if len(rgb_colours) > 40:
            print("  !! Many colours suggests JPEG-compressed or anti-aliased masks.")
            print("     Masks must be saved lossless (PNG). Check with the annotator.")
    else:
        vals = sorted(value_counts)
        print(f"\nStored as single-channel. {len(vals)} distinct value(s): {vals}")
        print(f"\n{'value':>8} {'% of pixels':>12} {'in N images':>12}")
        for v in vals:
            print(f"{v:>8} {100 * value_counts[v] / total_px:>11.3f}% "
                  f"{per_image_values[v]:>12}")

        if vals == list(range(len(vals))):
            print(f"\n  -> Already class indices 0..{len(vals) - 1}. "
                  f"NUM_CLASSES = {len(vals)}. Use directly with CrossEntropyLoss.")
        elif set(vals) <= {0, 255}:
            print("\n  -> Binary 0/255. Divide by 255 to get {0,1}.")
        else:
            print("\n  -> Values are not contiguous indices. Remap them, e.g.")
            print(f"     VALUE_TO_CLASS = {{{', '.join(f'{v}: {i}' for i, v in enumerate(vals))}}}")

    if value_counts or rgb_colours:
        src = value_counts or rgb_colours
        fracs = sorted((n / total_px for n in src.values()), reverse=True)
        if len(fracs) > 1 and fracs[-1] < 0.01:
            print(f"\n  !! Rarest class is {100 * fracs[-1]:.3f}% of pixels.")
            print("     Plain CrossEntropy will ignore it. Use class weights or Dice loss.")

    if SAVE_OVERLAY:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            picks = sample[:: max(1, len(sample) // OVERLAY_COUNT)][:OVERLAY_COUNT]
            fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 8))
            if len(picks) == 1:
                axes = axes.reshape(2, 1)

            for col, (k, img_p, mask_p, split) in enumerate(picks):
                img = ImageOps.exif_transpose(Image.open(img_p)).convert("RGB")
                msk = Image.open(mask_p)
                if msk.size != img.size:
                    msk = msk.resize(img.size, Image.NEAREST)
                img.thumbnail((512, 512))
                msk = msk.resize(img.size, Image.NEAREST)

                axes[0, col].imshow(img)
                axes[0, col].set_title(f"{k} ({split})", fontsize=9)
                axes[1, col].imshow(img)
                axes[1, col].imshow(np.array(msk.convert("L") if msk.mode != "RGB" else msk),
                                    alpha=0.5, cmap=None if msk.mode == "RGB" else "tab10",
                                    interpolation="nearest")
                for r in (0, 1):
                    axes[r, col].axis("off")

            plt.tight_layout()
            plt.savefig(SAVE_OVERLAY, dpi=90)
            print(f"\nOverlay preview written to {SAVE_OVERLAY}")
            print("Open it and confirm the masks sit on the right objects.")
        except ImportError:
            print("\nmatplotlib not available — skipped the overlay.")
        except Exception as e:
            print(f"\nOverlay failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
