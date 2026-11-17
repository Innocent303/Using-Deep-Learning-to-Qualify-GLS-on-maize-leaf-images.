import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


ROOT = Path("/content/drive/MyDrive/unet")
IMAGES_SUBDIR, MASKS_SUBDIR = "images", "labels"
MASK_SUFFIX = ""

NUM_CLASSES = 3
CLASS_NAMES = ["background", "leaf", "GLS_lesion"]

IMG_SIZE = 800
SEED = int(os.environ.get("GLS_SEED", 1))

PRETRAINED_ENCODER = "resnet34"

MODEL_TAG = "unet_scratch" if PRETRAINED_ENCODER is None \
    else f"unet_{PRETRAINED_ENCODER}_imagenet"

RUN_NAME = f"{MODEL_TAG}_seed{SEED}"

OVERFIT_N = 8
OVERFIT_EPOCHS = 150
OVERFIT_LR = 3e-4
OVERFIT_BATCH = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

assert IMG_SIZE % 16 == 0, (
    f"IMG_SIZE={IMG_SIZE} is not divisible by 16. The U-Net pools 4 times, so "
    "a non-divisible size makes the decoder's skip connections mismatched."
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stem_key(p: Path, suffix: str = "") -> str:
    s = p.stem
    if suffix and s.endswith(suffix):
        s = s[: -len(suffix)]
    return s.lower()


def list_pairs(split: str):
    img_dir = ROOT / split / IMAGES_SUBDIR
    mask_dir = ROOT / split / MASKS_SUBDIR
    images = {stem_key(p): p for p in sorted(img_dir.glob("*"))
              if p.is_file() and not p.name.startswith(".")}
    masks = {stem_key(p, MASK_SUFFIX): p for p in sorted(mask_dir.glob("*"))
             if p.is_file() and not p.name.startswith(".")}
    keys = sorted(set(images) & set(masks))
    return [(images[k], masks[k]) for k in keys]


class SegDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]

        img = ImageOps.exif_transpose(Image.open(img_p)).convert("RGB")
        mask = Image.open(mask_p)
        if mask.mode != "L":
            mask = mask.convert("L")

        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

        img = np.array(img)
        mask = np.array(mask)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


def train_transform(size=IMG_SIZE, augment=True):
    tf = [A.Resize(size, size)]
    if augment:
        tf += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                               rotate_limit=25, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
        ]
    tf += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(tf)


def eval_transform(size=IMG_SIZE):
    return train_transform(size, augment=False)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])

        self.head = nn.Conv2d(c[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_unet(num_classes=NUM_CLASSES, encoder=PRETRAINED_ENCODER):
    if encoder is None:
        return UNet(num_classes=num_classes)

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise SystemExit(
            "segmentation-models-pytorch is needed for a pretrained encoder:\n"
            "    !pip install -q segmentation-models-pytorch\n"
            "Or set PRETRAINED_ENCODER = None to use the scratch U-Net."
        )

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


def describe_model(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class DiceLoss(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)
        card = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, class_weights=None, ce_w=0.5, dice_w=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.ce_w, self.dice_w = ce_w, dice_w

    def forward(self, logits, target):
        return self.ce_w * self.ce(logits, target) + self.dice_w * self.dice(logits, target)


@torch.no_grad()
def per_class_dice(logits, target, num_classes=NUM_CLASSES):
    pred = logits.argmax(1)
    out = []
    for c in range(num_classes):
        p, t = (pred == c), (target == c)
        denom = p.sum() + t.sum()
        out.append(torch.tensor(float("nan")) if denom == 0
                   else (2.0 * (p & t).sum().float() / denom.float()))
    return torch.stack([o.to(logits.device) for o in out])


def overfit_test():
    seed_everything()
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pairs = list_pairs("train")
    print(f"train pairs: {len(pairs)}")
    if len(pairs) < OVERFIT_N:
        raise SystemExit("Not enough training pairs found — check ROOT.")

    subset = pairs[:OVERFIT_N]
    ds = SegDataset(subset, eval_transform())
    dl = DataLoader(ds, batch_size=OVERFIT_BATCH, shuffle=True,
                    num_workers=2, pin_memory=True)

    x, y = next(iter(dl))
    print(f"\nbatch image {tuple(x.shape)} {x.dtype}   "
          f"range [{x.min():.2f}, {x.max():.2f}]")
    print(f"batch mask  {tuple(y.shape)} {y.dtype}   "
          f"values {sorted(torch.unique(y).tolist())}")
    assert y.max() < NUM_CLASSES, f"mask has class {y.max()}, NUM_CLASSES={NUM_CLASSES}"
    assert y.min() >= 0, "negative class index in mask"

    model = build_unet().to(DEVICE)
    total, trainable = describe_model(model)
    kind = "from scratch" if PRETRAINED_ENCODER is None \
        else f"{PRETRAINED_ENCODER} encoder, ImageNet-pretrained"
    print(f"model: {MODEL_TAG}  ({kind})")
    print(f"parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable\n")

    weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
    criterion = ComboLoss(class_weights=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        tot, dice_acc, n = 0.0, torch.zeros(NUM_CLASSES, device=DEVICE), 0
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot += loss.item()
            d = per_class_dice(logits.float(), y)
            dice_acc += torch.nan_to_num(d)
            n += 1

        if epoch % 10 == 0 or epoch == 1:
            d = (dice_acc / n).tolist()
            ds_str = "  ".join(f"{name} {v:.3f}" for name, v in zip(CLASS_NAMES, d))
            print(f"epoch {epoch:>4}  loss {tot/n:.4f}   {ds_str}")

    final = (dice_acc / n).tolist()
    print("\n" + "=" * 55)
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Peak GPU memory: {peak:.1f} GB of {total:.1f} GB "
              f"at {IMG_SIZE}px, batch {OVERFIT_BATCH}")
        print(f"Rough headroom: batch ~{int(OVERFIT_BATCH * total * 0.85 / peak)} "
              f"should fit for Phase 3.\n")
    if min(final) > 0.95:
        print("PASS — the model memorised the subset. Pipeline is sound.")
        print("Move on to Phase 3: full training with augmentation.")
    else:
        print("FAIL — could not memorise 8 images. Do NOT proceed.")
        print("Check, in this order:")
        print("  1. the overlay PNG: are masks aligned with images?")
        print("  2. mask values: must be 0..NUM_CLASSES-1 int64")
        print("  3. is a class simply absent from these 8 images?")
        print("  4. try raising OVERFIT_EPOCHS or OVERFIT_LR")


if __name__ == "__main__":
    overfit_test()
