"""Dataset and preprocessing for chest X-ray lung segmentation.

Expected layout under ``--data-root`` (default ``data/``)::

    data/
      Montgomery/{img,mask}/MCUCXR_0001_0.png
      Shenzhen/{img,mask}/CHNCXR_0001_0.png

An image and its mask share a filename. The ``ann/`` folders shipped with the
Supervisely release of these datasets are not used.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SEED = 0

# Geometry. The network's valid convolutions shrink the input by 184 px, so the
# image is mirror-padded by PAD on every side and the label is left uncropped:
# the prediction then covers the whole 500x500 frame. Source images are 512x512,
# so OUT_SIZE=500 is the largest valid size that does not upsample.
OUT_SIZE = 500
PAD = 92
IN_SIZE = OUT_SIZE + 2 * PAD  # 684
BASE = 16

DATASETS = ("Montgomery", "Shenzhen")


def pairs_from(root, name):
    """img/X.png <-> mask/X.png, same filename. Returns (img_path, mask_path, source)."""
    d = Path(root) / name
    img_dir, mask_dir = d / "img", d / "mask"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"expected {img_dir} and {mask_dir}\n"
            f"Run `python scripts/download_data.py` or see the README."
        )
    out, missing = [], 0
    for im in sorted(img_dir.glob("*.png")):
        m = mask_dir / im.name
        if m.exists():
            out.append((str(im), str(m), name.lower()))
        else:
            missing += 1
    if missing:
        print(f"  {name}: {missing} images had no mask (skipped)")
    return out


def load_cache(items, size):
    """Decode every pair once into uint8. ~352 MB for 704 pairs at 500x500."""
    imgs = np.zeros((len(items), size, size), np.uint8)
    msks = np.zeros((len(items), size, size), np.uint8)
    for i, (ip, mp, _) in enumerate(items):
        imgs[i] = np.asarray(
            Image.open(ip).convert("L").resize((size, size), Image.BILINEAR)
        )
        # Antialiased mask edges: resize bilinear, THEN threshold. Thresholding
        # first would alias exactly the boundary the Dice score is decided on.
        msks[i] = (
            np.asarray(Image.open(mp).convert("L").resize((size, size), Image.BILINEAR))
            > 127
        )
    return imgs, msks


class CXRSeg(Dataset):
    """Chest X-rays + lung masks, fully decoded into RAM up front."""

    def __init__(self, items, out_size=OUT_SIZE, pad=PAD, augment=False):
        self.items, self.pad, self.augment = items, pad, augment
        self.srcs = [s for _, _, s in items]
        self.imgs, self.msks = load_cache(items, out_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.imgs[i].astype(np.float32) / 255.0
        m = self.msks[i].astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                x, m = x[:, ::-1].copy(), m[:, ::-1].copy()
            x = np.clip(
                x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1
            )

        # Standardise BEFORE padding -- padding first would pull mirrored border
        # pixels into the mean and std.
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = np.pad(x, self.pad, mode="reflect")  # OUT -> IN, mirrored context
        return torch.from_numpy(x)[None], torch.from_numpy(m)[None], self.srcs[i]


def preprocess_image(path, out_size=OUT_SIZE, pad=PAD):
    """Single-image inference path: the same steps as CXRSeg without augmentation.

    Returns a ``(1, IN, IN)`` float tensor ready for ``Unet.forward``.
    """
    x = np.asarray(
        Image.open(path).convert("L").resize((out_size, out_size), Image.BILINEAR),
        np.float32,
    ) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = np.pad(x, pad, mode="reflect")
    return torch.from_numpy(x)[None]


def split(items, tr=0.7):
    """70/30, applied per dataset so both appear on each side."""
    items = items[:]
    random.Random(SEED).shuffle(items)
    cut = int(len(items) * tr)
    return items[:cut], items[cut:]


def build_splits(data_root, verbose=True):
    """Returns (train_items, val_items), each a list of (img, mask, source)."""
    per_ds = {name: pairs_from(data_root, name) for name in DATASETS}
    for name, items in per_ds.items():
        if not items:
            raise RuntimeError(f"no image/mask pairs found for {name} under {data_root}")

    train_items, val_items, parts = [], [], []
    for name, items in per_ds.items():
        tr, va = split(items)
        train_items += tr
        val_items += va
        parts.append(f"{name.lower()} {len(tr)}/{len(va)}")

    if verbose:
        print(
            f"train {len(train_items)}  val {len(val_items)}   ({', '.join(parts)})"
        )
    return train_items, val_items
