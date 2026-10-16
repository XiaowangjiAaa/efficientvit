# Dataset preparation for binary crack segmentation.
# Faithful port of the pairing / validation / split logic from the reference
# project (Crack-seg-Yingjie-New/simple_seg/data.py, MMSeg backend).
#
# Layout expected under --root:
#   <root>/
#   ├── Image/   sample_001.jpg, sub/dir/...      (images)
#   ├── Mask/    sample_001.png, sub/dir/...      (masks, paired by relative path w/o extension)
#   └── splits/  train.json, val.json, test.json, summary.json   (generated)
#
# Rules copied from the reference:
#   * pair images and masks by relative path minus extension (sub-directories supported);
#   * masks must be grayscale and binary: values <= {0, 1, 255};
#   * white (255) is FOREGROUND, never an ignore label in source masks;
#   * duplicate stems, missing pairs and image/mask size mismatch are hard errors;
#   * deterministic 7:2:1 split with a fixed seed; counts floored (min 1 each).
#
# Example:
#   python prepare_seg_dataset.py --root dataset

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    """Resolve a possibly-relative path against the project root (so `dataset` works anywhere)."""
    path = Path(path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def read_binary_mask(path):
    """Load a binary crack mask as {0, 1}; 255 -> 1 (foreground), 0 -> background."""
    with Image.open(path) as image:
        if image.mode in ("RGB", "RGBA"):
            rgb = np.asarray(image.convert("RGB"))
            if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
                raise ValueError("Mask must be grayscale, not colored: {}".format(path))
        high_depth = ("I", "I;16", "I;16B", "I;16L", "F")
        mask = np.asarray(image) if image.mode in high_depth else np.asarray(image.convert("L"))
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1, 255}:
        raise ValueError("Mask has non-binary values {}: {}".format(sorted(values), path))
    return (mask > 0).astype(np.uint8)


def index_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("Missing directory: {}".format(folder))
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(folder).with_suffix("").as_posix()
            if key in files:
                raise ValueError("Duplicate sample stem: {}".format(key))
            files[key] = path
    if not files:
        raise ValueError("No images found: {}".format(folder))
    return files


def prepare(root="dataset", images="Image", masks="Mask", output="splits", seed=42,
            val_ratio=0.2, test_ratio=0.1, force=False):
    root = Path(root).resolve()
    destination = root / output
    if destination.exists() and not force:
        raise ValueError("Split output already exists; use --force to replace it: {}".format(destination))
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val/test ratios must be positive and sum to less than 1")
    image_files, mask_files = index_files(root / images), index_files(root / masks)
    if image_files.keys() != mask_files.keys():
        missing = sorted(image_files.keys() - mask_files.keys())
        extra = sorted(mask_files.keys() - image_files.keys())
        raise ValueError("Unpaired files. Missing masks: {}; masks without images: {}".format(missing[:10], extra[:10]))
    if len(image_files) < 3:
        raise ValueError("At least three samples are required for train/val/test")
    records = []
    for i, key in enumerate(sorted(image_files)):
        image, mask = image_files[key], mask_files[key]
        with Image.open(image) as source:
            source.load()
            size = source.size
        label = read_binary_mask(mask)
        if size != (label.shape[1], label.shape[0]):
            raise ValueError("Image/mask size mismatch: {}".format(key))
        records.append(dict(id=key, image=image.relative_to(root).as_posix(),
                            mask=mask.relative_to(root).as_posix()))
        if (i + 1) % 500 == 0:
            print("Checked {} pairs".format(i + 1), flush=True)
    random.Random(seed).shuffle(records)
    nv, nt = max(1, int(len(records) * val_ratio)), max(1, int(len(records) * test_ratio))
    if nv + nt >= len(records):
        raise ValueError("Ratios leave no training samples")
    groups = dict(val=records[:nv], test=records[nv:nv + nt], train=records[nv + nt:])
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(root=str(root), seed=seed, val_ratio=val_ratio, test_ratio=test_ratio,
                   counts={k: len(v) for k, v in groups.items()},
                   pairing_sha256=hashlib.sha256(
                       json.dumps(sorted(records, key=lambda x: x["id"]), sort_keys=True).encode()).hexdigest())
    for name, samples in groups.items():
        (destination / (name + ".json")).write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf8")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare crack segmentation split files (7:2:1 by default)")
    parser.add_argument("--root", type=str, required=True, help="dataset root containing Image/ and Mask/")
    parser.add_argument("--images", type=str, default="Image")
    parser.add_argument("--masks", type=str, default="Mask")
    parser.add_argument("--output", type=str, default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split directory")
    args = parser.parse_args()
    summary = prepare(resolve_path(args.root), args.images, args.masks, args.output, args.seed,
                      args.val_ratio, args.test_ratio, args.force)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
