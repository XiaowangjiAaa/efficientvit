# EfficientViT: training script for binary crack segmentation (background vs crack).
#
# The dataset preparation, augmentation, validation and metrics follow the
# reference project `Crack-seg-Yingjie-New` (simple_seg, MMSeg backend):
#   * pair Image/ and Mask/ by relative path, strict binary-mask validation;
#   * native-style augmentation: RandomResize -> RandomCrop(cat_max_ratio)
#     -> RandomFlip -> PhotoMetricDistortion -> optional RandomRotate;
#   * iteration-based schedule with val_interval / checkpoint_interval /
#     save_on_val / early stopping (all counted in training iterations);
#   * MMSeg-style IoUMetric: aAcc, mAcc, mIoU, mDice, mPrecision, mRecall,
#     mFscore (F1), with per-class (background / foreground) results;
#   * best checkpoint by strict mIoU, plus a held-out test evaluation.
#
# Expected data layout (either one):
#   A) splits mode  : <root>/Image, <root>/Mask, <root>/splits/{train,val,test}.json
#                     (generate with `python prepare_seg_dataset.py --root <root>`)
#   B) folder mode  : <root>/images/{training,validation}, <root>/annotations/{training,validation}
#
# Example (relative data_root resolves to the project root):
#   python train_seg_model.py --data_root dataset \
#       --model b0 --num_classes 2 --crop_size 512 --batch_size 16 --max_iters 30000 --amp

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from efficientvit.models.utils import load_state_dict_from_file
from efficientvit.seg_model_zoo import create_seg_model
from prepare_seg_dataset import EXTENSIONS, read_binary_mask
from seg_logging import SegLogger

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ("background", "foreground")
IMG_IGNORE = 255  # ignore label used only by training padding / rotation

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    """Resolve a possibly-relative path against the project root (so `dataset` works anywhere)."""
    path = Path(path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def load_experiment_config(path):
    """Flatten a JSON experiment config into argparse dest names."""
    with open(path, encoding="utf8") as f:
        cfg = json.load(f)
    flat = {}
    for key in ("data_root", "split_dir", "num_classes", "crop_size", "workers", "max_iters",
                "batch_size", "lr", "weight_decay", "ce_weight", "dice_weight", "grad_clip",
                "min_lr", "val_interval", "checkpoint_interval", "patience_iters", "min_delta",
                "seed", "gpus", "work_dir", "backbone_init", "resume", "model", "max_val_samples",
                "max_test_samples"):
        if key in cfg:
            flat[key] = cfg[key]
    for key in ("amp", "eval_only", "test", "save_on_val"):
        if key in cfg:
            flat[key] = cfg[key]
    aug = cfg.get("augmentation", {})
    for src, dest in (("mode", "aug_mode"), ("resize_scale", "resize_scale"), ("ratio_range", "ratio_range"),
                      ("cat_max_ratio", "cat_max_ratio"), ("horizontal_flip_prob", "horizontal_flip_prob"),
                      ("vertical_flip_prob", "vertical_flip_prob"), ("photometric_distortion", "photometric_distortion"),
                      ("rotate_prob", "rotate_prob"), ("rotate_degree", "rotate_degree")):
        if src in aug:
            flat[dest] = aug[src]
    sched = cfg.get("scheduler", {})
    if "type" in sched:
        flat["scheduler"] = sched["type"]
    for key in ("power", "warmup_iters", "warmup_ratio", "min_lr"):
        if key in sched:
            flat[key] = sched[key]
    if "early_stopping" in cfg:
        early = cfg["early_stopping"]
        if early is None:
            flat["patience_iters"] = 0
        elif isinstance(early, dict):
            if "patience_iters" in early:
                flat["patience_iters"] = early["patience_iters"]
            if "min_delta" in early:
                flat["min_delta"] = early["min_delta"]
    ev = cfg.get("eval", {})
    for src, dest in (("mode", "eval_mode"), ("val_scale", "val_scale"), ("window_size", "window_size"),
                      ("window_overlap", "window_overlap"), ("scales", "eval_scales")):
        if src in ev:
            flat[dest] = ev[src]
    logging_cfg = cfg.get("logging", {})
    if "tensorboard" in logging_cfg:
        flat["tensorboard"] = logging_cfg["tensorboard"]
    if "interval" in logging_cfg:
        flat["log_interval"] = logging_cfg["interval"]
    if "window_size" in logging_cfg:
        flat["log_window"] = logging_cfg["window_size"]
    return flat


# ---------------------------------------------------------------------------
# Sample listing (splits JSON or folder layout)
# ---------------------------------------------------------------------------
def list_split(data_root, split, split_dir="splits", images="Image", masks="Mask"):
    root = Path(data_root)
    manifest = root / split_dir / (split + ".json")
    if manifest.is_file():
        records = json.loads(manifest.read_text(encoding="utf8"))
        if not records:
            raise ValueError("Empty split: {}".format(manifest))
        return [(str(root / r["image"]), str(root / r["mask"])) for r in records]

    # folder layout fallback
    sub = {"train": "training", "val": "validation", "test": "validation"}[split]
    img_dir, ann_dir = root / "images" / sub, root / "annotations" / sub
    if not img_dir.is_dir():
        raise FileNotFoundError("No split {} and no folder layout under {}".format(manifest, root))
    pairs = []
    for dirpath, _, fnames in os.walk(img_dir):
        for fname in sorted(fnames):
            if not fname.lower().endswith(tuple(EXTENSIONS)):
                continue
            img_path = Path(dirpath) / fname
            rel_stem = img_path.relative_to(img_dir).with_suffix("")
            for ext in (".png", ".PNG", ".bmp", img_path.suffix):
                cand = ann_dir / (str(rel_stem) + ext)
                if cand.is_file():
                    pairs.append((str(img_path), str(cand)))
                    break
            else:
                raise FileNotFoundError("No annotation for {}".format(img_path))
    if not pairs:
        raise RuntimeError("No samples found for split {}".format(split))
    return pairs


# ---------------------------------------------------------------------------
# Native-style augmentation (numpy image HWC RGB + mask HW uint8)
# ---------------------------------------------------------------------------
def _rescale_size(size, scale, return_scale=False):
    w, h = size
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    max_long, max_short = max(scale), min(scale)
    factor = min(max_long / max(h, w), max_short / min(h, w))
    new_size = (int(round(w * factor)), int(round(h * factor)))
    return (new_size, factor) if return_scale else new_size


def random_resize(image, mask, scale, ratio_range):
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    target = (scale[0] * ratio, scale[1] * ratio)
    (new_w, new_h), _ = _rescale_size((image.shape[1], image.shape[0]), target, True)
    new_w, new_h = max(new_w, 1), max(new_h, 1)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def random_crop(image, mask, crop_size, cat_max_ratio, max_attempts=10):
    ch, cw = crop_size
    h, w = image.shape[:2]
    pad_h, pad_w = max(ch - h, 0), max(cw - w, 0)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IMG_IGNORE)
    h, w = image.shape[:2]
    top, left = 0, 0
    for _ in range(max_attempts):
        top, left = random.randint(0, h - ch), random.randint(0, w - cw)
        patch = mask[top:top + ch, left:left + cw]
        values, counts = np.unique(patch, return_counts=True)
        keep = values != IMG_IGNORE
        if not keep.any() or counts[keep].sum() == 0:
            break
        if counts[keep].max() / counts[keep].sum() <= cat_max_ratio:
            break
    return image[top:top + ch, left:left + cw], mask[top:top + ch, left:left + cw]


def random_flip(image, mask, h_prob, v_prob):
    if random.random() < h_prob:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if random.random() < v_prob:
        image, mask = image[::-1, :], mask[::-1, :]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def photo_metric_distortion(image, brightness_delta=32, contrast_range=(0.5, 1.5),
                            saturation_range=(0.5, 1.5), hue_delta=18):
    img = image.astype(np.float32)
    if random.random() < 0.5:
        img += random.uniform(-brightness_delta, brightness_delta)
    if random.random() < 0.5:
        img *= random.uniform(*contrast_range)
    img = np.clip(img, 0, 255)
    if random.random() < 0.5:
        gray = np.dot(img, [0.299, 0.587, 0.114])[..., None]
        img = np.clip(gray + (img - gray) * random.uniform(*saturation_range), 0, 255)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-hue_delta, hue_delta)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def random_rotate(image, mask, prob, degree):
    if random.random() >= prob:
        return image, mask
    angle = random.uniform(-degree, degree)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=IMG_IGNORE)
    return image, mask


def apply_train_aug(image, mask, aug):
    if aug["mode"] == "random_crop":
        image, mask = random_resize(image, mask, aug["resize_scale"], aug["ratio_range"])
        image, mask = random_crop(image, mask, aug["crop_size"], aug["cat_max_ratio"])
    elif aug["mode"] == "resize":
        image = cv2.resize(image, (aug["crop_size"][1], aug["crop_size"][0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (aug["crop_size"][1], aug["crop_size"][0]), interpolation=cv2.INTER_NEAREST)
    image, mask = random_flip(image, mask, aug["horizontal_flip_prob"], aug["vertical_flip_prob"])
    if aug["photometric_distortion"]:
        image = photo_metric_distortion(image)
    if aug["rotate_prob"] > 0:
        image, mask = random_rotate(image, mask, aug["rotate_prob"], aug["rotate_degree"])
    return image, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CrackSegDataset(Dataset):
    def __init__(self, pairs, crop_size, train, aug, ignore_index=IMG_IGNORE):
        self.pairs = pairs
        self.crop_size = crop_size
        self.train = train
        self.aug = aug
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError("Image/mask size mismatch: {}".format(image_path))
        if self.train:
            image, mask = apply_train_aug(image, mask, self.aug)
        else:
            h, w = self.crop_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div(255.0)
        tensor = TF.normalize(tensor, MEAN, STD)
        return tensor, torch.from_numpy(mask.astype(np.int64))


# ---------------------------------------------------------------------------
# Validation / test inference (metrics always on the ORIGINAL mask grid)
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """Returns (native_image, original_mask, original_size).

    The image is always kept at native resolution; any resizing for a chosen
    eval_mode is done in `evaluate` (so multi-scale TTA can reuse the native image).
    """

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError("Image/mask size mismatch: {}".format(image_path))
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div(255.0)
        tensor = TF.normalize(tensor, MEAN, STD)
        return tensor, torch.from_numpy(mask.astype(np.int64)), torch.tensor(mask.shape, dtype=torch.long)


def collate_eval(batch):
    """Stack only the images; keep masks/sizes as lists (original sizes may differ)."""
    images = torch.stack([item[0] for item in batch])
    masks = [item[1] for item in batch]
    sizes = [item[2] for item in batch]
    return images, masks, sizes


def forward_divisible(model, images):
    """EfficientViT needs H,W divisible by 32 for the head fusion; pad then crop the 1/8 output."""
    h, w = images.shape[-2:]
    pad_h, pad_w = (-h) % 32, (-w) % 32
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), mode="replicate")
    logits = model(images)
    return logits[:, :, :math.ceil(h / 8), :math.ceil(w / 8)]


def sliding_inference(model, image, num_classes, window, overlap):
    """Sliding-window inference at native resolution; returns logits on the 1/8 grid."""
    _, _, h, w = image.shape
    pad_h, pad_w = (-h) % 32, (-w) % 32
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    h2, w2 = image.shape[-2:]
    win_h = min(int(window[0]), h2)
    win_w = min(int(window[1]), w2)
    if win_h < h2:
        win_h -= win_h % 32
    if win_w < w2:
        win_w -= win_w % 32
    stride_h = max(32, int(round(win_h * (1 - overlap) / 32)) * 32)
    stride_w = max(32, int(round(win_w * (1 - overlap) / 32)) * 32)

    def starts(limit, win, stride):
        if win >= limit:
            return [0]
        positions = list(range(0, limit - win + 1, stride))
        if positions[-1] != limit - win:
            positions.append(limit - win)
        return positions

    ys, xs = starts(h2, win_h, stride_h), starts(w2, win_w, stride_w)
    grid_h, grid_w = h2 // 8, w2 // 8
    out = torch.zeros((1, num_classes, grid_h, grid_w), device=image.device, dtype=torch.float32)
    count = torch.zeros((1, 1, grid_h, grid_w), device=image.device, dtype=torch.float32)
    for y in ys:
        for x in xs:
            logits = model(image[:, :, y:y + win_h, x:x + win_w])
            gh, gw = logits.shape[-2:]
            out[:, :, y // 8:y // 8 + gh, x // 8:x // 8 + gw] += logits
            count[:, :, y // 8:y // 8 + gh, x // 8:x // 8 + gw] += 1
    return out / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Metrics (MMSeg IoUMetric style) and losses
# ---------------------------------------------------------------------------
class IoUMeter:
    def __init__(self, num_classes, ignore_index=IMG_IGNORE):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf = np.zeros((num_classes, num_classes), dtype=np.float64)  # rows=gt, cols=pred

    def reset(self):
        self.conf.fill(0.0)

    @torch.no_grad()
    def update(self, pred, target):
        pred = pred.detach().cpu().numpy().reshape(-1).astype(np.int64)
        target = target.detach().cpu().numpy().reshape(-1).astype(np.int64)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            pred, target = pred[keep], target[keep]
        if target.size == 0:
            return
        bincount = np.bincount(target * self.num_classes + pred, minlength=self.num_classes ** 2)
        self.conf += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self):
        conf = self.conf
        total = conf.sum()
        nan = float("nan")
        acc = np.full(self.num_classes, nan)
        iou = np.full(self.num_classes, nan)
        precision = np.full(self.num_classes, nan)
        recall = np.full(self.num_classes, nan)
        dice = np.full(self.num_classes, nan)
        fscore = np.full(self.num_classes, nan)
        for c in range(self.num_classes):
            tp = conf[c, c]
            fp = conf[:, c].sum() - tp  # predicted c, not c
            fn = conf[c, :].sum() - tp  # gt c, not predicted c
            tn = total - tp - fp - fn
            if tp + tn + fp + fn > 0:
                acc[c] = (tp + tn) / (tp + tn + fp + fn)
            if tp + fp + fn > 0:
                iou[c] = tp / (tp + fp + fn)
            if tp + fp > 0:
                precision[c] = tp / (tp + fp)
            if tp + fn > 0:
                recall[c] = tp / (tp + fn)
            if 2 * tp + fp + fn > 0:
                dice[c] = 2 * tp / (2 * tp + fp + fn)
            if precision[c] + recall[c] > 0:
                fscore[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        metrics = {
            "aAcc": float(np.diag(conf).sum() / total * 100) if total > 0 else 0.0,
            "mAcc": float(np.nanmean(acc) * 100) if total > 0 else 0.0,
            "mIoU": float(np.nanmean(iou) * 100) if total > 0 else 0.0,
            "mDice": float(np.nanmean(dice) * 100) if total > 0 else 0.0,
            "mPrecision": float(np.nanmean(precision) * 100) if total > 0 else 0.0,
            "mRecall": float(np.nanmean(recall) * 100) if total > 0 else 0.0,
            "mFscore": float(np.nanmean(fscore) * 100) if total > 0 else 0.0,
        }
        for c, name in enumerate(CLASS_NAMES[: self.num_classes]):
            metrics["IoU/{}".format(name)] = float(iou[c] * 100) if not math.isnan(iou[c]) else 0.0
            metrics["Dice/{}".format(name)] = float(dice[c] * 100) if not math.isnan(dice[c]) else 0.0
            metrics["Precision/{}".format(name)] = float(precision[c] * 100) if not math.isnan(precision[c]) else 0.0
            metrics["Recall/{}".format(name)] = float(recall[c] * 100) if not math.isnan(recall[c]) else 0.0
            metrics["F1/{}".format(name)] = float(fscore[c] * 100) if not math.isnan(fscore[c]) else 0.0
        return metrics


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=IMG_IGNORE, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1).float()
        else:
            valid = torch.ones_like(target).unsqueeze(1).float()
        target = target.clone()
        target[valid.squeeze(1) == 0] = 0
        onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        probs, onehot = probs * valid, onehot * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        return 1.0 - ((2 * inter + self.eps) / (union + self.eps)).mean()


def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def load_backbone_init(model, ckpt_path):
    state_dict = load_state_dict_from_file(ckpt_path)
    backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if not backbone_sd:
        backbone_sd = state_dict
    print("Backbone init from {}: {}".format(ckpt_path, model.backbone.load_state_dict(backbone_sd, strict=False)))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def make_loss(args, criterion, dice):
    def loss_fn(logits, target):
        return args.ce_weight * criterion(logits, target) + args.dice_weight * dice(logits, target)
    return loss_fn


def train_step(model, images, labels, criterion, dice, optimizer, scaler, args):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        logits = forward_divisible(model, images)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        ce_loss = criterion(logits, labels)
        dice_loss = dice(logits, labels)
        loss = args.ce_weight * ce_loss + args.dice_weight * dice_loss
    scaler.scale(loss).backward()
    if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return {"train/loss": loss.item(), "train/ce": float(ce_loss.detach()), "train/dice": float(dice_loss.detach())}


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index, loss_fn=None, desc="Val",
             mode="resize", val_scale=(2048, 512), window=(512, 512), overlap=0.25, scales=None,
             logger=None, log_step=0):
    """Evaluate on the ORIGINAL mask grid regardless of the inference resolution.

    resize: whole image rescaled with keep-ratio to val_scale (optionally averaged
            over `scales` factors -> multi-scale TTA on logits);
    full:   native-resolution whole image;
    slide:  native-resolution sliding window.
    """
    model.eval()
    meter = IoUMeter(num_classes, ignore_index)
    total_loss, n = 0.0, 0
    logged = 0
    max_images = getattr(logger, "tb_images", 0) if logger is not None else 0
    for images, masks, sizes in tqdm(loader, desc=desc, ncols=100):
        images = images.to(device, non_blocking=True)
        for b in range(images.shape[0]):
            image = images[b:b + 1]
            target = masks[b].to(device).unsqueeze(0)
            oh, ow = int(sizes[b][0]), int(sizes[b][1])
            if mode == "slide":
                logits = sliding_inference(model, image, num_classes, window, overlap)
            elif mode == "full":
                logits = forward_divisible(model, image)
            else:  # resize, possibly multi-scale
                logits = None
                for factor in (scales or [1.0]):
                    scale = (val_scale[0] * factor, val_scale[1] * factor)
                    new_w, new_h = _rescale_size((image.shape[3], image.shape[2]), scale)
                    resized = F.interpolate(image, size=(max(new_h, 1), max(new_w, 1)),
                                            mode="bilinear", align_corners=False)
                    scaled_logits = F.interpolate(forward_divisible(model, resized), size=(oh, ow),
                                                  mode="bilinear", align_corners=False)
                    logits = scaled_logits if logits is None else logits + scaled_logits
                logits = logits / len(scales or [1.0])
            if logits.shape[-2:] != (oh, ow):
                logits = F.interpolate(logits, size=(oh, ow), mode="bilinear", align_corners=False)
            if loss_fn is not None:
                total_loss += loss_fn(logits, target).item()
            pred = logits.argmax(dim=1)
            meter.update(pred, target)
            if logger is not None and logged < max_images:
                logger.log_seg_image(log_step, "{}_{}".format(desc.lower(), logged), image[0], pred[0], target[0])
                logged += 1
            n += 1
    metrics = meter.compute()
    if loss_fn is not None:
        metrics["loss"] = total_loss / max(n, 1)
    model.train()
    return metrics


def format_metrics(metrics):
    keys = ["mIoU", "mDice", "mPrecision", "mRecall", "mFscore", "aAcc",
            "IoU/foreground", "Dice/foreground", "F1/foreground"]
    return " ".join("{}={:.2f}".format(k, metrics[k]) for k in keys if k in metrics)


def main():
    parser = argparse.ArgumentParser(description="EfficientViT crack segmentation training (MMSeg-style validation)")
    parser.add_argument("--config", type=str, default=None, help="JSON experiment config; CLI args override it")
    # data
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default="splits")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=-1, help="-1 = use all validation samples")
    parser.add_argument("--max_test_samples", type=int, default=-1, help="-1 = use all test samples")
    # model
    parser.add_argument("--model", type=str, default="b0", choices=["b0", "b1", "b2", "b3", "l1", "l2"])
    parser.add_argument("--backbone_init", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    # optimization (iteration based, mirrors the reference config)
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0, help="0 to use plain CE (reference default)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="<=0 disables clipping")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", type=str, default="poly", choices=["poly", "cosine"])
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=1500)
    parser.add_argument("--warmup_ratio", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    # validation / checkpoint / early stopping
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--save_on_val", action="store_true", default=True)
    parser.add_argument("--no_save_on_val", dest="save_on_val", action="store_false")
    parser.add_argument("--patience_iters", type=int, default=5000, help="<=0 disables early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["mIoU", "mDice", "mFscore", "Precision", "Recall", "F1-score"])
    # augmentation (defaults follow the reference templates)
    parser.add_argument("--aug_mode", type=str, default="random_crop", choices=["random_crop", "resize"])
    parser.add_argument("--resize_scale", type=int, nargs=2, default=[2048, 512], metavar=("W", "H"),
                        help="train RandomResize target, official ADE20K default [2048, 512]")
    parser.add_argument("--ratio_range", type=float, nargs=2, default=[0.5, 2.0])
    parser.add_argument("--cat_max_ratio", type=float, default=0.75, help="official ADE20K RandomCrop default")
    parser.add_argument("--horizontal_flip_prob", type=float, default=0.5)
    parser.add_argument("--vertical_flip_prob", type=float, default=0.0)
    parser.add_argument("--photometric_distortion", action="store_true", default=True)
    parser.add_argument("--no_photometric_distortion", dest="photometric_distortion", action="store_false")
    parser.add_argument("--rotate_prob", type=float, default=0.0)
    parser.add_argument("--rotate_degree", type=float, default=10.0)
    # validation / test inference (metrics are always computed on the original mask grid)
    parser.add_argument("--eval_mode", type=str, default="resize", choices=["resize", "full", "slide"],
                        help="resize: keep-ratio resize to val_scale; full: native resolution; slide: sliding window")
    parser.add_argument("--val_scale", type=int, nargs=2, default=[2048, 512], metavar=("W", "H"),
                        help="official val Resize scale (width, height), keep-ratio")
    parser.add_argument("--eval_scales", type=float, nargs="+", default=None,
                        help="multi-scale TTA factors applied to val_scale in resize mode (e.g. 0.75 1.0 1.25)")
    parser.add_argument("--window_size", type=int, default=0, help="sliding-window size, 0 = crop_size")
    parser.add_argument("--window_overlap", type=float, default=0.25, help="sliding-window overlap ratio")
    parser.add_argument("--eval_only", action="store_true", help="skip training; evaluate --resume checkpoint on val/test")
    # misc
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--work_dir", type=str, default="work_dirs/seg/crack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="run held-out test evaluation after training")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--log_interval", type=int, default=20, help="write scalars to JSONL / console every N iters")
    parser.add_argument("--log_window", type=int, default=20, help="smoothing window for train/loss_smoothed")
    parser.add_argument("--tb_images", type=int, default=4, help="validation prediction images logged to TensorBoard")
    parser.add_argument("--limit_iters", type=int, default=-1, help="smoke-test override for max_iters")

    pre, _ = parser.parse_known_args()
    if pre.config:
        parser.set_defaults(**load_experiment_config(pre.config))
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data_root is required (set data_root in the JSON config or pass --data_root)")
    if args.limit_iters > 0:
        args.max_iters = args.limit_iters

    args.data_root = resolve_path(args.data_root)
    if args.backbone_init:
        args.backbone_init = resolve_path(args.backbone_init)
    if args.resume:
        args.resume = resolve_path(args.resume)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    device = torch.device("cuda" if device_ids else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger = SegLogger(run_dir, tensorboard=args.tensorboard, log_interval=args.log_interval,
                       window_size=args.log_window, tb_images=args.tb_images)
    log = logger.text
    scalars_path = logger.scalars_path
    val_metrics_path = logger.val_path
    test_metrics_path = logger.test_path

    # data
    aug = dict(mode=args.aug_mode, crop_size=(args.crop_size, args.crop_size),
               resize_scale=tuple(args.resize_scale), ratio_range=tuple(args.ratio_range),
               cat_max_ratio=args.cat_max_ratio, horizontal_flip_prob=args.horizontal_flip_prob,
               vertical_flip_prob=args.vertical_flip_prob, photometric_distortion=args.photometric_distortion,
               rotate_prob=args.rotate_prob, rotate_degree=args.rotate_degree)
    train_pairs = list_split(args.data_root, "train", args.split_dir)
    val_pairs = list_split(args.data_root, "val", args.split_dir)
    if args.max_val_samples > 0:
        val_pairs = val_pairs[: args.max_val_samples]
    train_set = CrackSegDataset(train_pairs, aug["crop_size"], True, aug)
    val_set = EvalDataset(val_pairs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    eval_batch = 1  # keep-ratio resize / native sizes vary across samples (reference uses batch_size=1 for val/test)
    val_loader = DataLoader(val_set, batch_size=eval_batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False, collate_fn=collate_eval)
    window = (args.window_size or args.crop_size, args.window_size or args.crop_size)
    log("train={} val={} data_root={} eval_mode={}".format(len(train_set), len(val_set), args.data_root, args.eval_mode))

    # model
    model = create_seg_model(args.model, "cityscapes", pretrained=False, n_classes=args.num_classes)
    if args.backbone_init:
        load_backbone_init(model, args.backbone_init)
    if args.resume:
        model.load_state_dict(load_state_dict_from_file(args.resume), strict=False)
        log("Resumed from {}".format(args.resume))
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=IMG_IGNORE)
    dice = DiceLoss(ignore_index=IMG_IGNORE)
    loss_fn = make_loss(args, criterion, dice)
    optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)

    def lr_lambda(it):
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.warmup_ratio + (1.0 - args.warmup_ratio) * it / args.warmup_iters
        prog = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        prog = min(max(prog, 0.0), 1.0)
        if args.scheduler == "poly":
            factor = (1.0 - prog) ** args.power
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * prog))
        return max(args.min_lr / args.lr, factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    log("model={} classes={} device={} amp={} loss=CE*{} + Dice*{}".format(
        args.model, args.num_classes, device, args.amp, args.ce_weight, args.dice_weight))

    net = model.module if isinstance(model, nn.DataParallel) else model
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log("params: total={:,} trainable={:,}".format(total_params, trainable_params))
    logger.write_manifest({
        "config": vars(args),
        "model": args.model,
        "num_classes": args.num_classes,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "params_total": total_params,
        "params_trainable": trainable_params,
    })
    logger.text_block("run/config", json.dumps(vars(args), indent=2, ensure_ascii=False))
    logger.text_block("run/params", "total={:,}\ntrainable={:,}".format(total_params, trainable_params))

    def save_checkpoint(name, step, metrics):
        net = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"state_dict": net.state_dict(), "iter": step, "metrics": metrics},
                   os.path.join(run_dir, name))

    best_iter = 0

    def run_test():
        test_pairs = list_split(args.data_root, "test", args.split_dir)
        if args.max_test_samples > 0:
            test_pairs = test_pairs[: args.max_test_samples]
        test_set = EvalDataset(test_pairs)
        test_loader = DataLoader(test_set, batch_size=eval_batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate_eval)
        test_metrics = evaluate(model, test_loader, device, args.num_classes, IMG_IGNORE, loss_fn, "Test",
                                args.eval_mode, tuple(args.val_scale), window, args.window_overlap, args.eval_scales,
                                logger=logger, log_step=best_iter)
        logger.log_test(best_iter, test_metrics)
        return test_metrics

    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval_only requires --resume <checkpoint>")
        val_metrics = evaluate(model, val_loader, device, args.num_classes, IMG_IGNORE, loss_fn, "Val",
                               args.eval_mode, tuple(args.val_scale), window, args.window_overlap, args.eval_scales,
                               logger=logger, log_step=0)
        logger.log_val(0, val_metrics)
        if args.test:
            run_test()
        logger.close()
        return

    best_miou, best_iter, last_val_iter = -1.0, 0, 0
    data_iter = iter(train_loader)
    model.train()
    step = 0
    while step < args.max_iters:
        data_t0 = time.time()
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        data_time = time.time() - data_t0
        iter_t0 = time.time()
        loss_dict = train_step(model, images, labels, criterion, dice, optimizer, scaler, args)
        iter_time = time.time() - iter_t0
        scheduler.step()
        step += 1
        lr = optimizer.param_groups[0]["lr"]
        logger.log_train(step, {**loss_dict, "train/lr": lr, "train/epoch": step / max(len(train_loader), 1)},
                         iter_time, data_time)
        if step % args.log_interval == 0 or step == 1:
            logger.log_epoch_summary(step, {**loss_dict, "lr": lr, "iter_s": iter_time, "data_s": data_time})

        run_val = (step % args.val_interval == 0) or (step == args.max_iters)
        if run_val:
            metrics = evaluate(model, val_loader, device, args.num_classes, IMG_IGNORE, loss_fn, "Val",
                               args.eval_mode, tuple(args.val_scale), window, args.window_overlap, args.eval_scales,
                               logger=logger, log_step=step)
            logger.log_val(step, metrics)
            if args.save_on_val:
                save_checkpoint("iter_{}.pth".format(step), step, metrics)
            if metrics["mIoU"] > best_miou + args.min_delta:
                best_miou, best_iter = metrics["mIoU"], step
                save_checkpoint("best_mIoU_iter_{}.pth".format(step), step, metrics)
                logger.log_scalars(step, {"val/best_mIoU": best_miou}, force_file=True)
                log("  -> new best mIoU={:.2f} at iter {}".format(best_miou, step))
            if args.patience_iters > 0 and step - best_iter >= args.patience_iters:
                log("Early stopping at iter {}: mIoU not improved for {} iters (best {:.2f} at iter {})".format(
                    step, step - best_iter, best_miou, best_iter))
                break
        elif step % args.checkpoint_interval == 0:
            save_checkpoint("iter_{}.pth".format(step), step, {})

    log("Training done. best mIoU={:.2f} at iter {}".format(best_miou, best_iter))

    if args.test:
        run_test()

    logger.close()


if __name__ == "__main__":
    main()
