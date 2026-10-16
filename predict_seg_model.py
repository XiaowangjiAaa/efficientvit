# EfficientViT: inference for binary crack segmentation.
# Mirrors the reference project's `predict` flow: read one image or a folder,
# run the model, and write original-resolution 0/255 PNG masks.
#
# Example:
#   python predict_seg_model.py --model b0 --num_classes 2 \
#       --checkpoint work_dirs/seg/crack/<run>/best_mIoU_iter_20000.pth \
#       --input dataset/Image --output predictions \
#       --eval_mode resize --val_scale 2048 512

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from efficientvit.models.utils import load_state_dict_from_file
from efficientvit.seg_model_zoo import create_seg_model
from prepare_seg_dataset import EXTENSIONS
from train_seg_model import MEAN, STD, _rescale_size, forward_divisible, sliding_inference
from train_seg_model import load_experiment_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientViT crack segmentation inference")
    parser.add_argument("--config", type=str, default=None, help="JSON experiment config; CLI args override it")
    parser.add_argument("--model", type=str, default="b0", choices=["b0", "b1", "b2", "b3", "l1", "l2"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="image file or directory")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--eval_mode", type=str, default="resize", choices=["resize", "full", "slide"])
    parser.add_argument("--val_scale", type=int, nargs=2, default=[2048, 512], metavar=("W", "H"))
    parser.add_argument("--eval_scales", type=float, nargs="+", default=None,
                        help="multi-scale TTA factors for resize mode (e.g. 0.75 1.0 1.25)")
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--window_overlap", type=float, default=0.25)
    parser.add_argument("--gpus", type=str, default="0")
    pre, _ = parser.parse_known_args()
    if pre.config:
        parser.set_defaults(**load_experiment_config(pre.config))
    args = parser.parse_args()
    if not args.checkpoint or not args.input:
        parser.error("--checkpoint and --input are required (CLI or JSON config)")
    return args


def _to_tensor(image_rgb, device):
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float().div(255.0)
    return TF.normalize(tensor, MEAN, STD).unsqueeze(0).to(device)


@torch.no_grad()
def predict_one(model, image_rgb, device, args):
    h, w = image_rgb.shape[:2]
    if args.eval_mode == "slide":
        logits = sliding_inference(model, _to_tensor(image_rgb, device), args.num_classes,
                                   (args.window_size, args.window_size), args.window_overlap)
    elif args.eval_mode == "full":
        logits = forward_divisible(model, _to_tensor(image_rgb, device))
    else:  # resize, possibly multi-scale TTA
        logits = None
        for factor in (args.eval_scales or [1.0]):
            scale = (args.val_scale[0] * factor, args.val_scale[1] * factor)
            new_w, new_h = _rescale_size((w, h), scale)
            resized = cv2.resize(image_rgb, (max(new_w, 1), max(new_h, 1)), interpolation=cv2.INTER_LINEAR)
            scaled = F.interpolate(forward_divisible(model, _to_tensor(resized, device)), size=(h, w),
                                   mode="bilinear", align_corners=False)
            logits = scaled if logits is None else logits + scaled
        logits = logits / len(args.eval_scales or [1.0])
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(resolve_path(args.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise SystemExit("Checkpoint does not exist: {}".format(checkpoint))
    source = Path(resolve_path(args.input)).resolve()
    if not source.exists():
        raise SystemExit("Input does not exist: {}".format(source))
    images = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise SystemExit("No input images found")

    output = Path(args.output).resolve()
    if source.is_dir() and (source == output or source in output.parents):
        raise SystemExit("Prediction output must be outside the input directory")

    model = create_seg_model(args.model, "cityscapes", pretrained=False, n_classes=args.num_classes)
    model.load_state_dict(load_state_dict_from_file(str(checkpoint)), strict=False)
    model = model.to(device).eval()

    targets = set()
    for image in images:
        relative = Path(image.name) if source.is_file() else image.relative_to(source)
        target = output / relative.with_suffix(".png")
        if target in targets or target.exists():
            raise SystemExit("Duplicate or existing prediction output: {}".format(target))
        targets.add(target)

    for image in images:
        relative = Path(image.name) if source.is_file() else image.relative_to(source)
        target = output / relative.with_suffix(".png")
        image_rgb = np.array(Image.open(image).convert("RGB"))
        label = predict_one(model, image_rgb, device, args)
        if not set(np.unique(label).tolist()) <= {0, 1}:
            raise SystemExit("Prediction is not binary: {}".format(image))
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((label * 255).astype(np.uint8)).save(target)

    print("Saved {} masks to {}".format(len(images), output))


if __name__ == "__main__":
    main()
