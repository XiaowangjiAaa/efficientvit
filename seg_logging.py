# Detailed training logger for the EfficientViT crack-segmentation scripts.
# Writes three parallel records per run:
#   * train.log              human-readable, timestamped, per-iteration + per-validation
#   * scalars.jsonl          machine-readable training scalars (JSON lines)
#   * TensorBoard event file all scalars + validation prediction images
# plus val_metrics.jsonl / test_metrics.jsonl / run_manifest.json.
import json
import os
import platform
import time
from collections import deque

import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Tags shown in the compact per-validation text table.
_TABLE_KEYS = ["mIoU", "mDice", "mPrecision", "mRecall", "mFscore", "aAcc"]


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _denormalize(image_chw):
    """(3,H,W) normalized float -> HWC uint8, clipped to [0, 255]."""
    image = image_chw.detach().cpu().float().numpy()
    image = image * STD[:, None, None] + MEAN[:, None, None]
    image = np.clip(image, 0.0, 1.0).transpose(1, 2, 0)
    return (image * 255.0).astype(np.uint8)


def _mask_to_rgb(mask_hw):
    mask = mask_hw.detach().cpu().numpy().astype(np.uint8)
    return np.repeat((mask * 255)[..., None], 3, axis=2)


class SegLogger:
    def __init__(self, run_dir, tensorboard=True, log_interval=20, window_size=20, tb_images=4):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_interval = max(1, int(log_interval))
        self.window_size = max(1, int(window_size))
        self.tb_images = int(tb_images)
        self.text_path = os.path.join(run_dir, "train.log")
        self.scalars_path = os.path.join(run_dir, "scalars.jsonl")
        self.val_path = os.path.join(run_dir, "val_metrics.jsonl")
        self.test_path = os.path.join(run_dir, "test_metrics.jsonl")
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(run_dir)
            except ImportError:
                self.text("TensorBoard not installed; scalar log kept in JSONL only.", level="WARN")
        self._loss_window = deque(maxlen=self.window_size)
        self.start_time = time.time()

    # ------------------------------------------------------------------ text
    def text(self, message, level="INFO", print_log=True):
        line = "[{}][{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), level, message)
        with open(self.text_path, "a", encoding="utf8") as fout:
            fout.write(line + "\n")
        if print_log:
            print(message, flush=True)

    def text_block(self, tag, text, step=0):
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    # --------------------------------------------------------------- scalars
    def log_scalars(self, step, scalars, force_file=False):
        clean = {key: _to_float(value) for key, value in scalars.items()}
        if self.writer is not None:
            for key, value in clean.items():
                self.writer.add_scalar(key, value, step)
        if force_file or step % self.log_interval == 0:
            with open(self.scalars_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps({"iter": step, "wall": time.time() - self.start_time, **clean}) + "\n")
        return clean

    def log_train(self, step, scalars, iter_time=None, data_time=None):
        if iter_time is not None:
            scalars["time/iter"] = iter_time
            scalars["time/data"] = data_time if data_time is not None else 0.0
        if "train/loss" in scalars:
            self._loss_window.append(_to_float(scalars["train/loss"]))
            scalars["train/loss_smoothed"] = sum(self._loss_window) / len(self._loss_window)
        self.log_scalars(step, scalars)

    def log_epoch_summary(self, step, scalars):
        parts = ["iter {}".format(step)]
        parts += ["{}={:.4g}".format(k, _to_float(v)) for k, v in scalars.items()]
        self.text("train " + " ".join(parts))

    # ------------------------------------------------------------------- val
    def log_val(self, step, metrics):
        self.log_scalars(step, {"val/" + key: value for key, value in metrics.items()}, force_file=True)
        with open(self.val_path, "a", encoding="utf8") as fout:
            fout.write(json.dumps({"iter": step, **metrics}) + "\n")
        self.text("iter {} val:".format(step))
        self.text(self._metric_table(metrics))

    def log_test(self, step, metrics):
        self.log_scalars(step, {"test/" + key: value for key, value in metrics.items()}, force_file=True)
        with open(self.test_path, "a", encoding="utf8") as fout:
            fout.write(json.dumps({"iter": step, **metrics}) + "\n")
        self.text("iter {} test:".format(step))
        self.text(self._metric_table(metrics))

    @staticmethod
    def _metric_table(metrics):
        lines = []
        header = "  {:<12}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
            "class", "IoU", "Dice", "Precision", "Recall", "F1", "Acc")
        lines.append(header)
        for name in ("background", "foreground"):
            if "IoU/{}".format(name) not in metrics:
                continue
            lines.append("  {:<12}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}".format(
                name,
                metrics.get("IoU/{}".format(name), float("nan")),
                metrics.get("Dice/{}".format(name), float("nan")),
                metrics.get("Precision/{}".format(name), float("nan")),
                metrics.get("Recall/{}".format(name), float("nan")),
                metrics.get("F1/{}".format(name), float("nan")),
            ))
        macro = "  {:<12}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}".format(
            "macro",
            metrics.get("mIoU", 0.0), metrics.get("mDice", 0.0), metrics.get("mPrecision", 0.0),
            metrics.get("mRecall", 0.0), metrics.get("mFscore", 0.0), metrics.get("aAcc", 0.0))
        lines.append(macro)
        if "loss" in metrics:
            lines.append("  loss={:.4f}".format(metrics["loss"]))
        return "\n".join(lines)

    # ---------------------------------------------------------------- images
    def log_seg_image(self, step, name, image_chw, pred_hw, target_hw=None):
        if self.writer is None:
            return
        height, width = image_chw.shape[-2:]

        def _resize(mask):
            resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode="nearest"
            )[0, 0].long()
            return resized

        pred = _resize(pred_hw)
        image = _denormalize(image_chw)
        pred_rgb = _mask_to_rgb(pred)
        overlay = 0.5 * image.astype(np.float32) + 0.5 * pred_rgb.astype(np.float32)
        mask = pred.detach().cpu().numpy().astype(bool)
        overlay[mask] = image[mask].astype(np.float32) * 0.4 + np.array([255, 0, 0], dtype=np.float32) * 0.6
        self.writer.add_image("val/image/{}".format(name), image.transpose(2, 0, 1) / 255.0, step)
        self.writer.add_image("val/pred/{}".format(name), pred_rgb.transpose(2, 0, 1) / 255.0, step)
        self.writer.add_image("val/overlay/{}".format(name), np.clip(overlay, 0, 255).transpose(2, 0, 1) / 255.0, step)
        if target_hw is not None:
            gt_rgb = _mask_to_rgb(_resize(target_hw))
            self.writer.add_image("val/gt/{}".format(name), gt_rgb.transpose(2, 0, 1) / 255.0, step)

    # ------------------------------------------------------------- manifest
    def write_manifest(self, extra):
        manifest = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            **extra,
        }
        with open(os.path.join(self.run_dir, "run_manifest.json"), "w", encoding="utf8") as fout:
            json.dump(manifest, fout, indent=2, ensure_ascii=False)
        self.text_block("run/manifest", json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.text("Run directory: {}".format(self.run_dir))
