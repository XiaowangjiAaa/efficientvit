# EfficientViT for Crack Segmentation

Binary crack segmentation (background vs crack) built on the EfficientViT backbone.
This project keeps the EfficientViT model code and adds a complete, self-contained
crack-segmentation workflow: dataset preparation, training, validation, inference and
TensorBoard logging.

> The crack-segmentation pipeline (data preparation, augmentation, validation, metrics and
> inference) is the focus of this repository. Model code is based on
> [EfficientViT](https://github.com/mit-han-lab/efficientvit) (see [Attribution](#attribution)).

## Features

- **Model**: EfficientViT semantic-segmentation models (`b0/b1/b2/b3/l1/l2`) with a
  configurable number of classes (`n_classes`).
- **Data**: paired `Image/` + `Mask/` folders, strict binary-mask validation, deterministic
  7:2:1 train/val/test split (`splits/*.json`).
- **Training**: multi-scale random resize + random crop (keep-ratio), flips, photometric
  distortion, optional rotation; CE + Dice loss; AMP; poly/cosine LR with warmup;
  iteration-based schedule with periodic validation, best-by-mIoU checkpointing and
  iteration-based early stopping.
- **Evaluation**: metrics computed on the **original mask grid** (`aAcc`, `mAcc`, `mIoU`,
  `mDice`, `mPrecision`, `mRecall`, `mFscore`, plus per-class background/foreground);
  inference modes `resize` (keep-ratio, official-style) / `full` (native resolution) /
  `slide` (sliding window), with optional multi-scale TTA.
- **Inference**: single image or folder �?original-resolution binary `0/255` PNG.
- **Logging**: timestamped text log, JSONL scalars/metrics, run manifest and full
  TensorBoard integration (scalars + validation prediction images).

## Installation

One-click (creates an isolated conda env under `.venv-efficientvit`).

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1            # CUDA 11.8
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Cpu       # CPU only
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Extras 'sam,export,tflite'
```

Linux / WSL2 / macOS (Bash):
```bash
bash scripts/setup.sh                        # CUDA 11.8
bash scripts/setup.sh --cpu                  # CPU only
bash scripts/setup.sh --extras sam,export    # optional extras (or: all)
bash scripts/setup.sh --print-only           # preview, execute nothing
```

Manual (existing Python 3.10 environment):
```bash
pip install -r requirements.txt          # CUDA 11.8; use requirements-cpu.txt for CPU
pip install -e .                         # core package; extras: pip install -e ".[sam,export]"
```

`requirements.txt` / `requirements-cpu.txt` pin the PyTorch wheel index; `requirements-common.txt`
holds the shared non-PyTorch dependencies; `pyproject.toml` declares the package and optional
extras (`train`, `sam`, `export`, `tflite`, `dev`, `all`). The setup scripts set
`PYTHONNOUSERSITE=1` so a stale `~/.local` package cannot shadow the environment.

### Activate the environment

The setup scripts create a conda environment at `<repo>/.venv-efficientvit`. Activate it before
running any script.

Windows (PowerShell / cmd):
```powershell
conda activate .\.venv-efficientvit
# or by absolute path:
conda activate F:\path\to\efficientvit\.venv-efficientvit
```

Linux / WSL2 / macOS:
```bash
conda activate ./.venv-efficientvit
# or by absolute path:
conda activate /data/yinwu/efficientvit/.venv-efficientvit
```

If `conda activate` does not accept the relative path, register the prefix once and then activate
by name:
```bash
conda config --append envs_dirs <repo>
conda activate .venv-efficientvit
```

Alternatively use the interpreter directly without activating:
```bash
.venv-efficientvit/bin/python -s train_seg_model.py --config configs/seg/crack_b0.json          # Linux/WSL2/macOS
.venv-efficientvit\Scripts\python.exe -s train_seg_model.py --config configs/seg/crack_b0.json # Windows
```
`-s` (equivalently `PYTHONNOUSERSITE=1`) keeps a stale user-site package from shadowing the env.

## Dataset

```text
dataset/
├── Image/            # crack images  (e.g. 896x896, 1280x720, 1920x1080)
├── Mask/             # binary masks: 0 = background, 255 (or 1) = crack; paired by relative path
└── splits/           # generated: train.json, val.json, test.json, summary.json
```

- Pair images and masks by relative path (extension may differ); sub-directories are supported.
- Masks must be grayscale and binary (`{0, 1, 255}`); white is **foreground**, never an ignore label.
- Images and masks must have identical size; duplicate stems / missing pairs are hard errors.

Validate all pairs and write a deterministic 7:2:1 split:
```bash
python prepare_seg_dataset.py --root dataset
# custom ratios
python prepare_seg_dataset.py --root dataset --val-ratio 0.2 --test-ratio 0.1 --force
```

## Training

Experiment configs live in `configs/seg/` (JSON); any CLI flag overrides the config.

```bash
python train_seg_model.py --config configs/seg/crack_b0.json
# override anything from the CLI
python train_seg_model.py --config configs/seg/crack_b2.json --max_iters 40000 --eval_mode slide
```

Defaults in the provided configs: `max_iters=30000`, `batch_size=12`, `val_interval=265`,
`checkpoint_interval=265`, `save_on_val=true`, early stopping when mIoU does not improve for
`patience_iters=5000`.

Config schema (see `configs/seg/crack_b0.json`):
```json
{
  "data_root": "dataset", "split_dir": "splits",
  "model": "b0", "num_classes": 2, "crop_size": 512, "batch_size": 12, "workers": 0,
  "max_iters": 30000, "lr": 0.0004, "weight_decay": 0.01, "ce_weight": 1.0, "dice_weight": 1.0,
  "amp": true,
  "augmentation": { "mode": "random_crop", "resize_scale": [2048, 512], "ratio_range": [0.5, 2.0],
                    "cat_max_ratio": 0.75, "horizontal_flip_prob": 0.5, "vertical_flip_prob": 0.0,
                    "photometric_distortion": true, "rotate_prob": 0.0, "rotate_degree": 10.0 },
  "scheduler": { "type": "poly", "power": 1.0, "warmup_iters": 1500, "warmup_ratio": 1e-6 },
  "eval": { "mode": "resize", "val_scale": [2048, 512], "window_size": 512,
            "window_overlap": 0.25, "scales": null },
  "val_interval": 265, "checkpoint_interval": 265, "save_on_val": true,
  "early_stopping": { "patience_iters": 5000, "min_delta": 0.0 },
  "logging": { "interval": 20, "window_size": 20, "tensorboard": true },
  "gpus": "0", "work_dir": "work_dirs/seg/crack_b0"
}
```

Common CLI options:
| Option | Meaning |
| --- | --- |
| `--eval_mode` | `resize` (keep-ratio to `val_scale`) / `full` (native res) / `slide` (sliding window) |
| `--eval_scales` | multi-scale TTA factors, e.g. `--eval_scales 0.75 1.0 1.25` |
| `--backbone_init` | load backbone weights from a classification checkpoint |
| `--resume` | resume or evaluate a full checkpoint |
| `--eval_only` | skip training; evaluate `--resume` checkpoint on val/test |
| `--test` | run the held-out test split after training |
| `--amp` / `--no_tensorboard` | mixed precision / disable TensorBoard event writing |

## Evaluation

```bash
# evaluate a checkpoint on val + test (no training)
python train_seg_model.py --config configs/seg/crack_b0.json \
  --resume work_dirs/seg/crack_b0/<run>/best_mIoU_iter_30000.pth --eval_only --test
```

Metrics follow the MMSeg `IoUMetric` definition; `mIoU/mDice/mFscore` are macro averages over
both classes, and the per-class `foreground` row is what matters for crack. Validation/test
always restore predictions to each image's **original size** before computing metrics.

## Inference

```bash
python predict_seg_model.py --config configs/seg/crack_b0.json \
  --checkpoint work_dirs/seg/crack_b0/<run>/best_mIoU_iter_30000.pth \
  --input dataset/Image --output predictions --eval_mode resize
```
Outputs original-resolution `0/255` PNG masks (single file or a whole folder supported).

## Logging & TensorBoard

Each run directory contains:
```text
<work_dir>/<timestamp>/
├── resolved_config.json     final parsed arguments
├── run_manifest.json        python/torch/CUDA, device, params, dataset sizes, config
├── train.log                timestamped per-iteration log + per-validation table
├── scalars.jsonl            training scalars (JSON lines)
├── val_metrics.jsonl        full metrics per validation
├── test_metrics.jsonl       test metrics
└── events.out.tfevents.*    TensorBoard scalars + validation prediction images + text
```
TensorBoard scalars include `train/loss`, `train/ce`, `train/dice`, `train/loss_smoothed`,
`train/lr`, `time/iter`, `time/data`, all `val/*` and `test/*` metrics, and images
`val/image|pred|overlay|gt/*`.

```bash
python scripts/webui.py --logdir work_dirs --port 6006   # -> http://127.0.0.1:6006
```

## Project structure (added for crack segmentation)

```text
prepare_seg_dataset.py      pair/validate Image+Mask and write 7:2:1 splits
train_seg_model.py          training, validation, metrics, checkpointing, TensorBoard
predict_seg_model.py        inference to original-resolution binary masks
seg_logging.py              text / JSONL / TensorBoard logger
configs/seg/*.json          experiment configs (b0 / b2 / l1)
scripts/setup.ps1           one-click Windows environment setup
scripts/setup.sh            one-click Linux/WSL2/macOS environment setup
scripts/verify_environment.py  import + model forward self-check
scripts/webui.py            TensorBoard launcher
requirements*.txt           layered dependencies (CUDA / CPU / common)
pyproject.toml              package metadata, dependencies and optional extras
```

## Attribution

Model and backbone code are based on
[EfficientViT](https://github.com/mit-han-lab/efficientvit) (MIT Han Lab), released under the
Apache-2.0 License (`LICENSE`). See the file headers for original copyright notices.

```bibtex
@inproceedings{cai2023efficientvit,
  title={Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17302--17313},
  year={2023}
}
```
