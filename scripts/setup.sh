#!/usr/bin/env bash
# One-command environment setup for the EfficientViT crack-segmentation workflow
# (Linux / WSL2 / macOS). Mirrors scripts/setup.ps1.
#
# Usage:
#   bash scripts/setup.sh                       # CUDA 11.8 (default)
#   bash scripts/setup.sh --cpu                 # CPU only
#   bash scripts/setup.sh --extras sam,export   # optional extras (or: all)
#   bash scripts/setup.sh --env-path ~/envs/ev  # custom conda prefix
#   bash scripts/setup.sh --skip-verify         # skip the import check
#   bash scripts/setup.sh --print-only          # show commands, execute nothing
set -euo pipefail

ENV_PATH=""
CPU=0
EXTRAS=""
SKIP_VERIFY=0
PRINT_ONLY=0

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-path)    ENV_PATH="${2:?--env-path needs a value}"; shift 2 ;;
        --cpu)         CPU=1; shift ;;
        --extras)      EXTRAS="${2:?--extras needs a value}"; shift 2 ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --print-only)  PRINT_ONLY=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
if [[ -z "$ENV_PATH" ]]; then
    ENV_PATH="$PROJECT/.venv-efficientvit"
fi
PYTHON="$ENV_PATH/bin/python"

run_step() {
    local label="$1"; shift
    echo
    echo "== $label =="
    if [[ $PRINT_ONLY -eq 1 ]]; then
        echo "$@"
        return 0
    fi
    # Route the target interpreter through `conda run` so the env is properly activated (DLL/lib paths).
    if [[ "$1" == "$PYTHON" ]]; then
        conda run --no-capture-output --prefix "$ENV_PATH" "$PYTHON" "${@:2}"
    else
        "$@"
    fi
}

if [[ $PRINT_ONLY -eq 0 ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found. Install Miniconda/Anaconda and run this script from a shell where 'conda' is on PATH." >&2
        exit 1
    fi
    if [[ -e "$ENV_PATH" && ! -d "$ENV_PATH/conda-meta" ]]; then
        echo "Target exists but is not a Conda environment: $ENV_PATH. Choose another --env-path." >&2
        exit 1
    fi
    LOG_DIR="$PROJECT/work_dirs/setup"
    mkdir -p "$LOG_DIR"
    LOG="$LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
    echo "Installation log: $LOG"
    exec > >(tee -a "$LOG") 2>&1
fi

if [[ ! -d "$ENV_PATH/conda-meta" ]]; then
    run_step "Create Python 3.10 environment" conda create --prefix "$ENV_PATH" python=3.10 pip -y
fi
run_step "Check target Python version" "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
run_step "Install package tools" "$PYTHON" -m pip install 'pip<26' 'setuptools>=65,<81' wheel

if [[ $CPU -eq 1 ]]; then
    REQ="$PROJECT/requirements-cpu.txt"
    run_step "Install CPU PyTorch + dependencies" "$PYTHON" -m pip install -r "$REQ"
else
    REQ="$PROJECT/requirements.txt"
    run_step "Install CUDA 11.8 PyTorch + dependencies" "$PYTHON" -m pip install -r "$REQ"
fi

if [[ -n "$EXTRAS" ]]; then
    run_step "Register project with extras [$EXTRAS]" "$PYTHON" -m pip install -e "$PROJECT[$EXTRAS]"
else
    run_step "Register project (editable)" "$PYTHON" -m pip install -e "$PROJECT"
fi
run_step "Verify dependency consistency" "$PYTHON" -m pip check

if [[ $SKIP_VERIFY -eq 0 ]]; then
    run_step "Verify imports and EfficientViT seg model" "$PYTHON" "$SCRIPT_DIR/verify_environment.py"
fi

if [[ $PRINT_ONLY -eq 1 ]]; then
    echo
    echo "Preview complete; no commands executed and no files written."
    exit 0
fi

echo
echo "Ready. Activate: conda activate '$ENV_PATH'"
echo "Seg train : $PYTHON '$PROJECT/train_seg_model.py' --config '$PROJECT/configs/seg/crack_b0.json'"
echo "Seg eval  : $PYTHON '$PROJECT/train_seg_model.py' --config '$PROJECT/configs/seg/crack_b0.json' --resume <ckpt> --eval_only --test"
echo "Seg pred  : $PYTHON '$PROJECT/predict_seg_model.py' --config '$PROJECT/configs/seg/crack_b0.json' --checkpoint <ckpt> --input dataset/Image --output predictions"
echo "Data prep : $PYTHON '$PROJECT/prepare_seg_dataset.py' --root dataset"
echo "TensorBoard: $PYTHON '$PROJECT/scripts/webui.py' --logdir '$PROJECT/work_dirs'"
