#!/usr/bin/env bash
# Idempotently prepare the Python environment for the OpenI H4 MoE experiment.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-openi}"
USE_VENV="${USE_VENV:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu124}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B}"
PREWARM_MODEL="${PREWARM_MODEL:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

cd "$PROJECT_DIR"

echo "=== OpenI H4 setup ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "USE_VENV=$USE_VENV"
echo "HF_ENDPOINT=$HF_ENDPOINT"

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python --version
python -m pip install --upgrade pip setuptools wheel

PIP_BASE=(python -m pip install --upgrade --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST")

need_torch=0
if [[ "$INSTALL_TORCH" == "1" ]]; then
  need_torch=1
elif [[ "$INSTALL_TORCH" == "auto" ]]; then
  if ! python - <<'PY'
try:
    import torch
    assert torch.cuda.is_available()
    print(torch.__version__)
except Exception:
    raise SystemExit(1)
PY
  then
    need_torch=1
  fi
fi

if [[ "$need_torch" == "1" ]]; then
  echo "Installing torch==$TORCH_VERSION from $CUDA_WHEEL_INDEX"
  python -m pip install --upgrade "torch==$TORCH_VERSION" --index-url "$CUDA_WHEEL_INDEX"
else
  echo "Keeping existing CUDA-enabled torch."
fi

"${PIP_BASE[@]}" \
  "transformers>=4.46,<4.58" \
  "accelerate>=1.0.0" \
  "datasets>=3.0.0" \
  "huggingface_hub>=0.26.0" \
  "sentencepiece>=0.2.0" \
  "einops>=0.8.0" \
  "pandas>=2.2,<3.0" \
  "pyarrow>=17.0.0" \
  "numpy>=1.26,<2.2" \
  "scipy>=1.13.0" \
  "scikit-learn>=1.5.0" \
  "matplotlib>=3.9.0" \
  "seaborn>=0.13.0" \
  "tqdm>=4.66.0"

export HF_ENDPOINT

if [[ "$PREWARM_MODEL" == "1" ]]; then
  echo "Prewarming model cache for $MODEL_ID"
  python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="${MODEL_ID}",
    allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.safetensors", "tokenizer*"],
)
print("Cached at:", path)
PY
fi

echo
echo "=== Setup complete ==="
if [[ "$USE_VENV" == "1" ]]; then
  echo "Activate: source $VENV_DIR/bin/activate"
fi
echo "Check:    bash scripts/openi_env_check.sh"
echo "Run:      bash scripts/openi_run_h4.sh"
