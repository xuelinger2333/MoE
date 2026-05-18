#!/usr/bin/env bash
# Idempotent environment installer for the Cloudlab node.
# Installs uv, creates .venv, installs deps, pre-warms the HF cache for DeepSeek-V2-Lite.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/MoE}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HF_MODEL="${HF_MODEL:-deepseek-ai/DeepSeek-V2-Lite}"

cd "$PROJECT_DIR"

echo "=== [1/5] Install uv (if missing) ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo
echo "=== [2/5] Create / reuse .venv ==="
if [[ ! -d .venv ]]; then
    uv venv --python "$PYTHON_VERSION" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python --version

echo
echo "=== [3/5] Install PyTorch (CUDA 12.1) ==="
uv pip install --quiet torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo
echo "=== [4/5] Install project + remaining deps ==="
uv pip install --quiet -e .

echo
echo "=== [5/5] Pre-warm HF cache for $HF_MODEL ==="
DISK_FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -d 'G ')
if [[ "$DISK_FREE_GB" -lt 50 ]]; then
    echo "WARN: only ${DISK_FREE_GB}G free in \$HOME; model needs ~32G + cache headroom" >&2
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# Use snapshot_download for a resumable, parallel download.
python - <<PY
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id=os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-V2-Lite"),
    allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.safetensors", "tokenizer*"],
)
print("Cached at:", path)
PY

echo
echo "=== Setup complete ==="
echo "Activate with:  source $PROJECT_DIR/.venv/bin/activate"
echo "Smoke run:      python scripts/00_smoke.py"
