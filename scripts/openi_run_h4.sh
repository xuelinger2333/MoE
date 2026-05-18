#!/usr/bin/env bash
# Run the self-contained OpenI H4 experiment.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-openi}"
USE_VENV="${USE_VENV:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/h4_openi}"
RUN_ENV_CHECK="${RUN_ENV_CHECK:-1}"
RUN_SETUP="${RUN_SETUP:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"

CONVS="${CONVS:-12}"
TURNS="${TURNS:-5}"
PREFIXES="${PREFIXES:-8}"
BRANCHES="${BRANCHES:-4}"
MAX_TURN_TOKENS="${MAX_TURN_TOKENS:-192}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-96}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TOP_PCT="${TOP_PCT:-0.3}"
EAMC_CAPACITY="${EAMC_CAPACITY:-32}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"

if [[ "$RUN_ENV_CHECK" == "1" ]]; then
  bash scripts/openi_env_check.sh || true
fi

if [[ "$RUN_SETUP" == "1" ]]; then
  bash scripts/openi_setup_env.sh
fi

if [[ "$USE_VENV" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

export HF_ENDPOINT
mkdir -p "$OUT_DIR"

echo "=== H4 smoke run ==="
if [[ "$RUN_SMOKE" == "1" ]]; then
  python remote/openi_h4_experiment.py \
    --model "$MODEL_ID" \
    --out "$OUT_DIR/smoke" \
    --smoke
fi

echo "=== H4 full run ==="
if [[ "$RUN_FULL" == "1" ]]; then
  python remote/openi_h4_experiment.py \
    --model "$MODEL_ID" \
    --out "$OUT_DIR/full" \
    --convs "$CONVS" \
    --turns "$TURNS" \
    --prefixes "$PREFIXES" \
    --branches "$BRANCHES" \
    --max-turn-tokens "$MAX_TURN_TOKENS" \
    --max-prefix-tokens "$MAX_PREFIX_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --top-pct "$TOP_PCT" \
    --eamc-capacity "$EAMC_CAPACITY" \
    --seed "$SEED"
fi

echo
echo "Outputs:"
find "$OUT_DIR" -maxdepth 3 -type f | sort
