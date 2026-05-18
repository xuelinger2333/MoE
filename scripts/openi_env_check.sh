#!/usr/bin/env bash
# Check whether an OpenI/Jupyter GPU task is ready to run the H4 MoE experiment.
set -u

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/outputs/openi_env}"
REPORT_FILE="$REPORT_DIR/env_report.txt"
MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$REPORT_DIR"

exec > >(tee "$REPORT_FILE") 2>&1

echo "=== OpenI H4 environment check ==="
date -Is
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MODEL_ID=$MODEL_ID"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo

echo "=== OS / kernel ==="
uname -a || true
if [[ -f /etc/os-release ]]; then
  cat /etc/os-release
fi
echo

echo "=== CPU / RAM / disk ==="
nproc || true
free -h || true
df -h "$PROJECT_DIR" /tmp / 2>/dev/null | sort -u || true
echo

echo "=== NVIDIA GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  echo
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv || true
else
  echo "ERROR: nvidia-smi not found."
fi
echo

echo "=== Python executables ==="
command -v python || true
python --version || true
command -v pip || true
pip --version || true
echo

echo "=== Python packages ==="
python - <<'PY' || true
import importlib
mods = ["torch", "transformers", "accelerate", "datasets", "pandas", "numpy", "pyarrow", "sklearn", "matplotlib", "huggingface_hub"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "ok")
        print(f"{name}: {ver}")
    except Exception as exc:
        print(f"{name}: MISSING ({type(exc).__name__}: {exc})")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda.device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"cuda[{i}]: {props.name}, {props.total_memory / 2**30:.1f} GiB")
except Exception as exc:
    print("torch cuda probe failed:", type(exc).__name__, exc)
PY
echo

echo "=== Network probes ==="
for url in \
  "https://github.com" \
  "https://huggingface.co" \
  "$HF_ENDPOINT" \
  "https://pypi.org/simple/transformers/" \
  "https://mirrors.aliyun.com/pypi/simple/transformers/"
do
  printf "%s -> " "$url"
  curl -L -sS -o /dev/null -w "HTTP %{http_code}, %{time_total}s\n" --max-time 12 "$url" || echo "unreachable"
done
echo

echo "=== Hugging Face model metadata probe ==="
HF_ENDPOINT="$HF_ENDPOINT" python - <<PY || true
from huggingface_hub import hf_hub_download
model = "${MODEL_ID}"
try:
    path = hf_hub_download(model, "config.json")
    print("config.json:", path)
except Exception as exc:
    print("metadata probe failed:", type(exc).__name__, exc)
PY
echo

echo "=== Project files ==="
for path in \
  "remote/openi_h4_experiment.py" \
  "scripts/openi_setup_env.sh" \
  "scripts/openi_run_h4.sh" \
  "scripts/120_h4_sanity.py" \
  "scripts/130_h4_analyze.py" \
  "pyproject.toml"
do
  if [[ -e "$PROJECT_DIR/$path" ]]; then
    echo "OK: $path"
  else
    echo "MISSING: $path"
  fi
done
echo

echo "Wrote report: $REPORT_FILE"
