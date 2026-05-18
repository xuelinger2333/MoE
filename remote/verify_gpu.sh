#!/usr/bin/env bash
# Sanity-check the Cloudlab node: 4× A100, drivers present, disk free, no other heavy users.
set -euo pipefail

echo "=== uname / OS ==="
uname -a
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    echo "OS: ${PRETTY_NAME:-unknown}"
fi

echo
echo "=== nvidia-smi ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers must be installed before continuing." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
A100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -ci 'A100' || true)
echo "GPU count: $GPU_COUNT (A100: $A100_COUNT)"

if [[ "$GPU_COUNT" -lt 4 ]]; then
    echo "WARN: expected ≥4 GPUs, got $GPU_COUNT" >&2
fi
if [[ "$A100_COUNT" -lt 1 ]]; then
    echo "WARN: no A100 detected" >&2
fi

echo
echo "=== CUDA / NVCC ==="
nvcc --version 2>/dev/null || echo "nvcc not in PATH (ok if torch ships its own runtime)"

echo
echo "=== CPU / RAM ==="
nproc && free -h | head -3

echo
echo "=== Disk free ==="
df -h /home "$HOME" / 2>/dev/null | sort -u

echo
echo "=== Python ==="
which python3 || true
python3 --version || true

echo
echo "=== Network reachability ==="
echo "huggingface.co:" && (curl -sSL -o /dev/null -w "  HTTP %{http_code}, %{size_download} bytes\n" -m 8 https://huggingface.co || echo "  unreachable")
echo "github.com:" && (curl -sSL -o /dev/null -w "  HTTP %{http_code}, %{size_download} bytes\n" -m 8 https://github.com || echo "  unreachable")

echo
echo "=== Done ==="
