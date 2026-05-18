#!/usr/bin/env bash
# Idempotent one-shot setup for the 算家云 4090 instance.
# Run via:  ssh moe-4090 'bash -s' < remote/setup_remote.sh
set -euo pipefail

echo "=== [1/6] DNS + apt deps ==="
grep -q 114.114.114.114 /etc/resolv.conf 2>/dev/null || {
  echo 'nameserver 114.114.114.114' >> /etc/resolv.conf
  echo 'nameserver 223.5.5.5' >> /etc/resolv.conf
}
apt-get install -y -q python3-pip python3-venv curl wget rsync screen tmux 2>&1 | tail -3

echo "=== [2/6] pip mirror + cache to sj-tmp ==="
mkdir -p /root/.config/pip /root/sj-tmp/pip_cache /root/sj-tmp/hf_cache
cat > /root/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
              mirror.sjtu.edu.cn
              download.pytorch.org
cache-dir = /root/sj-tmp/pip_cache
EOF
echo "pip mirror: aliyun + sj-tmp cache"

echo "=== [3/6] torch 2.3.1+cu121 from SJTU ==="
pip3 install --quiet --no-index --find-links https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/ \
    --trusted-host mirror.sjtu.edu.cn 'torch==2.3.1' 2>&1 | tail -3

echo "=== [4/6] HuggingFace stack from aliyun ==="
pip3 install --quiet \
    'transformers==4.46.0' 'datasets>=3.0' 'accelerate>=1.0' \
    'bitsandbytes>=0.43' 'pyarrow>=17' 'pandas>=2.2' 'numpy<2.2' \
    'scipy>=1.13' 'matplotlib>=3.9' 'tqdm>=4.66' 'einops>=0.8' \
    'sentencepiece>=0.2' 'huggingface_hub>=0.26' 'protobuf' 2>&1 | tail -3

echo "=== [5/6] HF cache env + mirror ==="
cat >> /root/.bashrc <<'EOF'
# H4 cheap test env
export HF_HOME=/root/sj-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/sj-tmp/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=https://hf-mirror.com
EOF

echo "=== [6/6] verify ==="
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/sj-tmp/hf_cache python3 - <<'PY'
import torch, transformers, datasets, accelerate, bitsandbytes
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
      "| compute:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
print("transformers:", transformers.__version__)
print("datasets:    ", datasets.__version__)
print("accelerate:  ", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
PY

df -h / /root/sj-tmp | head -8
echo "=== SETUP COMPLETE ==="
