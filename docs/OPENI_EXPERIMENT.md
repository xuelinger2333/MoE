# OpenI H4 Experiment Runbook

This repo is arranged so a fresh OpenI JupyterLab debug task can clone it,
check the machine, build the Python environment, and run the H4 experiment.

## Recommended OpenI Task

- Task type: debug task / JupyterLab
- Resource: NVIDIA GPU
- Minimum GPU: 1x A100 40GB
- Internet: enabled
- Image: `ubuntu22.04-cuda12.4.0-py310-torch2.6.0`

## One-Command Path

Inside the OpenI terminal:

```bash
cd /tmp/code
git clone <YOUR_GITHUB_REPO_URL> MoE
cd MoE
HF_ENDPOINT=https://hf-mirror.com bash scripts/openi_run_h4.sh
```

The script performs three steps:

1. `scripts/openi_env_check.sh`
   Writes `outputs/openi_env/env_report.txt` with GPU, Python package,
   disk, network, and Hugging Face model reachability diagnostics.

2. `scripts/openi_setup_env.sh`
   Creates `.venv-openi` with `--system-site-packages`, keeps a working
   CUDA-enabled PyTorch when present, and installs the experiment packages.

3. `remote/openi_h4_experiment.py`
   Runs smoke first, then the full H4 comparison.

## Outputs

Expected output files:

```text
outputs/openi_env/env_report.txt
outputs/h4_openi/smoke/summary_smoke.json
outputs/h4_openi/full/summary.json
```

The key fields in `summary.json` are:

- `h4_ratio.ratio`: `J_multi_turn / J_shared_prefix`
- `affinity.session_id.mean`: session-ID hit rate
- `affinity.eamc.mean`: MoE-Infinity EAMC hit rate
- `affinity.random.mean`: random baseline hit rate
- `session_id_over_eamc`: session-ID / EAMC

## Useful Overrides

Use a smaller formal run:

```bash
CONVS=6 PREFIXES=4 BRANCHES=3 MAX_NEW_TOKENS=64 bash scripts/openi_run_h4.sh
```

Only check and setup, without running the model:

```bash
RUN_SMOKE=0 RUN_FULL=0 bash scripts/openi_run_h4.sh
```

Use a different model:

```bash
MODEL_ID=allenai/OLMoE-1B-7B-0924 bash scripts/openi_run_h4.sh
```

If Hugging Face main is reachable and the mirror is not:

```bash
HF_ENDPOINT=https://huggingface.co bash scripts/openi_run_h4.sh
```

## Common Failure Modes

- `nvidia-smi not found`: the task was not created with an NVIDIA GPU.
- `torch.cuda.is_available: False`: the image lacks CUDA-enabled PyTorch;
  rerun setup with `INSTALL_TORCH=1`.
- Hugging Face timeout: keep `HF_ENDPOINT=https://hf-mirror.com`, or pre-upload
  the model as an OpenI model resource and pass its local path as `MODEL_ID`.
- Out of memory: lower `MAX_TURN_TOKENS`, `MAX_NEW_TOKENS`, `CONVS`, or use a
  smaller MoE model.
