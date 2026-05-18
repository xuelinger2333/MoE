# MoE H4 Locality Experiments

This repository contains MoE routing probes and the H4 experiment harness:

- compare multi-turn expert locality against the shared-prefix baseline
- report the headline ratio `J_multi_turn / J_shared_prefix`
- reproduce a MoE-Infinity-style EAMC baseline
- compare session-ID affinity, EAMC content matching, and random prefetching

The repo is designed to be cloned directly into an OpenI / JupyterLab GPU task.

## Quick Start On OpenI

Create an OpenI debug task with:

- NVIDIA GPU, preferably `1*A100 40GB`
- Internet enabled
- image `ubuntu22.04-cuda12.4.0-py310-torch2.6.0`

Then run:

```bash
cd /tmp/code
git clone <YOUR_GITHUB_REPO_URL> MoE
cd MoE
HF_ENDPOINT=https://hf-mirror.com bash scripts/openi_run_h4.sh
```

The run writes:

```text
outputs/openi_env/env_report.txt
outputs/h4_openi/smoke/summary_smoke.json
outputs/h4_openi/full/summary.json
```

See [docs/OPENI_EXPERIMENT.md](docs/OPENI_EXPERIMENT.md) for detailed run
options and troubleshooting.

## H4 Outputs

The main fields in `outputs/h4_openi/full/summary.json` are:

- `h4_ratio.ratio`: `J_multi_turn / J_shared_prefix`
- `h4_ratio.j_multi_turn_within`: adjacent-turn Jaccard within conversations
- `h4_ratio.j_shared_prefix_within`: branch Jaccard within shared prefixes
- `affinity.session_id.mean`: session-ID affinity hit rate
- `affinity.eamc.mean`: MoE-Infinity EAMC hit rate
- `affinity.random.mean`: random hit rate
- `session_id_over_eamc`: session-ID hit rate divided by EAMC hit rate

Interpretation:

- `J_multi_turn / J_shared_prefix ~= 1`: H4 is likely the same signal as
  shared-prefix locality.
- `> 0.5`: strong finding, because conversation-level locality approaches
  prefix-level locality without bit-exact shared prefixes.
- `< 0.2`: weak H4 signal.
- `session_id >= EAMC`: session ID is a cheap signal that matches or beats
  content matching.

## Local Sanity Checks

For local development:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
python scripts/120_h4_sanity.py
```

The synthetic sanity check validates:

- perfect conversation stickiness
- no-stickiness null
- shared-prefix stickiness
- session-ID / EAMC / random hit-rate behavior

## Repository Layout

- `remote/openi_h4_experiment.py`: self-contained OpenI experiment runner
- `scripts/openi_env_check.sh`: environment and model reachability report
- `scripts/openi_setup_env.sh`: Python environment setup
- `scripts/openi_run_h4.sh`: check + setup + smoke + full H4 run
- `analysis/`: reusable Jaccard, EAMC, and affinity analysis modules
- `scripts/100_probe_multiturn.py`: parquet trace collector for multi-turn data
- `scripts/110_probe_shared_prefix.py`: parquet trace collector for shared-prefix data
- `scripts/130_h4_analyze.py`: analysis driver for collected parquet traces
- `src/`: model loading, tracing hooks, trace writing utilities

## Notes

Large artifacts are intentionally ignored by git:

- `outputs/`
- parquet traces
- model weights (`*.safetensors`, `*.bin`, `*.pt`)

Keep results in OpenI storage or download them separately from the GitHub
source archive.
