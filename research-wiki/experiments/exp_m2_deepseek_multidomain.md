---
type: experiment
node_id: exp:m2_deepseek_multidomain
title: "M2 — DeepSeek-V2-Lite multi-domain routing probe (WikiText / C4 / MMLU)"
status: completed
date: 2026-05-15T03:45:39Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
model: "deepseek-ai/DeepSeek-V2-Lite"
datasets: ["wikitext", "c4 (streaming)", "mmlu"]
trace_dir: "outputs/traces/m2_deepseek_multidomain/"
config:
  batch_size: 4
  seq_len: 1024
  n_batches_per_domain: 15
  ep_size: 4
elapsed_s: 27.0
tested_claims: ["claim:C1"]
verdicts:
  C1: invalidated_per_domain
tags: ["MoE", "DeepSeek", "probe", "multidomain"]
notes: "Stack (bigcode/the-stack-smol) is HF-gated; dropped from default. MMLU dev split (285 examples) exhausted at 7 batches."
---

# M2 — DeepSeek-V2-Lite multi-domain routing probe

## What was measured

Same probe as M1, repeated across three open-domain datasets to test whether
routing skew differs by domain (which would justify a *dynamic* placement
runtime). Each domain ran 15 batches of (4, 1024) tokens.

## Headline results

| Domain | Top-20% (src,dst) share |
|---|---|
| WikiText-103 | 22.8% |
| C4 (en, streaming) | 21.9% |
| MMLU dev | 22.2% |

Per-domain CDFs overlap to within line-width on the F3 plot.

## Verdict

- C1 → INVALIDATED for every domain individually, with the same magnitude.
- The "dynamic per-domain re-grouping" rescue does not work — there is nothing
  to dynamically re-group, because the per-domain distributions are essentially
  identical and all near-uniform.

## Reproducibility

```bash
python scripts/20_probe_multidomain.py --domains wikitext c4 mmlu
```

## Connections

- tests claim: claim:C1
- invalidates: claim:C1 (per-domain rescue eliminated)
