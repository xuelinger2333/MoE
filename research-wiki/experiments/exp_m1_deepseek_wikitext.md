---
type: experiment
node_id: exp:m1_deepseek_wikitext
title: "M1 — DeepSeek-V2-Lite single-domain routing probe (WikiText-103)"
status: completed
date: 2026-05-15T03:38:30Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
model: "deepseek-ai/DeepSeek-V2-Lite"
dataset: "Salesforce/wikitext (wikitext-103-raw-v1, validation)"
trace_dir: "outputs/traces/m1_deepseek_wikitext/"
config:
  batch_size: 4
  seq_len: 1024
  n_batches: 50
  ep_size: 4
  num_routed_experts: 64
  top_k: 6
  num_moe_layers: 26
elapsed_s: 29.2
n_routing_events: 31948800
tested_claims: ["claim:C1", "claim:C2"]
verdicts:
  C1: invalidated
  C2: invalidated
tags: ["MoE", "DeepSeek", "probe", "wikitext"]
---

# M1 — DeepSeek-V2-Lite single-domain routing probe (WikiText-103)

## What was measured

Per-token, per-layer top-6 expert selection across 50 batches of WikiText-103
validation, packed to (batch=4, seq=1024). 31.9M routing events captured into
parquet shards. EP rank assignments simulated offline with `expert_id // 16`
(experts/rank) and `token_idx // 1024` (tokens/rank).

## Headline results

| Metric | Value |
|---|---|
| C1 — top-20% (src,dst) pair share | 22.7% (≈ uniform 20%) |
| C1 — top-5% pair share | 6.1% |
| C1 — top-50% pair share | 53.4% |
| C2 — best silhouette across all layers + k | 0.169 (layer 25, k=16) |
| Cross-rank token ratio (per-layer mean) | ~75% (= (ep-1)/ep) |

## Verdict

- C1 → INVALIDATED. Top-20% share is well below the 50% threshold; CDF lies
  on the uniform-routing diagonal.
- C2 → INVALIDATED. No (layer, k) clears 0.20.

## Reproducibility

```bash
ssh chen123@d8545-10s10501.wisc.cloudlab.us \
  'cd ~/MoE && source .venv/bin/activate && python scripts/10_probe_single.py'
```

Run config snapshot: `outputs/traces/m1_deepseek_wikitext/run_meta.json`.

## Connections

- tests claims: claim:C1, claim:C2
- invalidates: claim:C1, claim:C2 (and by extension idea:routeweaver)
