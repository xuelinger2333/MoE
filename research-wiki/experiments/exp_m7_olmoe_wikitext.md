---
type: experiment
node_id: exp:m7_olmoe_wikitext
title: "M7 — OLMoE-1B-7B single-domain routing probe (WikiText-103) — triangulation point"
status: completed
date: 2026-05-15T06:07:44Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
model: "allenai/OLMoE-1B-7B-0924"
dataset: "Salesforce/wikitext (wikitext-103-raw-v1, validation)"
trace_dir: "outputs/traces/m7_olmoe_wikitext/"
config:
  batch_size: 4
  seq_len: 1024
  n_batches: 50
  ep_size: 4
  num_routed_experts: 64
  top_k: 8
  num_moe_layers: 16
elapsed_s: 23.7
n_routing_events: 26214400
tested_claims: ["claim:C3"]
verdicts:
  C3: supported
tags: ["MoE", "OLMoE", "probe", "wikitext", "triangulation", "aux-loss"]
notes: "Added in round 2 to triangulate the Qwen vs DeepSeek MI gap. OLMoE is top-8 + auxiliary-loss balancing — the third corner of the design-space cube."
---

# M7 — OLMoE-1B-7B routing probe (triangulation)

## Why this experiment

Round-1 found Qwen1.5-MoE-A2.7B's cross-layer MI (1.534 nat) was ~2× DeepSeek-V2-Lite's (0.686 nat). Three confounded explanations were on the table: top-k (Qwen=4, DeepSeek=6), balancing scheme (aux-loss vs aux-loss-free), depth (Qwen=24, DeepSeek=26).

OLMoE-1B-7B (top-**8**, **aux-loss**, 16 layers, 64 experts) was added as the third corner of the design-space cube to disentangle these effects.

## Result

`MI(L, L+1) − within-seq null` (averaged across all adjacent layer pairs):

| Model | top_k | balancing | n_layers | MI − strict null |
|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 4 | aux-loss | 24 | **1.529 nat** |
| **OLMoE-1B-7B** (new) | **8** | **aux-loss** | 16 | **1.212 nat** |
| DeepSeek-V2-Lite | 6 | aux-loss-free | 26 | **0.683 nat** |

OLMoE's higher top-k pushed MI down by only ~0.3 nat vs Qwen, but DeepSeek's loss-free balancing pushed MI down by ~0.85 nat vs the comparable aux-loss models.

**Verdict: balancing scheme dominates; top-k is a secondary effect.** Auxiliary-loss-free balancing reduces (but does not eliminate) cross-layer routing predictability. DeepSeek is the outlier in the three-model triangle.

## Reproducibility

```bash
python scripts/10_probe_single.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --out outputs/traces/m7_olmoe_wikitext \
    --batch_size 4 --seq_len 1024 --n_batches 50
```

## Connections

- tests claim: claim:C3
- supports: claim:C3, idea:cross_layer_routing_predictability
- triangulates: exp:m6_cross_layer_mi (Qwen + DeepSeek)
