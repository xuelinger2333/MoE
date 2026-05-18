---
type: experiment
node_id: exp:m6_cross_layer_mi
title: "M6 — Cross-layer routing MI on Qwen + DeepSeek (offline replay of M1/M4 traces)"
status: completed
date: 2026-05-15T05:45:05Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501) — CPU-only analysis, no GPU"
mode: offline_analysis
seed: 42
inputs: ["exp:m1_deepseek_wikitext", "exp:m4_qwen_wikitext"]
outputs:
  - "outputs/figures/F9_cross_layer_mi.{png,pdf}"
  - "outputs/figures/F10_qwen_cond_dist.{png,pdf}"
  - "outputs/figures/F10_deepseek_cond_dist.{png,pdf}"
  - "outputs/figures/cross_layer_mi.json"
n_tokens_per_model: 204800
tested_claims: ["claim:C3"]
verdicts:
  C3: supported
elapsed_s: 8.4
tags: ["MoE", "mutual-information", "cross-layer", "offline-analysis"]
---

# M6 — Cross-layer routing MI

## What was measured

For each model (Qwen1.5-MoE-A2.7B and DeepSeek-V2-Lite), reuse the existing
M1/M4 traces (no new GPU work). For each token, extract its top-1 expert at
every MoE layer. For each layer pair `(L, L+d)` with `d ∈ {1, 2, 4, 8}`:
build the joint count matrix, compute MI in nats, compare to a shuffled-null
(3 random permutations of the `L+d` column), report `MI − null`.

Also pick the middle-layer pair, find the source expert with maximum
KL `(P(e_{L+1} | e_L = i) || P(e_{L+1}))`, and plot the conditional vs marginal.

## Headline results

| Model | Tokens | d=1 MI − null mean | d=1 normalized | d=8 MI − null mean | Mid pair top-source KL | Verdict |
|---|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B (60 experts, 24 layers) | 204,800 | **1.534 nat** | **39.5%** of H(L+1) | 1.306 nat | 2.696 nat | **STRONG** |
| DeepSeek-V2-Lite (64 experts, 26 layers) | 204,800 | **0.686 nat** | **22.0%** of H(L+1) | 0.543 nat | 3.631 nat | **STRONG** |

Strong threshold = 0.3 nat. Both models exceed it by 2–5×.

## Interpretation

The hypothesis is unambiguously alive. Per-layer load balancing does
**not** eliminate cross-layer routing structure; it just constrains the
marginals while leaving the joint free. Concrete consequences:

- A simple "predict next-layer expert from previous-layer expert id"
  baseline would explain 22-40% of the next layer's expert entropy.
- A more sophisticated predictor using the full top-k context could
  do even better.
- The structure persists across surprising depth (d=8).

The Qwen-vs-DeepSeek gap (1.5 vs 0.7 nat) is itself a finding worth
characterizing.

## Reproducibility

```bash
cd ~/MoE && source .venv/bin/activate
python scripts/50_cross_layer_mi.py
# Inputs read from outputs/traces/m1_deepseek_wikitext/, outputs/traces/m4_qwen_wikitext/
# Outputs written to outputs/figures/F9, F10, cross_layer_mi.json
```

Pure pandas/numpy; no GPU required. Total wall-clock 8.4 s on the Cloudlab box.

## Connections

- inputs ← exp:m1_deepseek_wikitext, exp:m4_qwen_wikitext (data reuse — zero new compute)
- tests claim: claim:C3
- supports: claim:C3, idea:cross_layer_routing_predictability
