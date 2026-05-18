---
type: experiment
node_id: exp:m4_qwen_wikitext
title: "M4 — Qwen1.5-MoE-A2.7B single-domain routing probe (WikiText-103)"
status: completed
date: 2026-05-15T03:56:15Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
model: "Qwen/Qwen1.5-MoE-A2.7B"
dataset: "Salesforce/wikitext (wikitext-103-raw-v1, validation)"
trace_dir: "outputs/traces/m4_qwen_wikitext/"
config:
  batch_size: 4
  seq_len: 1024
  n_batches: 50
  ep_size: 4
  num_routed_experts: 60
  top_k: 4
  num_moe_layers: 24
elapsed_s: 32.5
n_routing_events: 19672032
tested_claims: ["claim:C1", "claim:C2"]
verdicts:
  C1: invalidated
  C2: invalidated
tags: ["MoE", "Qwen", "probe", "wikitext", "cross-architecture-validation"]
notes: "Promoted from NICE-TO-HAVE to MUST-RUN after exp:m1_deepseek_wikitext failed both claims; goal was to verify the failure is not a DeepSeek-specific artifact of auxiliary-loss-free balancing."
---

# M4 — Qwen1.5-MoE-A2.7B cross-architecture probe

## What was measured

Same M1 probe re-run on Qwen1.5-MoE-A2.7B (60 experts, top-4, 24 MoE layers,
classical auxiliary-loss balancing). 19.7M routing events. Hook target was
`Qwen2MoeSparseMoeBlock` instead of DeepSeek's `MoEGate`.

## Headline results

| Metric | Value |
|---|---|
| C1 — top-20% (src,dst) pair share | 21.5% (≈ uniform 20%) |
| C1 — top-5% pair share | 5.7% |
| C1 — top-50% pair share | 52.2% |
| C2 — best silhouette | 0.186 (layer 15, k=16) |

Slightly more skew than DeepSeek (0.186 vs 0.169 silhouette) but still under
the 0.20 threshold.

## Verdict

- C1 → INVALIDATED. The failure is **not** a DeepSeek-specific artifact —
  Qwen's older auxiliary-loss balancing also produces near-uniform routing.
- C2 → INVALIDATED with marginally better silhouette.

## Why this matters

Confirms the post-mortem: modern MoE training of any flavor (auxiliary-loss
or loss-free) produces uniform routing on converged checkpoints. The death of
`idea:routeweaver` is not a one-model fluke.

## Connections

- tests claims: claim:C1, claim:C2
- invalidates: claim:C1, claim:C2
