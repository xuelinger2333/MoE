---
type: claim
node_id: claim:C3
title: "C3 — On a load-balanced converged MoE, cross-layer top-1 expert MI exceeds the shuffled-null by ≥ 0.3 nat (strong) — i.e., load balancing does not eliminate cross-layer routing predictability"
status: supported
testable_threshold: "Mean over (L, L+1) pairs of [observed MI − shuffled-null MI] ≥ 0.3 nats"
related_idea: "idea:cross_layer_routing_predictability"
created: 2026-05-15T10:00:00Z
confirmed: 2026-05-15T10:15:00Z
supporting_experiments: ["exp:m6_cross_layer_mi"]
tags: ["MoE", "mutual-information", "cross-layer"]
---

# C3 — Cross-layer routing MI exceeds shuffled-null by ≥ 0.3 nat

## Status: ✅ SUPPORTED (2026-05-15)

## Statement

For a converged, load-balanced MoE LLM running inference on natural text,
take the top-1 expert at each layer per token. For each adjacent layer pair
`(L, L+1)`, compute the mutual information `MI(e_L, e_{L+1})` from the joint
empirical distribution and compare to the MI obtained after shuffling the
`e_{L+1}` column across tokens. The claim holds if the mean of `(MI − null_MI)`
across all adjacent pairs is **≥ 0.3 nats** (strong predictability).

## Verdict (round-2: strict null + per-source distribution + triangulation)

| Experiment | Model | MI − i.i.d. null | MI − within-seq null (strict) | frac sources ≥0.5 nat |
|---|---|---|---|---|
| exp:m6 | Qwen1.5-MoE-A2.7B | 1.534 nat | **1.529 nat** (39.4% of H(L+1)) | **97.3%** |
| exp:m6 | DeepSeek-V2-Lite | 0.686 nat | **0.683 nat** (17.5% of H(L+1)) | **72.1%** |
| exp:m7 | OLMoE-1B-7B | 1.230 nat | **1.212 nat** (30.5% of H(L+1)) | **92.5%** |

Loose-vs-strict null gap is **<0.02 nat in every case** → sequence-level locality (H2) contributes nearly zero. Signal is genuinely per-token cross-layer.

The per-source entropy reduction histogram (F12) is bulk-shifted right of the 0.5-nat threshold for every model — the phenomenon is universal, not driven by a few cherry-picked extreme pairs.

Even at d=8 (skipping 7 layers): Qwen 1.306 / DeepSeek 0.543 nat (round 1; round 2 strict-null d≥2 numbers are similar — sequence-locality doesn't grow with distance).

## Strongest single observation

For Qwen layer pair (L=12, L+1=13): conditioned on `e_{12} = 36`, the next-layer
expert distribution puts 62% mass on a single expert (id 16) vs the ~1.5%
marginal probability — KL divergence 2.70 nat. See `outputs/figures/F10_qwen_cond_dist.png`.

## Interpretation

Per-layer load-balancing constrains marginals but leaves the joint
free. The model trained with this objective still develops a strongly
predictable routing *trajectory* — knowing which expert handled a token
at L tells you a lot about which expert will handle it at L+1 and even
at L+8. This is exploitable for prefetch (Pre-gated MoE-style), placement
(co-locate frequently-paired (e_L, e_{L+1}) experts), and compression
(fuse high-predictability paths).

The Qwen vs DeepSeek gap (1.534 vs 0.686 nat) is itself interesting and
worth characterizing — possibly a function of top_k (Qwen=4, DeepSeek=6:
larger top_k → noisier top-1, less predictability) or of balancing
mechanism (auxiliary-loss vs loss-free).

## Connections

- tested_by ← exp:m6_cross_layer_mi
- supports → idea:cross_layer_routing_predictability
- contrasts with: claim:C1 (within-layer concentration, dead) — orthogonal phenomena
