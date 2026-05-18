---
type: paper
node_id: paper:go2025_moetuner
title: "MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing"
authors: ["Seokjin Go", "..."]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2502.06643"
  doi: null
  s2: null
tags: ["MoE", "inference", "placement", "ILP", "cross-layer", "prior-art", "load-balance"]
added: 2026-05-18T15:30:00Z
relevance: core
note: "Author list incomplete — only first author confirmed from search result. Verify full list before citation."
---

# MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

## One-line thesis

Two-stage ILP — (1) cluster experts within each layer by routing dependencies
to balance per-cluster load, then (2) assign clusters to GPUs minimizing
inter-GPU communication — to jointly solve expert placement and token routing
for MoE serving.

## Problem / Gap

Expert parallelism distributes experts across GPUs but suffers from:
- Unbalanced token routing → tail latency
- Communication skew from layer-independent placement
- No prior system jointly resolves load imbalance AND communication skew

## Method

Three-stage system: **Token Routing Profiling** → **ILP Optimization** →
**Custom Expert Parallelism Initialization**.

ILP optimization is two-stage:

**ILP 1 — Load-Balanced Expert Clustering**
- Group experts within each layer into clusters
- Objective: minimize absolute deviation between each cluster's token load
  and the per-layer average load
- Output: per-layer expert → cluster assignment

**ILP 2 — Cluster-to-GPU Assignment**
- Assign clusters from each layer to GPUs
- Objective: minimize inter-GPU communication
- Cost: number of tokens routed between expert pairs across adjacent layers
  (i.e., the inter-layer affinity matrix)
- Output: per-cluster → GPU mapping

The key insight is "**Affinity Towards Certain Experts Across Layers**" —
tokens routed to a specific expert at layer L tend to go to a limited set at
L+1, so placing those experts on the same GPU reduces remote routing.

## Key Results

- Evaluated on **Mixtral-8x7B** across configurations
- **9.3% end-to-end speedup** on single-node
- **17.5% end-to-end speedup** on multi-node
- Compared against Megatron-LM naive (round-robin) placement baseline

## Assumptions

- Balanced placement (cluster sizes equal across GPUs)
- Static one-shot optimization at deployment time
- Token routing profile representative of production workload
- Standard dispatch-and-return dispatch (each layer independent in execution
  even though placement considers cross-layer affinity)

## Limitations / Failure Modes

- **Only Mixtral evaluated.** Does not characterize cross-model variability —
  whether the gain generalizes to aux-loss-free models, top-k variation,
  smaller MoE architectures.
- Static placement — no drift detection or online re-optimization
- Topology-agnostic (treats all GPU-GPU pairs as equal cost) — addressed by
  follow-up `paper:cluster_topology_placement_2025`
- Does not jointly optimize with expert replication

## Reusable Ingredients

- Two-stage ILP formulation (clustering + assignment)
- Per-layer balanced-load constraint with cross-layer-aware assignment
- Routing profiling as input format

## Open Questions

- Does the 9.3%/17.5% gain transfer to Qwen / DeepSeek / OLMoE?
- How does the gain scale with cross-layer MI magnitude (e.g., predicted lower
  for aux-loss-free models per our `claim:C3` exp:m7)?
- What's the optimal cluster count per layer for different ep_sizes?

## Claims

- MoETuner's reported gains (9.3%/17.5%) are consistent with the 41-86%
  simulator-level reduction in cross-rank rate that our `analysis/placement_simulator.py`
  measures — but the *end-to-end* gain is 1 order of magnitude smaller because
  cross-rank reduction does not 1:1 convert to latency.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
- invalidates → idea:pipelined_runtime_placement
- extends → paper:wei2024_exflow
- extended_by ← paper:cluster_topology_placement_2025
- addresses_gap → gap:G6

## Relevance to This Project

**Critical prior art.** Confirmed kill of `idea:pipelined_runtime_placement`.
Becomes the headline comparator in any future measurement-heavy paper. The
gap MoETuner does NOT close — cross-model generalization — is our remaining
contribution surface:
- MoETuner: 1 model (Mixtral), 1 placement strategy → 1 number
- Our potential paper: 3 models × 3 domains × MoETuner baseline → 9 numbers
  + the mechanistic explanation (saturated-router) of why the gain varies
