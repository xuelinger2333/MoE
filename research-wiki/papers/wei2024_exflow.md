---
type: paper
node_id: paper:wei2024_exflow
title: "Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference"
authors: ["Jinghan Yao", "Quentin Anthony", "Aamir Shafi", "Hari Subramoni", "Dhabaleswar K. Panda"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2401.08383"
  doi: null
  s2: null
tags: ["MoE", "inference", "placement", "integer-programming", "cross-layer", "prior-art"]
added: 2026-05-18T15:30:00Z
relevance: core
note: "Author list pulled from arXiv metadata expected at https://arxiv.org/abs/2401.08383 — verify before citation."
---

# Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference (ExFlow)

## One-line thesis

Use integer programming over conditional cross-layer routing probabilities to
place experts on GPUs, reducing cross-GPU routing latency in MoE inference.

## Problem / Gap

MoE inference suffers from expensive AllToAll communication when tokens cross
GPU boundaries between expert dispatches. Prior placement heuristics (random,
round-robin) ignore the structure that tokens routed to specific experts at
layer L are not uniformly distributed across experts at L+1.

## Method

- Profile **conditional probability of tokens' routing across multiple layers**
- Formulate an **integer programming** model that places experts to minimize
  expected cross-GPU dispatch
- Single AllToAll communication per layer (vs. two in prior implementations)
- Applied without fine-tuning or accuracy degradation

## Key Results

- **Up to 67%** reduction in cross-GPU routing latency
- **Up to 2.2×** improvement in inference throughput vs. SOTA MoE
  implementations
- Tested on MoE models with 8 to 64 experts

## Assumptions

- Token routing distribution at deployment matches the profile distribution
- Balanced placement (each GPU holds equal number of experts)
- Standard expert-parallel dispatch model (not pipelined-stay-at-expert)
- Inference (not training) workload

## Limitations / Failure Modes

- Static one-shot placement — does not adapt to workload drift
- Does not consider datacenter topology (treats all cross-GPU hops as equal cost)
- Evaluated on architectures up to 64 experts; behavior at larger expert counts
  uncertain
- Specific MoE families evaluated not confirmed from abstract alone

## Reusable Ingredients

- Conditional cross-layer routing probability extraction — exactly the signal
  measured in `idea:cross_layer_routing_predictability`
- Integer programming framework for balanced expert placement
- "One AllToAll instead of two" trick — applicable beyond placement

## Open Questions

- Does the gain transfer to aux-loss-free models (e.g., DeepSeek-V2-Lite)?
- How robust is the static placement to domain shift (code vs. NL vs. math)?
- What is the gap to a true online / drift-aware placement?

## Claims

- ExFlow's gain implies a 22–40% normalized cross-layer MI is exploitable to
  the tune of ≥50% cross-rank reduction — quantitative consistency check
  with our `claim:C3` succeeds at order-of-magnitude

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
- invalidates → idea:pipelined_runtime_placement
- addresses_gap → gap:G6
- extended_by ← paper:go2025_moetuner

## Relevance to This Project

**Critical prior art.** Killed `idea:pipelined_runtime_placement` because
ExFlow already solved the "cross-layer aware placement for MoE inference"
problem 16 months before this project's pivot attempt. Becomes a comparator
in any future measurement-heavy paper: re-running ExFlow's algorithm on our
3-model × 3-domain trace stack and reporting whether the gain generalizes
(MoETuner-style cross-model audit).
