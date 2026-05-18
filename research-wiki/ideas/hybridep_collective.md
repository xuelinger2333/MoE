---
type: idea
node_id: idea:hybridep_collective
title: "HybridEP-Collective — hierarchical exact/approx EP primitive"
stage: proposed
outcome: null
status: candidate
proposed: 2026-05-15T09:27:00Z
based_on: ["paper:zhu2025_megascale_infer", "paper:warraich2025_optireduce"]
target_gaps: ["gap:G3"]
tested_by: []
tags: ["MoE", "collective", "primitive", "API"]
---

# HybridEP-Collective — hierarchical exact/approx EP primitive

## Status: 🟡 candidate (alternative direction from deep-research-report.md 方案三)

## Statement

A two-level expert-parallel collective primitive: intra-rank exact / high-bandwidth,
cross-rank chosen per (token, expert) from {exact-send, compressed-residual,
parity-coded-resend, skip-secondary-expert} based on token redundancy + expert
collaboration + deadline.

The publishable contribution is the **API shift**: today NCCL-style collectives
are byte-driven; HybridEP-Collective lifts the API to "this batch's metadata,
which class each pair is in, what its deadline is".

## Why this survives RouteWeaver's death

HybridEP-Collective doesn't need routing concentration. It treats *every* token
as input to a class-routing decision. It's a primitive, not a placement
heuristic.

## Risks

- High implementation cost — must integrate with DeepEP / UCCL-EP / Megatron Core.
- Inference-only first pass is realistic; training adds quality-guard complexity.

## Connections

- `inspired_by` → paper:zhu2025_megascale_infer, paper:warraich2025_optireduce
- `addresses_gap` → gap:G3
