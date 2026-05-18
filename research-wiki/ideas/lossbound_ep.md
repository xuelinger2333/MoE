---
type: idea
node_id: idea:lossbound_ep
title: "LossBound-EP — quality-budget-driven semantic AllToAll"
stage: proposed
outcome: null
status: candidate
proposed: 2026-05-15T09:27:00Z
based_on: ["paper:warraich2025_optireduce", "paper:gale2023_megablocks"]
target_gaps: ["gap:G1"]
tested_by: []
tags: ["MoE", "transport", "approximate", "quality-budget"]
---

# LossBound-EP — quality-budget-driven semantic AllToAll

## Status: 🟡 candidate (alternative direction from deep-research-report.md 方案一)

## Statement

An online quality-budget controller for MoE EP transport: per training round
or per inference batch, set a quality-degradation upper bound (Δloss / ΔPPL /
Δdownstream); runtime maximizes throughput / minimizes p99 within that budget
by classifying each (token, expert) pair into one of four delivery classes —
exact / priority-redundant / approximate-with-residual-reconstruction / defer.

Differs from OptiReduce/MLT in that the control variable is **MoE token
semantics** (router margin, rarity, layer criticality) rather than gradient
chunks.

## Why this survives RouteWeaver's death

LossBound-EP does **not** assume skewed routing. It exploits per-token *value*
heterogeneity, which is real even when expert *count* is uniform.

## Risks

- Value estimator failure mode: if it persistently down-weights rare-but-important
  tokens, expert specialization drifts. (See MegaBlocks — token dropping hurts
  quality.)
- Hard to evaluate "quality budget" cleanly during training — needs careful shadow
  monitoring infra.

## Connections

- `inspired_by` → paper:warraich2025_optireduce, paper:gale2023_megablocks
- `addresses_gap` → gap:G1
