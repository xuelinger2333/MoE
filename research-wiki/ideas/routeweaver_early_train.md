---
type: idea
node_id: idea:routeweaver_early_train
title: "RouteWeaver-EarlyTrain — locality-aware placement on pre-equilibrium MoE checkpoints"
stage: proposed
outcome: null
status: candidate
proposed: 2026-05-15T09:27:00Z
based_on: ["paper:liu2025_netmoe", "paper:dai2024_deepseek_v3"]
target_gaps: ["gap:G4"]
tested_by: []
tags: ["MoE", "system", "early-training", "placement"]
---

# RouteWeaver-EarlyTrain — locality-aware placement on pre-equilibrium MoE checkpoints

## Status: 🟡 candidate (proposed pivot from dead idea:routeweaver)

## Statement

Same closed-loop placement runtime as RouteWeaver, but **scoped to the regime
where routing is provably non-uniform**: the first 10–30% of training steps,
before auxiliary-loss / loss-free balancing pushes routing to its uniform
equilibrium. Hypothesis: in this window, expert load skew is real, locality
headroom is real, and a placement runtime can deliver measurable speedup.

## Preconditions to verify before committing

- C1' — early-training top-20% (src,dst) pair share is ≥ 50%, decaying over
  training to the converged-checkpoint baseline (~22%).
- C2' — early-training co-activation has silhouette ≥ 0.20 for at least one layer.
- C3' — speedup window is wide enough to amortize placement-update overhead.

## Why it might still die

- The early-training regime may be too short / too noisy to make a placement
  scheme stable.
- If the locality-skew decay is fast (say, within 1k steps), the scheme can't
  amortize its overhead.
- The gain may be subsumed by simpler tricks (warmup placement, then static).

## Reuses from idea:routeweaver

- The entire trace + EP-simulator stack (`src/probes/`, `analysis/*`) ports directly.
- The two claim definitions (C1', C2') just need a checkpoint-step axis added.

## Connections

- `inspired_by` → paper:liu2025_netmoe, paper:dai2024_deepseek_v3
- `addresses_gap` → gap:G4
