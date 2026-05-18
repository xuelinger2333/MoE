---
type: paper
node_id: paper:luo2025_occult
title: "Occult: Optimizing Collaborative Communications across Experts for Accelerated Parallel MoE Training and Inference"
authors: ["Shuqing Luo", "Pingzhi Li", "Jie Peng", "et al."]
year: 2025
venue: "ICML"
external_ids:
  arxiv: "2502.02432"
  doi: null
  s2: null
tags: ["MoE", "co-activation", "collocation", "approximate"]
relevance: core
added: 2026-05-15T09:24:30Z
---

# Occult: Optimizing Collaborative Communications across Experts for Accelerated Parallel MoE Training and Inference

## One-line thesis

Statistically frequent expert *co-activations* enable collocation and / or
controlled communication pruning, yielding either exact or controllably-lossy
speedup.

## Problem / Gap

Co-activated experts force redundant cross-device traffic when placed apart;
no system uses the co-activation matrix online.

## Method

Build per-layer co-activation matrix; solve placement with collaborative-cost
objective; offer exact and pruning modes.

## Key Results

Up to 1.5×+ end-to-end speedup with exact mode; controllable quality-speed
tradeoff with pruning.

## Reusable Ingredients

- The co-activation matrix formulation (which we re-implemented in
  `analysis/coactivation_matrix.py`).
- The "collaborative-cost" objective.

## Limitations / Failure Modes

- Assumes co-activation matrix has *cluster-able* structure. exp:m1+m4 show
  this assumption fails on converged DeepSeek-V2 / Qwen1.5-MoE (silhouette
  < 0.20). Occult's "frequent pair" angle may still survive — bright pairs
  exist, just not as clusters.

## Relevance to This Project

Primary inspiration for the C2 claim. After exp:m1+m4 invalidated cluster
structure, the "scattered hot pair" insight from gap:G5 might still be
exploited by an Occult-style pair-pinning method.

## Connections

- inspired_by ← idea:routeweaver
- linked from gap:G5
