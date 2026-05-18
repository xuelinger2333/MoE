---
type: paper
node_id: paper:liu2025_netmoe
title: "NetMoE: Accelerating MoE Training through Dynamic Sample Placement"
authors: ["Xinyi Liu", "Yujie Wang", "Fangcheng Fu", "et al."]
year: 2025
venue: "ICLR"
external_ids:
  arxiv: "2501.06370"
  doi: null
  s2: null
tags: ["MoE", "sample-placement", "locality", "training"]
relevance: core
added: 2026-05-15T09:24:30Z
---

# NetMoE: Accelerating MoE Training through Dynamic Sample Placement

## One-line thesis

Reorder samples across DP ranks each step so that more of each layer's
top-k expert traffic stays intra-node — i.e., the placement decision is on
*tokens*, not on *experts*.

## Problem / Gap

MoE AllToAll fights node-internal vs cross-node bandwidth asymmetry, and the
router doesn't know about physical topology.

## Method

Two-stage optimization (assignment + permutation) per step, hardware-aware
locality cost.

## Key Results

Up to 1.67× / 1.37× / 1.33× over FastMoE / FasterMoE / SmartMoE on 32×A800
with NVLink-400GB intra and IB-100GB inter.

## Reusable Ingredients

- The two-stage placement formulation
- The intra/inter-node bandwidth cost matrix abstraction

## Open Questions

- Does the gain survive on uniform-routing models like DeepSeek-V2 / Qwen-MoE?
  (Implicitly answered NO by exp:m1+m4 at the *expert* level — but the *sample*
  level may still help.)

## Relevance to This Project

Primary inspiration for `idea:routeweaver`. After exp:m1+m4 invalidated the
expert-side placement angle, NetMoE-style sample placement remains the most
promising surviving direction (see `idea:routeweaver_early_train` and the
"re-scope to NetMoE-style sample placement" recommendation in EXPERIMENT_RESULTS.md).

## Connections

- inspired_by ← idea:routeweaver, idea:routeweaver_early_train
- addresses_gap → gap:G2
