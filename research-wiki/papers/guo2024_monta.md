---
type: paper
node_id: paper:guo2024_monta
title: "MoNTA: Accelerating Mixture-of-Experts Training with Network-Traffic-Aware Parallel Optimization"
authors: ["Jingming Guo", "Yan Liu", "Yu Meng", "et al."]
year: 2024
venue: "arXiv (preprint)"
external_ids:
  arxiv: "2411.00662"
  doi: null
  s2: null
tags: ["MoE", "network-aware", "parallel-strategy", "training"]
relevance: related
added: 2026-05-15T09:24:30Z
---

# MoNTA: Accelerating MoE Training with Network-Traffic-Aware Parallel Optimization

## One-line thesis

Search the parallel strategy + chunking that minimizes intra/inter-node
AllToAll traffic for the workload at hand.

## Problem / Gap

Static parallel strategies misalign with the dynamic, non-uniform AllToAll
patterns of MoE.

## Method

Offline search over (DP, EP, chunk size, pipeline stage) given a network
cost model.

## Key Results

Speedups on A800 cluster vs DeepSpeed-MoE; explicit modeling of intra vs
inter-node bandwidth.

## Reusable Ingredients

- The cost-model formulation (intra vs inter per-pair bandwidth).
- The chunking parameter sweep methodology.

## Limitations / Failure Modes

Offline; doesn't react to runtime congestion or routing-distribution drift.

## Relevance to This Project

Inspiration for `idea:routeweaver`'s online-runtime ambition. MoNTA solves a
weaker (offline) version of the same coupling RouteWeaver targets. With C2
invalidated, the *online co-activation update* part of RouteWeaver dies;
MoNTA-style offline planning may still remain useful as a baseline.

## Connections

- inspired_by ← idea:routeweaver
- addresses_gap → gap:G2
