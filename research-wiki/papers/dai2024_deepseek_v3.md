---
type: paper
node_id: paper:dai2024_deepseek_v3
title: "DeepSeek-V3 Technical Report"
authors: ["DeepSeek-AI", "Damai Dai", "Daya Guo", "et al."]
year: 2024
venue: "arXiv (technical report)"
external_ids:
  arxiv: "2412.19437"
  doi: null
  s2: null
tags: ["MoE", "auxiliary-loss-free", "load-balance", "production"]
relevance: core
added: 2026-05-15T09:24:30Z
---

# DeepSeek-V3 Technical Report

## One-line thesis

Production-scale MoE with **auxiliary-loss-free balancing** (per-expert bias
adjusted online) so that load balance does not require a quality-degrading
auxiliary loss term.

## Problem / Gap

Classical auxiliary-loss balancing damages expert specialization; loss-free
balancing avoids that.

## Method

Online per-expert bias term; bias is increased when an expert is over-routed,
decreased when under-routed; gradient does not flow through the bias.

## Key Results

Strong specialization + uniform load on 671B-class MoE training.

## Reusable Ingredients

- The online bias adjustment as a load-balance technique.
- Quantitative evidence that the bias mechanism *succeeds* — relevant context
  for why we observed near-perfectly uniform routing in exp:m1.

## Relevance to This Project

This paper is the **direct cause** of exp:m1's negative result on
DeepSeek-V2-Lite (which inherits the same family of balancing techniques).
The paper's success is RouteWeaver's death warrant: if loss-free balancing
keeps routing uniform, there is nothing for RouteWeaver's medium path to
re-cluster.

## Connections

- extends → exp:m1_deepseek_wikitext (causally explains its uniform-routing observation)
- linked from gap:G4
