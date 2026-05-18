---
type: paper
node_id: paper:muennighoff2024_olmoe
title: "OLMoE: Open Mixture-of-Experts Language Models"
authors: ["Niklas Muennighoff", "Luca Soldaini", "Dirk Groeneveld", "et al."]
year: 2024
venue: "arXiv (preprint)"
external_ids:
  arxiv: "2409.02060"
  doi: null
  s2: null
tags: ["MoE", "open-weights", "OLMo", "auxiliary-loss"]
relevance: related
added: 2026-05-15T11:00:00Z
---

# OLMoE: Open Mixture-of-Experts Language Models

## One-line thesis

Fully open MoE LLM (1B active / 7B total, 64 experts, top-8) with public
training data, code, and intermediate checkpoints — a clean reference point
for any MoE research that needs full reproducibility.

## Reusable Ingredients

- Available intermediate training checkpoints — useful for
  idea:routeweaver_early_train (probing pre-equilibrium routing).
- Auxiliary-loss balancing (NOT aux-loss-free) — useful as a third corner
  in the design-space cube alongside Qwen (aux-loss) and DeepSeek (aux-loss-free).

## Relevance to This Project

- Used as the **triangulation model in exp:m7** to disentangle top-k vs
  balancing-scheme vs depth confounds.
- Public intermediate checkpoints make it the natural target for
  **idea:routeweaver_early_train** (probe early-training cross-layer MI evolution).

## Connections

- inspired_by ← idea:routeweaver_early_train, exp:m7_olmoe_wikitext
