---
type: paper
node_id: paper:gale2023_megablocks
title: "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"
authors: ["Trevor Gale", "Deepak Narayanan", "Cliff Young", "Matei Zaharia"]
year: 2023
venue: "MLSys"
external_ids:
  arxiv: "2211.15841"
  doi: null
  s2: null
tags: ["MoE", "block-sparse", "dropless", "kernels"]
relevance: core
added: 2026-05-15T09:24:30Z
---

# MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

## One-line thesis

Token dropping is the system community's free lunch and it costs model quality;
dropless MoE via block-sparse kernels is the right answer.

## Problem / Gap

Capacity-factor token dropping under-utilizes GPUs and harms convergence.

## Method

Custom block-sparse GPU kernels for variable-size expert batches; dropless
forward and backward.

## Key Results

Up to 40% faster than Tutel; eliminates token-dropping quality regression.

## Reusable Ingredients

- The "no token left behind" design constraint.
- Block-sparse kernel patterns.

## Limitations / Failure Modes

- Doesn't address cross-node bandwidth (orthogonal to RouteWeaver's question).

## Relevance to This Project

Strong negative signal for any approximate / drop-tokens design (informs
`idea:lossbound_ep`'s safety constraints — value-aware drops, not uniform).

## Connections

- inspired_by ← idea:lossbound_ep
- relevant to: claim:C1 (interpretation of why uniform routing is enforced)
