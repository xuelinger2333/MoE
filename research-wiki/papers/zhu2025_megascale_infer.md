---
type: paper
node_id: paper:zhu2025_megascale_infer
title: "MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism"
authors: ["Ruidong Zhu", "Ziheng Jiang", "Chao Jin", "et al."]
year: 2025
venue: "SIGCOMM"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["MoE", "serving", "disaggregated", "M2N", "primitive"]
relevance: related
added: 2026-05-15T09:24:30Z
---

# MegaScale-Infer: Serving MoE at Scale with Disaggregated Expert Parallelism

## One-line thesis

Disaggregate attention from FFN for MoE serving and route between them with a
custom M2N library that eliminates the GPU↔CPU copies / group-init / sync
costs of NCCL collectives.

## Problem / Gap

NCCL primitives are byte-driven and rigid; MoE serving at scale needs
metadata-driven, sparse, asymmetric primitives.

## Method

Architectural disaggregation + M2N library.

## Key Results

Production-scale MoE serving wins on throughput and tail vs naive collectives.

## Reusable Ingredients

- The M2N library design (custom EP transport).
- The disaggregation pattern (relevant to inference-only experiments).

## Relevance to This Project

Primary inspiration for `idea:hybridep_collective`. The "API shift from bytes
to metadata + classes" argument lifts directly from MegaScale-Infer.

## Connections

- inspired_by ← idea:hybridep_collective
- addresses_gap → gap:G3
