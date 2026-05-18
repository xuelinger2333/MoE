---
type: paper
node_id: paper:hwang2024_pregated_moe
title: "Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference"
authors: ["Ranggi Hwang", "et al."]
year: 2024
venue: "ISCA"
external_ids:
  arxiv: "2308.12066"
  doi: null
  s2: null
tags: ["MoE", "inference", "prefetch", "pre-gating", "algorithm-system-codesign"]
relevance: core
added: 2026-05-15T10:15:00Z
---

# Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference

## One-line thesis

Predict the next layer's expert selection from the *previous layer's*
hidden state (a tiny pre-gate network), then prefetch that expert's
weights from CPU before they are needed — letting a single GPU run
larger MoE models than its memory would normally allow.

## Problem / Gap

MoE inference on memory-constrained hardware is bottlenecked by expert
weight loading; on-demand fetching is too slow.

## Method

Add a small pre-gating network at layer L that predicts which expert
will be selected at L+1. Use the prediction to start the CPU→GPU
weight transfer one layer ahead.

## Key Results

Successfully runs MoE-LLMs on a single GPU that would not otherwise fit;
prediction accuracy is high enough that prefetched-but-wrong cases are
rare and the fallback path is acceptable.

## Reusable Ingredients

- The pre-gating predictor architecture (a tiny MLP on hidden state).
- The async prefetch pipeline scaffolding.

## Limitations / Failure Modes

- Treats cross-layer predictability as a *given* exploited by their
  method. Does not measure it as a phenomenon — no MI numbers, no
  decay-with-distance characterization, no cross-model comparison,
  no analysis of how training stages or load-balancing choices affect
  predictability. **This is exactly the gap that exp:m6 begins to fill.**
- Their predictor is *conditional on hidden state*, not on expert id
  alone. Our M6 measurement is a strict lower bound — using the full
  hidden state would make MI even higher.

## Relevance to This Project

The single most important paper for `idea:cross_layer_routing_predictability`.
Pre-gated MoE proves the phenomenon can be exploited by a *method*; we are
asking the more basic question — *how strong is the phenomenon, on which
models, and how does it decay with distance and training stage?* — which
they don't answer.

## Connections

- inspired_by ← idea:cross_layer_routing_predictability
- linked_to: gap:G6
