---
type: paper
node_id: paper:warraich2025_optireduce
title: "OptiReduce: Resilient and Tail-Optimal AllReduce for Distributed Deep Learning in the Cloud"
authors: ["Ertza Warraich", "Omer Shabtai", "Khalid Manaa", "et al."]
year: 2025
venue: "NSDI"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["transport", "AllReduce", "tail-latency", "bounded-loss"]
relevance: related
added: 2026-05-15T09:24:30Z
---

# OptiReduce: Resilient and Tail-Optimal AllReduce for Distributed Deep Learning in the Cloud

## One-line thesis

Tail-aware bounded-loss AllReduce — Hadamard-transform-based resilience
plus a tail-bypass primitive — gives publishable cloud-deployable transport
that tolerates bounded losses without converging-quality damage.

## Problem / Gap

Cloud-network jitter + straggler tail latency dominate distributed-DL transport.

## Method

Hadamard transform spreads gradient information so packet loss costs little;
explicit tail-bypass to skip slowest contributors per round.

## Key Results

Better tail behavior + competitive loss-tolerance vs strict-reliable AllReduce.

## Reusable Ingredients

- The bounded-loss-with-quality-budget framing.
- The shape of the per-step shadow / quality-guard plumbing.

## Limitations / Failure Modes

- Acts on gradient chunks, not on token-level semantics; doesn't transfer
  directly to AllToAll where token *value* is heterogeneous.

## Relevance to This Project

Primary inspiration for `idea:lossbound_ep`. After RouteWeaver died,
LossBound-EP becomes the strongest still-publishable direction; OptiReduce
provides both the precedent ("bounded-loss is acceptable in cloud DL
transport") and the warning (HOW you spread the loss matters).

## Connections

- inspired_by ← idea:lossbound_ep, idea:hybridep_collective
- addresses_gap → gap:G1
