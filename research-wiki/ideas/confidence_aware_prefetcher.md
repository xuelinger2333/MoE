---
type: idea
node_id: idea:confidence_aware_prefetcher
title: "Confidence-Aware Cross-Layer Expert Prefetcher"
stage: proposed
outcome: null
status: candidate
proposed: 2026-05-15T07:00:00Z
based_on: ["paper:hwang2024_pregated_moe", "exp:m9_router_entropy"]
target_gaps: ["gap:G8"]
tested_by: []
tags: ["MoE", "prefetch", "system", "confidence"]
---

# Confidence-Aware Cross-Layer Expert Prefetcher

## Status: 🟡 candidate (proposed from exp:m9 disambiguation)

## Statement

Pre-gated MoE prefetches the next-layer expert based on a learned predictor.
Our exp:m9 shows that **(a)** prediction is hard precisely when the router
is less confident, and **(b)** the OOD/confidence component explains 15-60%
of the cross-domain prediction-quality gap depending on model.

This idea wraps any cross-layer expert predictor (Pre-gated MoE-style or
just a (e_L → e_{L+1}) lookup table) with a **confidence gate**:

- if `H(P(e_L | token)) < threshold`: trust the prediction, prefetch the
  predicted top-1 next expert.
- else: fall back to either (a) prefetching the top-K most likely next
  experts (wider net) or (b) skipping prefetch entirely.

The threshold is calibrated per-model (e.g., NL-corpus median entropy from
exp:m9). The system gracefully degrades on OOD inputs instead of suffering
miss penalties.

## Why it might work

- exp:m9 quantifies the dividend exactly: filtering to confident tokens
  closes 60% of the code-vs-NL MI gap on Qwen → simple confidence gate
  recovers most of the OOD-driven prefetch miss-rate.
- The full router entropy is computed BY THE GATE itself; we get it
  for free during the forward pass.

## What it does NOT solve

The structural residual (G8b, OLMoE 0.49 nat) is independent of confidence.

**Note (revised 2026-05-15)**: an earlier version of this section called
this "models with strong domain-specific training (OLMoE-style)". A
follow-up training-mix lookup (see exp:m9 §"Why is OLMoE the
structural-residual outlier") falsified that — OLMoE's published code
share (~2.5%, OLMoE paper Table 2) is the **lowest** of the three, not
the highest. The right framing is: the structural residual is large
precisely on models whose router is already saturated (highest baseline
NL entropy), where the OOD/confidence axis has no dynamic range left.
For such models a **domain-conditional** (or layer-conditional)
predictor — not a confidence-conditional one — is the natural
complement. See gap:G9.

## Experiment plan

1. Build a per-(L, e_L) → top-K predictor from exp:m8 traces.
2. Evaluate on held-out tokens; measure top-1 / top-K hit rate.
3. Add the confidence gate. Measure the same hit rates as a function of
   entropy threshold.
4. Compare to the unconditional Pre-gated MoE baseline.

## Connections

- inspired_by → paper:hwang2024_pregated_moe, exp:m9_router_entropy
- addresses_gap → gap:G8
