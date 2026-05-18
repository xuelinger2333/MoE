---
type: claim
node_id: claim:C1
title: "C1 — Router-induced cross-rank traffic on modern open-source MoE LLMs is concentrated enough to support locality-aware placement"
status: invalidated
testable_threshold: "Top 20% of (src_rank, dst_expert) pairs carry ≥ 50% of cross-rank tokens"
related_idea: "idea:routeweaver"
created: 2026-05-15T08:00:00Z
invalidated: 2026-05-15T09:26:00Z
invalidating_experiments: ["exp:m1_deepseek_wikitext", "exp:m2_deepseek_multidomain", "exp:m4_qwen_wikitext"]
tags: ["MoE", "traffic", "locality"]
---

# C1 — Router-induced cross-rank traffic is concentrated

## Status: ❌ INVALIDATED (2026-05-15)

## Statement

For a representative open-source MoE LLM running inference on natural-text data
under simulated EP=4, the cumulative distribution of cross-rank tokens over
`(src_rank, dst_expert)` pairs is sufficiently concentrated that the top **20%**
of pairs carry **≥ 50%** of all cross-rank tokens. (Motivation-strong if ≥ 70%.)

## Verdict by experiment

| Experiment | Top-20% share | Verdict |
|---|---|---|
| exp:m1_deepseek_wikitext | 22.7% | invalidates (well below 50%) |
| exp:m2_deepseek_multidomain (wikitext) | 22.8% | invalidates |
| exp:m2_deepseek_multidomain (c4) | 21.9% | invalidates |
| exp:m2_deepseek_multidomain (mmlu) | 22.2% | invalidates |
| exp:m4_qwen_wikitext | 21.5% | invalidates |

Top-X% shares are within ~2 pp of X% for all X ∈ {5, 10, 20, 50} — i.e., the CDF
is essentially the uniform-routing diagonal.

## Interpretation

This is the *expected* outcome of modern auxiliary-loss / loss-free balancing
training objectives. It does not mean the experiment failed; it means the claim
was wrong about *converged* checkpoints. The claim is open for **early-training
checkpoints** (see claim:C1' implied by idea:routeweaver_early_train).

## Connections

- tested_by ← exp:m1_deepseek_wikitext, exp:m2_deepseek_multidomain, exp:m4_qwen_wikitext
- supports → idea:routeweaver (would have, if true)
- invalidates → idea:routeweaver (actual outcome)
