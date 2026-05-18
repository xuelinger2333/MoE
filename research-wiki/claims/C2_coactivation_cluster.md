---
type: claim
node_id: claim:C2
title: "C2 — Expert co-activation on modern open-source MoE LLMs has cluster structure that hierarchical clustering can recover"
status: invalidated
testable_threshold: "Best silhouette score (across layers, k ∈ {2,4,8,16}) ≥ 0.20"
related_idea: "idea:routeweaver"
created: 2026-05-15T08:00:00Z
invalidated: 2026-05-15T09:26:00Z
invalidating_experiments: ["exp:m1_deepseek_wikitext", "exp:m4_qwen_wikitext"]
tags: ["MoE", "coactivation", "clustering"]
---

# C2 — Expert co-activation has clustered structure

## Status: ❌ INVALIDATED (2026-05-15)

## Statement

For each MoE layer, build the symmetric co-occurrence matrix `M[i][j]` = number
of tokens whose top-k expert set contains both expert i and expert j. Apply
hierarchical clustering with average linkage on `1 - M_normalized`. Compute
silhouette score for `k ∈ {2, 4, 8, 16}`. The claim holds if at least one
(layer, k) combination yields silhouette ≥ 0.20.

## Verdict by experiment

| Experiment | Best layer | Best k | Best silhouette | Verdict |
|---|---|---|---|---|
| exp:m1_deepseek_wikitext | 25 | 16 | 0.169 | invalidates (below 0.20) |
| exp:m4_qwen_wikitext | 15 | 16 | 0.186 | invalidates (below 0.20) |

Both fail with 0.169 / 0.186 — close enough to threshold to suggest weak structure
exists, but not enough to support clean hierarchical grouping. Heatmaps
(`outputs/figures/F4_coactivation_heatmap.png`,
`outputs/figures/F7_compare_heatmap.png`) show scattered hot pixels rather than
contiguous blocks.

## Interpretation

Some pairs are clearly preferred (normalized co-activation up to ~0.7–0.9 for a
few off-diagonal cells), but the *overall* distance matrix is too uniform for
hierarchical clustering to find big groups. Sparser methods (max-weight matching,
top-K pair pinning) might extract value; hierarchical clustering does not.

This is now logged as a methodology lesson under gap:G5.

## Connections

- tested_by ← exp:m1_deepseek_wikitext, exp:m4_qwen_wikitext
- supports → idea:routeweaver (would have, if true)
- invalidates → idea:routeweaver (actual outcome)
