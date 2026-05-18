---
type: experiment
node_id: exp:m5_compare
title: "M5 — Cross-model comparison: DeepSeek vs Qwen on the same probe"
status: completed
date: 2026-05-15T04:03:58Z
hardware: "offline (CPU)"
mode: analysis
inputs: ["exp:m1_deepseek_wikitext", "exp:m4_qwen_wikitext"]
outputs:
  - "outputs/figures/F7_compare_heatmap.{png,pdf}"
  - "outputs/figures/F8_compare_cdf.{png,pdf}"
  - "outputs/figures/model_compare.json"
tested_ideas: ["idea:routeweaver"]
verdicts:
  routeweaver: invalidated
tags: ["analysis", "comparison"]
---

# M5 — Cross-model comparison

## What was measured

Side-by-side analysis of `exp:m1_deepseek_wikitext` and `exp:m4_qwen_wikitext`
traces. Outputs (a) cross-model traffic CDF with uniform reference diagonal,
(b) cross-model best-clustered co-activation heatmaps, (c) machine-readable
JSON of all C1/C2 metrics for both models.

## Key visual finding

`F8_compare_cdf.png` shows DeepSeek and Qwen aggregate CDFs both essentially
overlay the uniform-routing reference line across the entire `[0, 1]` x-axis.
The difference between the two models is < line width.

`F7_compare_heatmap.png` shows the best-clustered layer of each model — both
exhibit the same scattered-bright-pixel pattern with no contiguous block
structure.

## Verdict

- idea:routeweaver → INVALIDATED. The failure is universal across two distinct
  MoE balancing schools (auxiliary-loss-free vs auxiliary-loss). It is a
  **property of the problem class**, not a property of one model.

## Reproducibility

```bash
python scripts/40_compare_models.py
```

## Connections

- inputs: exp:m1_deepseek_wikitext, exp:m4_qwen_wikitext
- invalidates: idea:routeweaver
