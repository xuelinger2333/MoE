---
type: claim
node_id: claim:C4
title: "C4 — Cross-layer routing MI is domain-DEPENDENT but the phenomenon survives in every (model, domain) cell. Code is the universally least-predictable domain."
status: supported
testable_threshold: "Across 3 models × 3 domains, every cell clears the 0.3-nat strong threshold AND code has the lowest MI in each model"
related_idea: "idea:cross_layer_routing_predictability"
created: 2026-05-15T11:30:00Z
confirmed: 2026-05-15T11:30:00Z
supporting_experiments: ["exp:m8_multidomain_mi"]
tags: ["MoE", "domain-shift", "stability", "mutual-information"]
---

# C4 — Cross-layer MI is domain-dependent but universal in shape

## Status: ✅ SUPPORTED (2026-05-15)

## Statement

For every (model, domain) cell in the 3 × 3 grid {Qwen1.5-MoE / DeepSeek-V2-Lite / OLMoE-1B-7B} × {code / math / nl}, the per-token cross-layer MI exceeds the strict (within-sequence) null by at least 0.3 nat (the strong-evidence threshold from claim:C3). Within each model, the ranking is consistent: **code < math ≈ nl**, with code always the lowest-MI domain. The magnitude of the domain spread varies by model.

## Verdict

| Model | code | math | nl | range | strong-threshold check |
|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 1.190 | 1.580 | 1.529 | 0.39 | all > 0.3 (4-5×) |
| DeepSeek-V2-Lite | 0.545 | 0.652 | 0.683 | 0.14 | all > 0.3 (1.8-2.3×) |
| OLMoE-1B-7B | 0.640 | 1.077 | 1.212 | 0.57 | all > 0.3 (2-4×) |

**9-of-9 cells exceed the strong threshold.** **3-of-3 models have code as the lowest cell.**

## Two sub-findings

1. **H1a (domain-invariant) is rejected** — domain matters, with ranges of
   0.14-0.57 nat depending on the model.
2. **The directionality is universal** — code is always the bottom-MI cell.
3. **The phenomenon doesn't break** — even the lowest cell (OLMoE × code = 0.640 nat) is well above the strong threshold.

## Mechanism (open — see gap:G8)

Why does code show systematically lower cross-layer MI? Three candidate
explanations:
- Code has more identifier-level diversity per token.
- Code is slightly OOD vs the predominantly-NL training mix → router less confident → top-1 less stable.
- Code's syntactic regularity is mediated by different expert combinations than prose.

These are not mutually exclusive and need ablation.

## Connections

- tested_by ← exp:m8_multidomain_mi
- supports → idea:cross_layer_routing_predictability (extends C3 to multi-domain)
- opens → gap:G8 (why code? mechanism unknown)
