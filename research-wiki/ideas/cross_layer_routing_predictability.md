---
type: idea
node_id: idea:cross_layer_routing_predictability
title: "Cross-Layer Routing Predictability — phenomenon-level measurement of joint P(e_L, e_{L+1})"
stage: tested
outcome: positive
status: ALIVE
proposed: 2026-05-15T10:00:00Z
confirmed: 2026-05-15T10:15:00Z
based_on: ["paper:hwang2024_pregated_moe"]
target_gaps: ["gap:G6"]
tested_by: ["exp:m6_cross_layer_mi"]
tags: ["MoE", "phenomenon", "routing", "mutual-information", "cross-layer"]
---

# Cross-Layer Routing Predictability — phenomenon-level measurement

## Status: ✅ ALIVE — strongly supported by exp:m6_cross_layer_mi

## Hypothesis

Per-layer load-balancing losses (auxiliary or auxiliary-loss-free) constrain
the **marginal** distribution `P(expert | layer)` to be uniform. They do **not**
constrain the **joint** distribution `P(e_L, e_{L+1})`. Therefore the conditional
`P(e_{L+1} | e_L = i)` may differ substantially from the marginal `P(e_{L+1})`
even on a perfectly load-balanced model — and it does.

## Why it survived precondition checks

- exp:m6_cross_layer_mi shows MI(L, L+1) − null = **1.534 nats** on Qwen
  (39.5% of H(L+1) explained) and **0.686 nats** on DeepSeek (22.0% of
  H(L+1) explained). Both exceed the 0.3-nat strong threshold by 2–5×.
- The signal **persists across distance**: MI − null at d=8 is still
  1.306 nats (Qwen) / 0.543 nats (DeepSeek). This isn't local noise; it's
  deep structural predictability of the routing trajectory.
- F10 visualization is striking: for one specific (L=12, e_L=36) pair on
  Qwen, the conditional P(e_{13} | ·) puts **62%** mass on a single expert
  (id 16) vs the marginal's ~1.5% — almost-deterministic forwarding.

## Round-2 robustness checks (2026-05-15, see exp:m7 + updated exp:m6)

Three concerns from a domain expert were addressed:

### Strict null (within-sequence shuffle)
The original "shuffled null" was a global i.i.d. shuffle that breaks all
structure. A *strict* null that shuffles `e_{L+1}` only within each 1024-token
sequence preserves topic-level coupling and breaks only the per-token (L, L+1)
pairing. **Result:** strict-null MI is ≤0.03 nat for every model — i.e., the
gap between MI and *strict* null is essentially the same as the gap between
MI and *loose* null (Qwen: 1.529 vs 1.534; DeepSeek: 0.683 vs 0.686; OLMoE:
1.212 vs 1.230). **Sequence-level locality (H2 hypothesis) contributes ~0%.
The cross-layer signal is genuinely per-token.**

### Per-source distribution (not cherry-picked)
For every (L, source_expert i), compute the entropy reduction
`H(P(e_{L+1})) − H(P(e_{L+1} | e_L=i))`. Histogram (F12):
- Qwen: 100% of (L, source) pairs reduce by ≥0.2 nat; **97.3% by ≥0.5 nat**.
- OLMoE: 99% by ≥0.2 nat; **92.5% by ≥0.5 nat**.
- DeepSeek: 92% by ≥0.2 nat; 72.1% by ≥0.5 nat.

The 0.62 conditional probability finding is NOT an outlier — virtually every
source expert at every layer substantially reduces next-layer uncertainty.

### Triangulation with OLMoE (third design-cube corner)
OLMoE-1B-7B (top-8, **aux-loss**) lands at MI−null = 1.212 nat — much closer
to Qwen (top-4, aux-loss; 1.529 nat) than to DeepSeek (top-6, **aux-loss-free**;
0.683 nat). **This isolates balancing scheme as the dominant factor over top-k
or depth.** Auxiliary-loss-free balancing (DeepSeek-V3 style) reduces but
does not eliminate cross-layer routing predictability. This is itself a finding
worth a paper bullet.

## Round-3 multi-domain stability (2026-05-15, see exp:m8 + claim:C4)

3 models × 3 domains (code / math / nl) = 9 cells. Headline:

| Model | code | math | nl | range |
|---|---|---|---|---|
| Qwen | 1.190 | 1.580 | 1.529 | 0.39 nat |
| DeepSeek | 0.545 | 0.652 | 0.683 | **0.14 nat** |
| OLMoE | 0.640 | 1.077 | 1.212 | **0.57 nat** |

- **H1a (domain-invariant) is rejected** — domain matters, but the phenomenon
  survives in every cell (all 9 cells > 0.3-nat strong threshold).
- **Code is the universally lowest-MI domain** across all three architectures.
  This is a clean, generalizable finding (gap:G8 — why?).
- **DeepSeek has the smallest domain spread (0.14 nat) AND the lowest absolute
  MI**. Both consistent with a single mechanism: aux-loss-free balancing
  flattens the joint distribution along multiple axes (gap:G7 strengthened).

System implication: a prefetcher tuned on NL workloads will underperform on
code workloads by 20-47% in MI. Domain-conditional predictors required for
production deployment.

## Why no prior paper killed this

- **Pre-gated MoE** (ISCA'24, paper:hwang2024_pregated_moe) uses essentially
  this signal as a *method* (predict next-layer expert from previous-layer
  hidden state, prefetch on CPU). They demonstrate the signal is exploitable
  but never measure it as a phenomenon, never quantify MI / KL, never compare
  models, never trace decay vs distance.
- **Occult** (paper:luo2025_occult) uses *within-layer* co-activation, not
  cross-layer.
- **NetMoE / SmartMoE / MoNTA** target placement, not the trajectory.

So the phenomenon is real, exploited tactically by one paper, but unmeasured
as a phenomenon — a clean opening.

## Publishable angles

1. **Measurement paper.** Systematic characterization across model
   families (Mixtral / DeepSeek / Qwen / OLMoE), training stages
   (early vs converged), domains, and layer distances. The M6 result
   is a single data point on this map; expand to a measurement study.
2. **Trajectory-aware prefetch / placement.** Use the conditional
   structure to prefetch likely-next experts onto the same rank as
   the current. Saves a cross-rank hop for the predicted ones.
3. **Cross-layer regularization.** A new training objective that
   *encourages* (or *discourages*) cross-layer correlation, depending
   on whether you want predictability (for systems) or
   diversity (for capacity).
4. **Compression/prune insight.** If `P(e_{L+1} | e_L = i)` is highly
   peaked, the (i, top-conditional-expert) pair acts as a "super-expert
   path". A model could be compressed by pre-fusing these paths.

## Anti-repetition memory

- Do NOT confuse this with within-layer co-activation (claim:C2 — that
  one is dead). Cross-layer is a different and live phenomenon.
- DeepSeek vs Qwen show **different magnitudes** (DeepSeek ~half of
  Qwen's MI). Need to characterize what training/architecture choices
  modulate this — this is itself a research question.
- The d=8 number is striking (still > 1 nat on Qwen). DON'T assume
  the structure is local in depth — it's surprisingly long-range.

## Connections

- inspired_by → paper:hwang2024_pregated_moe
- addresses_gap → gap:G6
- tested_by ← exp:m6_cross_layer_mi
- supported_by ← exp:m6_cross_layer_mi (via claim:C3)
