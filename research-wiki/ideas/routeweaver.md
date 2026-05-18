---
type: idea
node_id: idea:routeweaver
title: "RouteWeaver — router/placement/topology joint optimization runtime"
stage: tested
outcome: negative
status: DEAD
proposed: 2026-05-15T08:00:00Z
killed: 2026-05-15T09:26:30Z
based_on: ["paper:liu2025_netmoe", "paper:luo2025_occult", "paper:guo2024_monta"]
target_gaps: ["gap:G2"]
tested_by: ["exp:m1_deepseek_wikitext", "exp:m2_deepseek_multidomain", "exp:m4_qwen_wikitext", "exp:m5_compare"]
tags: ["MoE", "system", "runtime", "placement", "DEAD"]
---

# RouteWeaver — router/placement/topology joint optimization runtime

## Status: ❌ DEAD

Killed 2026-05-15 by `exp:m1_deepseek_wikitext` + `exp:m4_qwen_wikitext`. Both
empirical preconditions failed cleanly on two modern MoE checkpoints
representing the two main load-balancing schools (DeepSeek's auxiliary-loss-free
balancing, and Qwen's classical auxiliary-loss balancing). See post-mortem
below for what was learned and what survives as follow-up ideas.

## Original statement

An online runtime for MoE training/inference that closes the loop between three
signals current systems treat independently:

1. Router output (top-k per token, per layer, per step)
2. Expert placement / replica layout
3. Physical interconnect topology (NVLink-domain vs cross-node, live congestion, p99)

Three timescales:

- **Fast path** (per step) — batch permutation + dispatch plan from current cost matrix
- **Medium path** (per N steps) — expert grouping / replica placement from live co-activation
- **Slow path** (per checkpoint) — light router regularization to keep grouping stable

## Preconditions and how they failed

| Claim | Threshold | DeepSeek-V2-Lite | Qwen1.5-MoE-A2.7B | Verdict |
|---|---|---|---|---|
| C1 — top-20% (src,dst) pair share | ≥ 50% | 22.7% | 21.5% | ❌ both fail |
| C2 — best silhouette (any layer, any k) | ≥ 0.20 | 0.169 | 0.186 | ❌ both fail |

The CDFs of both models lie within ~2 percentage points of the uniform
diagonal, and per-domain CDFs (wikitext / c4 / mmlu) also overlap. See
`outputs/figures/F8_compare_cdf.png` and `outputs/figures/F7_compare_heatmap.png`.

## Post-mortem — what killed it

1. **Modern MoE training explicitly minimizes the very skew RouteWeaver was
   built to exploit.** DeepSeek-V2 uses auxiliary-loss-free balancing; Qwen
   uses an auxiliary load-balance loss. Both succeed at producing routing that
   is statistically uniform across the expert set. From the network's view,
   every expert receives ~the same volume.
2. **Co-activation has visible non-uniformity at the pair level** (some pairs
   reach 4–10× the uniform expectation in normalized form), but the structure
   is *scattered* rather than *block-clustered*. Hierarchical clustering
   collapses to one big cluster + many singletons → silhouette < 0.20.
3. **Domain shift does not break uniformity.** All three open domains we
   probed (encyclopedic prose / web text / structured Q&A) produced CDFs
   that differ by < 1 percentage point — so the "dynamic per-domain
   re-grouping" rescue does not work either.

## Why this is still useful

The negative result is informative, not just a dead end:

- It **eliminates an entire family of medium-path placement heuristics** that
  assume routing skew on converged checkpoints.
- It **redirects the research question** to: *where DOES routing skew live?*
  → early-training checkpoints, narrow-domain fine-tuning, low-precision
  speculative regimes. These all become candidates.
- It produces a **reusable trace harness** (50M+ routing-event scale) that
  any follow-up idea can consume without re-engineering.

## Failure notes (anti-repetition memory)

- Do not propose any RouteWeaver variant whose first precondition is
  "modern MoE has skewed routing across experts at convergence". It does not.
- Do not propose hierarchical clustering on co-activation as a placement
  signal for these models. The structure exists but is too sparse for
  silhouette/dendrogram methods. Try sparse pair pinning or maximum-weight
  matching instead.
- Do not assume domain shift creates measurable routing-distribution shift
  on converged open-source MoE LLMs.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

- `inspired_by` → paper:liu2025_netmoe, paper:luo2025_occult, paper:guo2024_monta
- `addresses_gap` → gap:G2
- tested_by → exp:m1_deepseek_wikitext, exp:m2_deepseek_multidomain, exp:m4_qwen_wikitext, exp:m5_compare
- invalidated_by edges (from experiments): exp:m5_compare → idea:routeweaver

## Successor ideas

| Successor | Direction | Why it survives the death of routeweaver |
|---|---|---|
| idea:routeweaver_early_train | Probe early-training checkpoints | Routing balance is the *equilibrium* — early training should still show skew |
| idea:lossbound_ep | Per-token value-aware bounded-loss AllToAll | Does not require routing skew; consumes per-token value instead |
| idea:hybridep_collective | Hierarchical exact/approx EP collective primitive | Does not require routing skew; defines an API shift |
