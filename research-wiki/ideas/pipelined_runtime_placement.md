---
type: idea
node_id: idea:pipelined_runtime_placement
title: "Cross-layer-aware expert placement as a plug-in to pipelined MoE runtimes (COMET / MegaScale-Infer)"
stage: killed
outcome: negative
status: DEAD
proposed: 2026-05-18T14:00:00Z
killed: 2026-05-18T15:30:00Z
based_on: ["idea:cross_layer_routing_predictability", "paper:zhu2025_megascale_infer"]
target_gaps: ["gap:G6"]
invalidated_by: ["paper:wei2024_exflow", "paper:go2025_moetuner", "paper:cluster_topology_placement_2025"]
tags: ["MoE", "system", "placement", "DEAD", "sniped-by-prior-art"]
---

# Cross-layer-aware expert placement as a plug-in to pipelined MoE runtimes

## Status: ❌ DEAD — sniped by prior art

Killed 2026-05-18 by literature lookup before any new code was written. The
"novel cross-layer-aware placement on top of pipelined MoE runtime" framing
was already published in Jan 2024 (ExFlow) and Feb 2025 (MoETuner), with a
topology-aware follow-up in Aug 2025. Process win: the kill happened in
~30 minutes of WebFetch instead of weeks of implementation.

## Original statement

Take the cross-layer routing correlation phenomenon validated in
`idea:cross_layer_routing_predictability` (22–40% normalized MI in 9/9 cells,
robust against permutation null at 99% confidence) and convert it into a
**spatial placement contribution** that plugs into existing pipelined
MoE runtimes:

- Pipelined runtimes (COMET, MegaScale-Infer) solve the *temporal* problem
  (compute/communication overlap)
- We add the *spatial* layer: a cross-layer-aware expert placement optimizer
- Headline goal: additive X% end-to-end latency improvement on top of their
  already-optimized temporal scheduling

Simulator on the existing traces showed **37–86% incremental cross-rank
reduction** vs. per-layer-optimal baseline under a pipelined dispatch model
(see `outputs/H1_DECISION.md` Exp 2 + `outputs/placement_sim/summary.csv`).

Plan composition (from `idea-stage/IDEA_REPORT_v2_pipelined_placement.md`):
- Plan A — static-only (offline placement)
- **A+C** (recommended) — + topology-aware cost model
- Plan B — drift detector ablation
- Plan D — k=2 joint trajectory ablation

## What killed it (lit lookup, 2026-05-18)

| Prior art | Date | What it already does | Severity |
|---|---|---|---|
| **ExFlow** [paper:wei2024_exflow] | Jan 2024 | Integer programming on conditional cross-layer routing probabilities. Reports up to 67% cross-GPU routing latency reduction; up to 2.2× throughput. | Plan A territory fully covered |
| **MoETuner** [paper:go2025_moetuner] | Feb 2025 | Two-stage ILP: (1) cluster experts within layer by routing dependencies, (2) assign clusters to GPUs minimizing inter-GPU comm. End-to-end 9.3% single-node / 17.5% multi-node speedup on Mixtral-8x7B vs Megatron-LM. | Plan A+B territory fully covered |
| **Cluster Topology-Driven Placement** [paper:cluster_topology_placement_2025] | Aug 2025 | MoETuner's ILP + datacenter topology cost (NVLink vs IB). | Plan C territory fully covered |

## Math behind the prior-art collision

Under the pipelined cross-rank metric we use:

```
Pipelined cost = Σ_t Σ_L  1[place(L, e_L^t) ≠ place(L-1, e_{L-1}^t)]
              = Σ_L Σ_{e1,e2}  count(e_{L-1}=e2, e_L=e1) · 1[place(L,e1) ≠ place(L-1,e2)]
```

The second line factors over the inter-layer expert affinity matrix
count(e_{L-1}, e_L), which is **exactly** MoETuner's ILP input. Our trajectory
simulator (`analysis/placement_simulator.py`) is a **greedy heuristic** for
this objective. Greedy ≤ ILP optimum, so MoETuner is provably no worse than
our trajectory under the shared objective. Predicted E0 outcome:
`delta = (ours − MoETuner) / MoETuner ≈ 0 or negative`.

## Post-mortem — what we learned

1. **The cross-layer signal is real and our measurement is sound** — the H1
   finding (`idea:cross_layer_routing_predictability`, claims C3/C4) survives.
   It's the *system-paper grafting* that's been done.
2. **Prior art moves fast in MoE-systems.** ExFlow → MoETuner → topology
   follow-up is a 19-month chain. The Aug 2025 paper preemptively closed
   what was going to be our Plan C extension.
3. **What still might be ours** (none alone are system-paper headlines):
   - Measurement protocol (MI + null + filtered-MI) — cleaner than co-activation counts
   - Cross-model audit (Qwen + DeepSeek + OLMoE × code/math/nl = 9 cells) — MoETuner only ran Mixtral
   - Mechanism (saturated-router hypothesis, G7 aux-loss-free halves MI)
   - Predictive observation (DeepSeek's low-MI / high-placement-slack inversion)
4. **The right pivot is measurement-heavy, not system-heavy.** MoETuner
   becomes a *comparator* (cite + run on our traces to characterize the
   cross-model gain), not a *competitor*.

## Decision (recorded for future-self)

Retreat to measurement-heavy framing per `outputs/H1_DECISION.md`. Open
a new idea node when the measurement angle is concretely scoped — likely
title: "Cross-model placement-gain audit: when does ExFlow/MoETuner help?".

## Paths NOT pursued (open for future-self)

- **Path 3a — online adaptive placement under workload drift.** ExFlow/MoETuner
  are static one-shot. A real online placement layer with bounded migration
  cost is not in the existing literature (would need a thorough novelty check).
- **Path 3b — placement-aware router training.** Train the router with a
  placement-aware regularizer. Big scope; might violate "don't fight training
  objectives" (PLAN.md §1.3) — but maybe that lesson was scoped wrong.
- **Path 3c — joint placement + replication.** ExFlow/MoETuner enforce
  balanced placement. Replication for hot experts is in Lina/SmartMoE.
  Joint optimization is not (to my knowledge) studied.
- **Path 3d — sub-GPU granularity placement.** SM-level / weight-streaming /
  KV-cache locality. Needs deep CUDA work; not amenable to current hardware.

## Artifacts preserved

- `idea-stage/IDEA_REPORT_v2_pipelined_placement.md` — full 10-section brainstorm of the killed plan
- `idea-stage/IDEA_CANDIDATES.md` — compact summary
- `analysis/placement_simulator.py` — trajectory simulator (still useful — feeds the measurement paper's Exp 2)
- `scripts/90_placement_simulator.py` — driver
- `outputs/placement_sim/` — 18-config sweep with cross-rank rates under DR + PL metrics
- `outputs/permutation_null/` — Exp 1 results (orthogonal — independently strong)
- `outputs/H1_DECISION.md` — pre-pivot decision memo (still valid for measurement framing)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
- inspired_by → idea:cross_layer_routing_predictability
- inspired_by → paper:zhu2025_megascale_infer
- invalidated_by ← paper:wei2024_exflow
- invalidated_by ← paper:go2025_moetuner
- invalidated_by ← paper:cluster_topology_placement_2025
- addresses_gap → gap:G6

## Relevance to This Project

Negative result, kept for anti-repetition memory. Any future idea that tries
to claim "we add cross-layer placement to a pipelined runtime" must first
read this kill record and the three prior-art papers below.
