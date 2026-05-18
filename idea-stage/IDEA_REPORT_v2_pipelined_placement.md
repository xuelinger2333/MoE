# Idea Report v2 — Cross-Layer-Aware Placement as a Plug-In to Pipelined MoE Runtimes

**Date**: 2026-05-18
**Source pivot**: from `idea-stage/IDEA_REPORT.md` (RouteWeaver three-timescale runtime) → narrower system positioning
**Evidence base**: `outputs/H1_DECISION.md` (Exp 1 + Exp 2 results from this session)
**Status**: pre-pilot brainstorm; literature verification points flagged inline

---

## 0. The reframing in one paragraph

Today's SOTA MoE inference runtimes split cleanly:
- **Temporal axis** — overlap compute and communication. COMET (kernel-level fine-grained overlap), MegaScale-Infer (prefill/decode disaggregation + pipelined dispatch), Lina, Tutel-Pipeline. *Solved-ish.*
- **Spatial axis** — which expert lives on which GPU. Default policy in all of the above is **layer-independent random / round-robin / hash placement** assigned at deployment time. *Not solved.*

Our finding: under a pipelined dispatch model (token's location at layer L+1 = location of its layer-L expert), the cross-layer routing correlation in real MoE models is exploitable to reduce cross-rank traffic by **41-86% beyond the per-layer-optimal baseline**. This is exactly the slack that no current runtime is taking.

**Repositioning claim**: We add a *cross-layer-aware placement layer* that plugs into existing pipelined runtimes (target: COMET-class or MegaScale-Infer-class) and yields additive X% end-to-end latency improvement on top of their already-optimized temporal scheduling.

Why this framing is stronger than "RouteWeaver three-timescale runtime":
- Doesn't compete with SOTA on the temporal axis (which is mature)
- Frames us as **the missing piece**, not a competitor
- Quantitative result (X%) sits on top of a strong baseline, harder for reviewers to dismiss
- Smaller engineering scope → realistic for a 1-2 person team to ship

---

## ⚠️ 0.5 Premises that need literature verification BEFORE committing

The whole pivot rests on three claims about prior systems. Flagging these explicitly so we don't anchor on assumptions:

| Premise | Status in my current understanding | Risk if wrong | Verification |
|---|---|---|---|
| **P1**: COMET uses round-robin / random expert placement (does not exploit routing structure) | Likely true — COMET's contribution is kernel-level dispatch overlap; placement is the user's choice | If false (COMET has hidden placement heuristic), our "additive X%" claim collapses | Read COMET (ASPLOS'25 or arXiv 2501.xxxxx) "expert placement" section + `placement.py`-equivalent in repo |
| **P2**: MegaScale-Infer places experts per-layer independently | Likely true — MegaScale-Infer's contribution is decode/prefill split + ping-pong execution; placement is orthogonal | Same as above | Read MegaScale-Infer (arXiv 2504.xxxxx) §3 / §4 |
| **P3**: Pre-gated MoE / SiDA / similar already exploit cross-layer signals for placement (not prefetch) | Possibly true for **prefetch**, less certain for **placement** | If a paper already did "cross-layer-aware placement on pipelined runtime", we are sniped | `/novelty-check "cross-layer expert placement pipelined moe"` — quick check, not blocking |

**Action**: 1 hour of focused reading before writing a single line of new code. Mark this as a **gating step** in the plan.

---

## 1. Four candidate plans (with trade-offs)

All plans share the same end-to-end shape:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ Calibration trace (one-time, 30-200 sec) on representative input │
  └──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────┐
              │ Cross-layer placement optimizer  │  <── this is our contribution
              └──────────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────┐
       │ Pipelined runtime  (COMET / MegaScale-Infer)   │  <── unchanged baseline
       └────────────────────────────────────────────────┘
                                  │
                                  ▼
                          End-to-end latency
```

What changes between plans is **the optimizer's scope and lifecycle**.

### Plan A — Static-only (offline-once placement)

**Idea**: Profile routing on a calibration trace, solve a balanced assignment that maximizes cross-layer co-location, deploy. Never re-optimized.

**Strengths**:
- Cleanest baseline. If this works, ship it.
- Zero runtime overhead.
- Lowest engineering risk — placement decision is offline pure-Python.
- Matches what every other "placement-aware" paper has done (NetMoE, Occult), so the comparison is direct.

**Weaknesses**:
- Assumes calibration trace ≈ production traffic. False on:
  - Domain shift (code → math at runtime)
  - Prefill vs decode phase mix changing
  - Multi-tenant workloads
- Cannot recover from bad initial placement.

**When this is enough**: single-tenant, stable workload, large dispatch latency (interconnect-bound).

**Estimated X**: 30-50% incremental over baseline (lower bound of our trace-level X).

### Plan B — Static + drift detector (semi-dynamic)

**Idea**: Plan A as default; lightweight statistical test on routing histograms (e.g., per-layer marginal KL divergence between trailing window and calibration); if drift > threshold, re-solve placement and migrate. Migration is expensive (~ms-scale on intra-node, ~s-scale inter-node), so the drift threshold is tuned to be conservative.

**Strengths**:
- Handles workload drift without paying online optimization cost on every request.
- Detector itself is O(layers × experts) per step → negligible.
- Recovers from bad initial placement after first drift event.

**Weaknesses**:
- Migration cost is non-trivial (weights move; could stall in-flight tokens).
- Risk of oscillation if threshold tuned wrong.
- Adds state to a previously stateless runtime.

**When this wins over Plan A**: serving stack with traffic mix variance, e.g., shared LLM serving multiple tenants.

**Estimated X**: ~Plan A + 5-15% recovery from drift.

### Plan C — Two-tier topology-aware placement

**Idea**: Real GPU clusters have a bandwidth hierarchy:
- Intra-node NVLink: ~600-900 GB/s
- Inter-node IB / NVLink-fabric: 100-400 GB/s
- Inter-node Ethernet: 25-100 GB/s

A "cross-rank" event isn't binary — its cost depends on which two ranks. Plan C makes the cost matrix topology-aware: it tries to keep correlated expert pairs in the **same node** even if they end up on different GPUs within the node.

**Strengths**:
- Maps directly to the metric that matters end-to-end (latency, not just hop count).
- Generalizes Plan A — Plan A is the special case with uniform cost matrix.
- Naturally handles multi-machine deployment (which the user explicitly asked about).
- Strong defense against "your simulator counts hops, not bytes" reviewer attack.

**Weaknesses**:
- Optimization is now a weighted quadratic assignment problem (NP-hard); needs heuristics or LP relaxation.
- Requires a topology model (probe the cluster at deploy time → cost matrix).
- Per-deployment retuning.

**When this wins**: any multi-machine deployment, especially with mixed interconnect (e.g., 2x 8-GPU nodes with Ethernet between).

**Estimated X**: 40-60% over per-layer-optimal *under realistic topology weighting*. The 41-86% from Exp 2 likely overstates the real-world gain because it assumed uniform cross-rank cost.

### Plan D — Joint k-layer trajectory optimizer

**Idea**: Plan A treats placement layer-by-layer (greedy). Plan D considers k consecutive layers jointly: maximize expected sum of (1 - cross_rank) across k layers. For k=2 this is a bilinear program; for k>2 it grows combinatorially.

**Strengths**:
- Captures cross-layer correlation at the depth where it actually exists (PLAN.md §2.2 shows MI at d=4, d=8 still > 0.8 nat).
- Theoretical upper bound: closer to the "oracle" placement.

**Weaknesses**:
- Combinatorial explosion. k=2 doable; k=4 needs LP relaxation / column generation; k=full model = NP-hard.
- Diminishing return: greedy k=1 (our Exp 2) already captures most of the easy correlation.
- Risk: looks like over-engineering. Reviewer asks "why not just k=1?" and we have to defend.

**When this wins**: as a small ablation row in the paper, NOT as the headline. "Our greedy k=1 captures 80% of the achievable saving; jumping to k=2 buys another 5-10%."

**Estimated X**: 5-15% incremental over Plan A.

---

## 2. Recommended plan composition

**Headline = Plan A + Plan C as one system, Plan B and Plan D as ablations.**

Rationale:
- Plan A alone is too simple for a top-venue system paper. Reviewers will ask about drift and topology.
- Plan C is the smallest extension of A that defends against the two strongest reviewer attacks (topology, byte-vs-hop).
- Plan B is a natural ablation row ("with vs without drift detector"). Skip if calibration covers expected workloads.
- Plan D is a theoretical ablation ("k-layer lookahead"). Show diminishing returns to justify k=1 as default.

**Architecture name suggestion**: drop "RouteWeaver" (was the old three-timescale framing). Try something like **Cohere** (cross-layer coherence placement), **Trajectree**, or just **CL-Place / XL-Place**. Naming is a writing-phase decision.

---

## 3. Challenges, ranked by severity (= "what kills the paper")

### C1 — API surface in COMET / MegaScale-Infer (SEVERITY: HIGH)
**Problem**: Neither system was designed with a pluggable placement layer. If COMET hard-codes `expert_id % world_size` in CUDA kernels, integration is a fork-and-patch nightmare.

**Resolution paths**:
- (a) **Check the source code** before plan finalization. COMET is open source. If placement is decoupled (e.g., a Python-level dispatch lookup table), great. If it's compiled into kernels, that's a problem.
- (b) Fork strategy: minimal-diff fork with a placement table override. Document the patch as "≤200 LOC change to enable the optimization".
- (c) Fallback: pick a less-optimized but more pluggable baseline (Tutel, DeepSpeed-MoE) and report numbers against COMET separately in a discussion section.

**Module**: [M1: Runtime integration layer]

### C2 — Replication policy collision (SEVERITY: HIGH)
**Problem**: All production MoE serving replicates popular experts to deal with hot-spots and load imbalance. Our placement assumes 1:1 expert-to-rank mapping. Replication breaks the placement decision (which copy do you dispatch to?).

**Resolution paths**:
- (a) **Acknowledge in scope**: "This work assumes 1:1 placement. Joint optimization with replication is future work." Honest, but weak for a system paper.
- (b) **Co-design**: extend the optimizer to choose both placement *and* replication factor per expert, subject to a memory budget. Strong contribution but bigger scope.
- (c) **Compose with existing replication**: take a replication plan as input (e.g., from Lina / hot-spot detector), constrain our optimizer to respect it. Cleanest separation of concerns.

**Recommendation**: option (c). Cite Lina / SmartMoE for replication; our work is orthogonal.

**Module**: [M5: Replication-aware optimizer]

### C3 — Load balance constraint vs trajectory preservation (SEVERITY: MEDIUM)
**Problem**: Per-rank compute load must be roughly balanced (one slow rank blocks all). Trajectory placement might cluster correlated experts on the same rank, violating compute balance.

**Resolution paths**:
- (a) Add an explicit load-balance constraint to the optimizer (e.g., max-load / min-load ratio ≤ 1.1).
- (b) Soft penalty in the objective.
- (c) Profile: does our trajectory placement actually violate balance? On stress traces it might not.

**Modules**: balance constraint is one line in the optimizer; the analysis (does it violate?) is a separate experiment.

### C4 — Cold-start / trace acquisition (SEVERITY: MEDIUM)
**Problem**: Need a routing trace before optimizing. Where does it come from? Cost?

**Resolution paths**:
- (a) Self-bootstrap: deploy with random placement, collect 30-60 sec of runtime trace, re-optimize, hot-swap.
- (b) Calibration set: ship a frozen calibration prompt set with the model. Per-deployment one-time profile (≤2 min).
- (c) Synthetic from router weights: feed a synthetic distribution through the router on a single GPU at deploy time (no actual forward through experts needed). Fast but distributional fidelity is a research question.

**Module**: [M2: Trace profiler]

### C5 — Decode vs prefill divergence (SEVERITY: MEDIUM)
**Problem**: Prefill routes batched parallel tokens; decode routes single tokens with KV-cache state. Routing distributions can differ. MegaScale-Infer disaggregates the two — your placement may need to differ per phase.

**Resolution paths**:
- (a) Separate placement per phase. Doubles the deployment complexity but aligns with MegaScale-Infer's existing decode/prefill split.
- (b) Show empirically (this is a missing experiment in PLAN.md §5.4) that the placement is robust across phases. If yes, ignore.
- (c) Per-phase rerun of the optimizer at calibration time. Cheap.

**Module**: [M2 trace profiler must tag prefill vs decode tokens]

### C6 — Static placement re-balancing under traffic skew (SEVERITY: LOW-MEDIUM)
**Problem**: Production LLM traffic is bursty / skewed across tenants. Static placement may be suboptimal for any single request even if globally optimal.

**Resolution paths**:
- (a) Plan B (drift detector) addresses the slow drift case.
- (b) For fast skew: this is a fundamentally hard problem; punt to future work. Mention in discussion.

### C7 — End-to-end latency measurement requires real hardware (SEVERITY: HIGH for the paper, not for the idea)
**Problem**: Simulator gives cross-rank rate. The paper needs **wall-clock latency speedup**. That requires a real run of COMET + our placement, on a real cluster.

**Resolution paths**:
- (a) **Single-node validation first**: 2× P100 on local box → measure cross-rank rate AND latency for ep=2. Confirms the simulator predicts real latency.
- (b) **Multi-node via cloud**: 2 nodes × A100 on Lambda / cloudlab / vast.ai. Necessary for the headline number. Cost: ~$500-2000.
- (c) Cite simulator as upper bound; demand: "≤2 weeks of real-cluster testing" as a hard scope.

**Module**: [M3: end-to-end benchmark harness] — this is a big chunk of work, not a small module.

### C8 — Pipelined dispatch is non-standard (THE caveat from H1_DECISION) (SEVERITY: VARIABLE)
**Problem**: Our metric assumes tokens stay at expert location between layers. This is exactly what COMET and MegaScale-Infer do (they pipeline through experts without returning to home rank for every layer). **This is the entire point of choosing these runtimes as the baseline.** But we need to verify per-runtime that this is in fact what they do.

**Resolution path**: Verify P1/P2 above. If true, this caveat disappears — we are simply quantifying what was always there but unused. If false, we have a fundamentally different paper.

---

## 4. Module map

For the recommended plan (A + C, with B/D as ablations):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CL-Place system                                 │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐  │
│  │ M1 Runtime   │◀──▶│ M3 Placement Table   │◀──▶│ M4 Optimizer Core│  │
│  │ Integration  │    │ (per-layer e → rank) │    │ (balanced QAP)   │  │
│  │ (COMET hook) │    └──────────────────────┘    └──────────────────┘  │
│  └──────┬───────┘                                          ▲           │
│         │                                                  │           │
│         ▼                                                  │           │
│  ┌──────────────┐    ┌──────────────────────┐              │           │
│  │ M2 Trace     │───▶│ M6 Topology Cost     │──────────────┤           │
│  │ Profiler     │    │ Model (NVLink / IB)  │              │           │
│  └──────────────┘    └──────────────────────┘              │           │
│         │                                                  │           │
│         ▼                                                  │           │
│  ┌──────────────┐    ┌──────────────────────┐              │           │
│  │ M5 Replica-  │───▶│ Constraints feed     │──────────────┘           │
│  │ tion-aware   │    │ (balance + replica)  │                          │
│  └──────────────┘    └──────────────────────┘                          │
│                                                                         │
│  Optional: M7 Drift Detector  →  triggers M4 re-solve + M1 hot-swap    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module specs

**M1 — Runtime integration layer**
- ≤200 LOC patch to COMET (or chosen runtime) to read placement from a table instead of `id % world_size`
- Hot-swappable: change the table → next request uses new placement
- Per-runtime: write 2 (COMET + MegaScale-Infer); pick the easier one as primary

**M2 — Trace profiler**
- Run a calibration workload through baseline placement, collect (token, layer, expert) trace
- Tag prefill vs decode
- Reuse existing `src/probes/hooks.py` (already written!)
- Output: parquet trace identical schema to `outputs/traces/`

**M3 — Placement table format**
- Per-layer dict / tensor: `placement[L][expert_id] = rank`
- Per-deployment cached. Optional: per-phase variant.

**M4 — Optimizer core**
- Input: trace + topology cost matrix + balance constraint + (optional) replication map
- Output: placement table
- Algorithm: greedy heuristic for fast first cut; LP relaxation if quality matters
- Variant: k=2 joint optimizer as Plan D ablation
- Reuse + extend `analysis/placement_simulator.py` (already written)

**M5 — Replication-aware constraint**
- Take a per-expert replication factor as input (from external module or static config)
- Modify M4's balance constraint to account for replicated experts
- Smallest viable: just respect "expert X is on ranks {a, b, c}" as a hard constraint

**M6 — Topology cost model**
- Probe cluster: NVLink topology, NUMA, network bandwidth (from `nvidia-smi nvlink` etc.)
- Cost matrix: cost[r1][r2] = expected latency / bandwidth penalty
- M4 uses this in the objective

**M7 — Drift detector (Plan B optional)**
- Per-layer per-rank routing histogram in a sliding window
- KL divergence vs calibration baseline
- Threshold-based trigger; conservative

---

## 5. Critical experiments to add (beyond what we have)

Given the new framing, our current results are insufficient. New must-runs:

| # | Experiment | Effort | Gating |
|---|---|---|---|
| **E1** | **Verify P1/P2/P3** — read COMET, MegaScale-Infer, Pre-gated MoE source/papers | 4-8 hours | Blocks all coding |
| E2 | Confirm pipelined dispatch model matches what COMET actually does (instrumentation or paper read) | 2 hours | Blocks contribution claim |
| E3 | Per-phase profile: prefill vs decode routing on the existing traces (C5) | 2 hours | Determines per-phase optimizer need |
| E4 | Workload drift: how stable is cross-layer MI across (model, domain) — partially done in PLAN.md §2.4, extend to a continuous time-series within one workload | 4 hours | Determines if Plan B is needed |
| E5 | Topology-weighted cross-rank rate using a realistic NVLink + IB cost matrix | 4-8 hours | Realistic X — Plan C headline number |
| **E6** | **End-to-end latency on 2× P100** with random vs. our placement on a small MoE (e.g., OLMoE-1B) running through Tutel or a minimal pipelined runtime | 1-2 days | Smoke test for the X claim — **single most important new experiment** |
| E7 | End-to-end on 2× nodes (any cloud) for the headline result | 3-5 days | Headline paper number |
| E8 | Greedy k=1 vs k=2 ablation (Plan D) | 1 day | Justifies k=1 default |

**E6 is the linchpin.** It's the cheapest experiment that converts the simulator number into a real-system number. Two P100s on one box can run a small MoE end-to-end. If E6 confirms a ≥10% latency improvement, the paper has a path. If E6 shows the simulator overestimates by 5-10×, the paper needs to retreat to "simulator quantifies upper bound" framing (and reviewers will be skeptical).

---

## 6. What was *not* in the brainstorm scope

- **Training-time placement** (placement changes during training) — that's a separate paper; inference-only first.
- **Router-aware placement** (re-training the router to be placement-friendly) — explicitly ruled out by PLAN.md §1.3 ("don't fight training objectives"). We exploit the existing router; we don't modify it.
- **MoE architectures with non-balanced load** (e.g., Mixtral-instruct with expert sparsity) — out of scope, future work.
- **Replication strategy itself** — we *compose* with existing replication, don't redesign it (C2 option c).

---

## 7. Three-stage execution plan (proposed timeline)

### Stage 1 — Verification + simulator extension (Week 1)
- E1: Verify premises (1-2 days)
- E2: Confirm pipelined dispatch model (½ day)
- E3, E4, E5: extend existing simulator with phase split, drift profile, topology weighting (2-3 days)
- **Gate**: if E1 invalidates premises, pivot back to measurement-paper framing immediately.

### Stage 2 — Runtime integration + single-node E2E (Week 2-3)
- M1: COMET/Tutel placement hook patch (3-5 days)
- M2: trace profiler (1 day, reuses existing hooks)
- E6: 2× P100 end-to-end smoke test (2-3 days)
- **Gate**: if E6 shows < 5% latency improvement on single node, reconsider — maybe simulator overestimates by 10×, and we're chasing a small number.

### Stage 3 — Multi-node validation + paper writing (Week 4-6)
- E7: cloud multi-node (3-5 days)
- E8: k=2 ablation (1 day)
- Paper draft (parallel with experiments)
- **Gate**: target submission window MLSys 2027 or NSDI 2027.

**Total**: 6 weeks if everything goes well; 8-10 weeks realistic with debugging.

---

## 8. Lit-verification queue (open the source code, not just the abstract)

Required reading before Stage 1, in order:

1. **COMET** (ASPLOS'25, "Fine-grained Computation-Communication Overlapping for MoE", arXiv 2502.19811)
   - Look at: expert placement code path, what is the placement-decision interface
   - Need to know: is placement compile-time or runtime configurable?
2. **MegaScale-Infer** (arXiv 2504.02263)
   - Look at: §3 prefill/decode architecture, §4 ping-pong execution
   - Need to know: is expert placement a separate phase or interleaved with execution?
3. **Pre-gated MoE** (ISCA'24)
   - Need to know: do they place experts based on gate predictions? If yes, this is the closest prior art and we MUST cite + differentiate.
4. **Lina** (ATC'23, "Accelerating Distributed MoE Training and Inference with Lina")
   - Need to know: their replication policy. We'll cite as the orthogonal direction we compose with.
5. **Occult** (ICML'25, "Communication-Efficient MoE via ...")
   - Need to know: per-layer optimization scope; how it differs from our cross-layer scope.
6. **NetMoE** (ICLR'25)
   - Need to know: token-level placement; orthogonal to ours but adjacent.
7. **SmartMoE / Tutel / DeepSpeed-MoE** — quick scan for placement defaults.

**This list maps directly to the related-work section of the paper.**

---

## 9. Open questions for human decision

1. **Runtime choice**: COMET or MegaScale-Infer as primary integration target? COMET has a cleaner public codebase last time I checked; MegaScale-Infer's pipelined model matches our metric more naturally. Pick one and stick with it; mention the other as "applies similarly".
2. **Cluster access for E7**: do you have a 2-node A100 cluster, or do we need to budget ~$500-2000 for cloud time?
3. **Naming**: "RouteWeaver" was the three-timescale framing. Drop it? Suggestions: CLPlace, XLPlace, Trajectree, Cohere. Or punt to writing phase.
4. **Backup paper**: if E1 invalidates the premise, do we retreat to measurement-paper framing per `outputs/H1_DECISION.md`? Yes / no / different fallback.
5. **Venue**: MLSys 2027 (March deadline) vs NSDI 2027 (May deadline). 6-8 week timeline fits both; NSDI is harder bar for measurement-light system papers.

---

## 10. TL;DR for the next decision

**Strongest 6-week plan**:
- Verify the 3 premises (1 week)
- Build M1 + M2 + M4 + M6 (placement layer + topology-aware optimizer), patch into the chosen runtime (2 weeks)
- E6 single-node end-to-end, E7 multi-node headline (2 weeks)
- Paper draft + ablations (1-2 weeks parallel)

**Headline contribution if everything works**:
> "Existing pipelined MoE runtimes (COMET, MegaScale-Infer) solve compute-communication overlap but inherit layer-independent placement from upstream. We add a topology-aware cross-layer placement layer that, on the existing pipelined dispatch substrate, reduces inter-node dispatch latency by X% on Qwen-MoE / DeepSeek-MoE / OLMoE-MoE, by exploiting the cross-layer routing correlation we measure in §3."

**Single most important next action**: spend 1 hour reading COMET's placement code path (P1 verification) before any new line of system code is written.
