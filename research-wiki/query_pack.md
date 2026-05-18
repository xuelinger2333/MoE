# Query Pack — MoE direction snapshot
Generated 2026-05-15T13:40:30Z. Budget 8000 chars.

## Project direction
MoE AllToAll communication. Original RouteWeaver pitch (within-layer skew exploitation)
DIED. New live direction: **cross-layer routing predictability** — strongly supported
by exp:m6, opens a measurement-paper + prefetch/placement/compression-method roadmap.

## Top gaps (8, ordered by current priority)
- **G6 [LIVE, supported on 3 models × 3 domains] — Cross-layer joint distribution unconstrained.**
  exp:m6 + m7 + m8 jointly confirm: every model × domain cell > 0.3-nat strong threshold.
  Pre-gated MoE exploits this signal as a method but never measures it. Now covered:
  3 models, 3 domains, 4 distances, strict null. Best near-term direction (measurement paper).
- **G7 [open, narrowing]** — Why does aux-loss-free balancing reduce cross-layer MI ~2× vs aux-loss?
  Strengthened by m8: DeepSeek also has the smallest *domain* spread (0.14 vs 0.39/0.57 nat),
  hinting at a single mechanism that flattens both. Mechanism unknown.
- **G8 [partially resolved by m9]** — Why is code consistently the lowest-MI domain?
  Disambiguated: BOTH G8a (OOD/confidence) AND G8b (structural) contribute. Code router
  entropy is +0.06 to +0.20 nat above NL (G8a). Filtered-MI test shows OOD explains
  60% of gap on Qwen, 36% on DeepSeek, only 15% on OLMoE. OLMoE has the largest STRUCTURAL
  residual (0.49 nat). **Revised 2026-05-15 (training-mix lookup)**: prior "code more
  in-distribution for OLMoE" hypothesis is FALSIFIED — OLMoE has the LOWEST published
  code share (~2.5%, OLMoE paper Table 2; Qwen + DeepSeek-V2-Lite ratios undisclosed but
  V2 corpus is treated as "NL portion only" by Coder-V2). New explanation: saturated-router
  hypothesis — OLMoE's NL median entropy (3.812 nat) is highest of three, so OOD/confidence
  axis has no dynamic range left; structural difference must carry the gap.
- **G9 [LIVE, new]** — For aux-loss-balanced models with saturated routers, the
  structural code-vs-NL MI difference is independent of router confidence. Mechanism
  unknown — candidates: top-k spread (OLMoE k=8 vs Qwen 4 / DeepSeek 6), aux-loss
  flattening pressure, training-data quality vs quantity. Open — needs new idea.
- G2 — Router/locality/placement loosely coupled. **Partially killed**: expert-side
  has no headroom on converged checkpoints; sample-side (NetMoE) still open.
- G4 — Open-source MoEs too well load-balanced on converged checkpoints to expose
  expert-side locality. Methodology gap; addressed by routeweaver_early_train.
- G5 — Co-activation has real but scattered structure that hierarchical clustering
  misses; needs sparse methods (max-weight matching).
- G1 — Token semantics not first-class in collective/transport. Targeted by lossbound_ep.
- G3 — Portable expert-parallel primitive still platform-bound. Targeted by hybridep_collective.

## Paper clusters
- **Cross-layer / trajectory**: Pre-gated MoE (ISCA'24) — only paper exploiting the
  signal, no measurement. exp:m6 fills the measurement gap and creates 4 publishable
  angles (measurement paper, prefetch method, placement, compression).
- **Within-layer locality (mostly killed for converged ckpts)**: NetMoE (sample-placement,
  ICLR'25) survives; Occult (co-activation collocation, ICML'25) needs reformulation
  for sparse pairs; MoNTA (offline traffic-aware, arXiv'24) viable as baseline.
- **Transport / primitives**: OptiReduce (bounded-loss, NSDI'25), MegaScale-Infer (M2N,
  SIGCOMM'25), MegaBlocks (dropless, MLSys'23). Inspire lossbound_ep + hybridep_collective.

## Failed ideas — DO NOT REPROPOSE without addressing the failure mode
- **idea:routeweaver — DEAD 2026-05-15**. Both within-layer claims (top-20% pair share
  ≥50%, silhouette ≥0.20) failed cleanly on Qwen AND DeepSeek (cross-architecture).
  DO NOT propose any RouteWeaver variant whose first precondition is "modern MoE has
  skewed routing across experts at convergence". DO NOT propose hierarchical clustering
  on co-activation (silhouette < 0.20 universally). DO NOT assume per-domain shift
  produces measurable routing skew (3 open domains differ by < 1pp).
- **CRITICAL DISTINCTION**: cross-layer (idea:cross_layer_routing_predictability, ALIVE)
  is NOT the same as within-layer (idea:routeweaver, DEAD). Don't conflate.

## Successful ideas
- **idea:cross_layer_routing_predictability — ALIVE 2026-05-15**.
  Round 1: MI(L,L+1) − null = 1.534 nat Qwen / 0.686 DeepSeek; persists to d=8.
  Round 2 (strict null + per-source + OLMoE triangulation): null is per-token, not
  topic-level; 92-100% of sources reduce entropy ≥0.5 nat; balancing scheme dominates
  over top-k.
  Round 3 (3 models × 3 domains, exp:m8 + claim:C4): every cell > strong threshold;
  code consistently lowest; DeepSeek smallest domain spread. Mature enough for a
  measurement paper with 4-axis evidence (architecture, domain, distance, source-expert).

## Top papers
- paper:hwang2024_pregated_moe — Pre-gated MoE (ISCA'24). Single most important paper for
  the new direction; uses cross-layer signal as a method but doesn't characterize it.
- paper:dai2024_deepseek_v3 — auxiliary-loss-free balancing → CAUSAL explanation for
  why within-layer C1 failed.
- paper:liu2025_netmoe — sample placement (ICLR'25). Best surviving within-layer angle.
- paper:luo2025_occult — within-layer co-activation (ICML'25). C2 inspiration; sparse
  pair extraction (gap:G5) might still work.
- paper:warraich2025_optireduce — bounded-loss AllReduce (NSDI'25). Inspires lossbound_ep.
- paper:gale2023_megablocks — dropless (MLSys'23). Anti-pattern anchor for any drop scheme.
- paper:zhu2025_megascale_infer — M2N library (SIGCOMM'25). Inspires hybridep_collective.
- paper:guo2024_monta — offline network-aware (arXiv'24). Possible baseline for survivors.

## Active relationship chains
- DeepSeek-V3 loss-free balancing → uniform routing (exp:m1) → kills within-layer
  RouteWeaver (idea:routeweaver) → opens cross-layer angle (idea:cross_layer_routing_predictability,
  via exp:m6) — same data, different question.
- Pre-gated MoE (method paper) → opens phenomenon question (idea:cross_layer_routing_predictability)
  → exp:m6 quantifies it → next step is a measurement paper across architectures + a
  prefetch-as-system-paper.

## Open unknowns (priority for next round)
- Does cross-layer MI scale with model size (Mixtral 8×7B, OLMoE, Qwen3-MoE)? **HIGH**
- How does cross-layer MI evolve over training? Strong hypothesis: it grows with training
  (specialization deepens), and may saturate. **HIGH**
- Why do Qwen (1.534 nat) and DeepSeek (0.686 nat) differ so much? Is it top-k (4 vs 6),
  the balancing scheme, or model depth? **MEDIUM**
- Can we exploit the strongest paths (where conditional KL > 2 nat) for placement
  on a single rank, even when within-layer skew is uniform? **MEDIUM**
- Does the structure persist on multi-domain data (probe with M2-style mixed corpus)? **LOW** (cheap)
