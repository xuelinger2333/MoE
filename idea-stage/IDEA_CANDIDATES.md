# Idea Candidates — Pipelined-Runtime Placement Layer Pivot

**Pivot from**: `idea-stage/IDEA_REPORT.md` (RouteWeaver three-timescale runtime)
**To**: cross-layer placement layer plugged into existing pipelined MoE runtimes
**Date**: 2026-05-18
**Full brainstorm**: `idea-stage/IDEA_REPORT_v2_pipelined_placement.md`

## Active plan: A + C composed, B and D as ablations

| # | Plan | Scope | Estimated X | Status |
|---|------|-------|------------|--------|
| A | Static-only (offline placement) | per-layer offline solve | 30-50% over baseline | Baseline; ship if alone works |
| **A+C** | **+ topology-aware cost model** | weighted QAP with NVLink/IB tiers | **40-60% realistic** | **RECOMMENDED HEADLINE** |
| B | + drift detector | semi-dynamic re-solve | A + 5-15% recovery | Ablation row |
| D | + k=2 joint trajectory | bilinear assignment | A + 5-15% | Ablation row |

## Top challenges, with mitigation modules

| # | Challenge | Severity | Module |
|---|---|---|---|
| C1 | COMET/MegaScale-Infer may not expose placement API | HIGH | M1 runtime hook (verify first) |
| C2 | Conflict with expert replication for hot spots | HIGH | M5 replication-aware constraint |
| C7 | End-to-end latency needs real cluster | HIGH | M3 benchmark harness, E6/E7 experiments |
| C8 | Pipelined dispatch is non-standard for some runtimes | VARIABLE | premise-verification, gating |
| C3 | Load-balance constraint vs trajectory preserving | MEDIUM | balance constraint in optimizer |
| C4 | Cold-start (need trace before optimizing) | MEDIUM | M2 trace profiler (reuses existing hooks) |
| C5 | Decode vs prefill routing divergence | MEDIUM | per-phase optimizer or robust placement |
| C6 | Multi-tenant traffic skew | LOW-MED | covered by drift detector |

## Gating action before any code

**E1: read COMET + MegaScale-Infer source/paper, verify they use random/round-robin placement** (4-8 h).
If false → pivot back to `outputs/H1_DECISION.md` measurement-heavy paper.
If true → proceed to Stage 1.

## Linchpin experiment

**E6: single-node 2×P100 end-to-end latency with random vs. our placement on OLMoE-1B-MoE via Tutel/COMET.**
- Cheapest path from simulator → real-system claim
- Confirms simulator predicts real latency (or doesn't — fail-fast signal)
- If E6 shows ≥10% latency win, paper has a clear road
- 2-3 days of effort, $0 cost (local hardware)

## Six-week timeline

| Week | Deliverable |
|---|---|
| 1 | E1-E5 verification + simulator extensions; gate decision |
| 2-3 | M1-M4 + M6; E6 single-node E2E |
| 4-5 | E7 multi-node headline; E8 k=2 ablation |
| 5-6 | Paper draft (parallel with E7) |

Target: MLSys 2027 (March) or NSDI 2027 (May).

## Open questions for human decision

1. COMET vs MegaScale-Infer as primary integration target?
2. Cloud budget for multi-node E7 ($500-2000) or in-house cluster access?
3. Name: keep "RouteWeaver" or rebrand (CLPlace / XLPlace / Trajectree / Cohere)?
4. Backup paper plan if E1 invalidates premises?
5. Target venue: MLSys vs NSDI?
