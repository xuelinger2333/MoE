# Research Wiki Index

## Direction
MoE AllToAll communication research. Original RouteWeaver direction (within-layer
expert load skew + co-activation clustering) is **DEAD**; a new live direction
opened on 2026-05-15: **cross-layer routing predictability** as a measurable
phenomenon and exploitable system signal. A 2026-05-18 attempt to pivot that
finding into a system-paper grafting on COMET/MegaScale-Infer is **DEAD on
prior-art** (ExFlow Jan 2024, MoETuner Feb 2025, topology follow-up Aug 2025).
Surviving angle: measurement-heavy paper using ExFlow/MoETuner as comparators
to characterize cross-model generalization (new gap G10).

## Stats

```
📚 Research Wiki Stats
Papers: 12 (9 core, 3 related)
Ideas:  7  (2 DEAD, 1 ALIVE, 4 candidates)
Experiments: 9 (9 completed) — Exp 1/2 (permutation null + placement simulator) are post-hoc analyses on existing traces, not new exp: nodes
Claims: 4  (2 supported, 2 invalidated)
Edges:  59
Gaps:   10 (5 unresolved/active, 1 partially-killed, 1 newly-supported, 2 partially-resolved, 1 new from kill)
Last updated: 2026-05-18T15:30:00Z
```

## Ideas

| Node | Title | Stage | Outcome |
|---|---|---|---|
| [idea:cross_layer_routing_predictability](ideas/cross_layer_routing_predictability.md) | Cross-layer routing predictability — phenomenon-level measurement | tested | ✅ **ALIVE** |
| [idea:pipelined_runtime_placement](ideas/pipelined_runtime_placement.md) | Cross-layer placement plug-in for COMET / MegaScale-Infer | killed | ❌ **DEAD** (sniped by ExFlow + MoETuner + topology follow-up) |
| [idea:routeweaver](ideas/routeweaver.md) | Within-layer router/placement/topology joint runtime | tested | ❌ **DEAD** |
| [idea:routeweaver_early_train](ideas/routeweaver_early_train.md) | RouteWeaver scoped to pre-equilibrium checkpoints | proposed | candidate |
| [idea:lossbound_ep](ideas/lossbound_ep.md) | Quality-budget-driven semantic AllToAll | proposed | candidate |
| [idea:hybridep_collective](ideas/hybridep_collective.md) | Hierarchical exact/approx EP primitive | proposed | candidate |
| [idea:confidence_aware_prefetcher](ideas/confidence_aware_prefetcher.md) | Cross-layer expert prefetcher with confidence gate | proposed | candidate |

## Claims

| Node | Status | Last verdict source |
|---|---|---|
| [claim:C4](claims/C4_domain_stability.md) — cross-layer MI domain-dependent, but every cell ≫ null; code lowest | ✅ supported | exp:m8 (9-of-9 cells > 0.3 nat; code always lowest) |
| [claim:C3](claims/C3_cross_layer_mi.md) — cross-layer MI ≫ null | ✅ supported | exp:m6 (round-2: strict null + per-source); exp:m7 (triangulation) |
| [claim:C1](claims/C1_traffic_concentration.md) — within-layer traffic concentration | ❌ invalidated | exp:m1+m2+m4 |
| [claim:C2](claims/C2_coactivation_cluster.md) — co-activation clusters | ❌ invalidated | exp:m1+m4 |

## Experiments

| Node | Date | Verdict |
|---|---|---|
| [exp:m9_router_entropy](experiments/exp_m9_router_entropy.md) | 2026-05-15 | partially resolves G8 (mixed G8a OOD + G8b structural; OLMoE 85% structural) |
| [exp:m8_multidomain_mi](experiments/exp_m8_multidomain_mi.md) | 2026-05-15 | supports C4 (3 models × 3 domains: code always lowest, every cell > 0.3 nat) |
| [exp:m7_olmoe_wikitext](experiments/exp_m7_olmoe_wikitext.md) | 2026-05-15 | supports C3 (triangulation: aux-loss balancing → high MI) |
| [exp:m6_cross_layer_mi](experiments/exp_m6_cross_layer_mi.md) | 2026-05-15 | supports C3 (round-2: strict null + per-source universal) |
| [exp:m5_compare](experiments/exp_m5_compare.md) | 2026-05-15 | invalidates idea:routeweaver |
| [exp:m4_qwen_wikitext](experiments/exp_m4_qwen_wikitext.md) | 2026-05-15 | invalidates C1, C2 (cross-arch) |
| [exp:m2_deepseek_multidomain](experiments/exp_m2_deepseek_multidomain.md) | 2026-05-15 | invalidates C1 per-domain |
| [exp:m1_deepseek_wikitext](experiments/exp_m1_deepseek_wikitext.md) | 2026-05-15 | invalidates C1, C2 |

## Papers

| Node | Title | Year | Venue | Relevance |
|---|---|---|---|---|
| [paper:dai2024_deepseek_v3](papers/dai2024_deepseek_v3.md) | DeepSeek-V3 (auxiliary-loss-free balancing) | 2024 | arXiv | core (causal for C1 fail) |
| [paper:gale2023_megablocks](papers/gale2023_megablocks.md) | MegaBlocks (dropless MoE) | 2023 | MLSys | core |
| [paper:guo2024_monta](papers/guo2024_monta.md) | MoNTA (network-traffic-aware) | 2024 | arXiv | related |
| [paper:hwang2024_pregated_moe](papers/hwang2024_pregated_moe.md) | Pre-gated MoE (cross-layer prefetch) | 2024 | ISCA | core (only paper exploiting cross-layer signal) |
| [paper:liu2025_netmoe](papers/liu2025_netmoe.md) | NetMoE (dynamic sample placement) | 2025 | ICLR | core |
| [paper:luo2025_occult](papers/luo2025_occult.md) | Occult (within-layer collaborative) | 2025 | ICML | core |
| [paper:muennighoff2024_olmoe](papers/olmoe2024.md) | OLMoE (open MoE LLM) | 2024 | arXiv | related (triangulation model) |
| [paper:warraich2025_optireduce](papers/warraich2025_optireduce.md) | OptiReduce (bounded-loss AllReduce) | 2025 | NSDI | related |
| [paper:zhu2025_megascale_infer](papers/zhu2025_megascale_infer.md) | MegaScale-Infer (M2N library) | 2025 | SIGCOMM | related |
| [paper:wei2024_exflow](papers/wei2024_exflow.md) | ExFlow (inter-layer expert affinity) | 2024 | arXiv | core (kills idea:pipelined_runtime_placement) |
| [paper:go2025_moetuner](papers/go2025_moetuner.md) | MoETuner (two-stage ILP for placement) | 2025 | arXiv | core (kills idea:pipelined_runtime_placement) |
| [paper:cluster_topology_placement_2025](papers/cluster_topology_placement_2025.md) | Cluster-topology-driven expert placement | 2025 | arXiv | core (kills Plan C extension) |

## Gaps

See [gap_map.md](gap_map.md). G6 is the live measurement gap (cross-layer joint distribution unconstrained by training). G10 is the new gap from the 2026-05-18 kill: cross-model placement-gain audit — does ExFlow/MoETuner generalize beyond Mixtral?
