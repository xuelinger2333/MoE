# Wiki Log

Append-only audit trail.

| Time (UTC)            | Event |
|-----------------------|-------|
| 2026-05-15T09:24:00Z  | Wiki initialized (manually, helper not installed) |
| 2026-05-15T09:24:30Z  | Ingested 5 anchor papers: NetMoE, Occult, MoNTA, MegaBlocks, DeepSeek-V3 |
| 2026-05-15T09:25:00Z  | Created idea routeweaver — initial proposed |
| 2026-05-15T09:25:30Z  | Recorded experiments exp_m1, exp_m2, exp_m4, exp_m5 (DeepSeek + Qwen probes) |
| 2026-05-15T09:26:00Z  | Created claims C1 (traffic concentration) and C2 (coactivation cluster) — both invalidated by exp_m1 + exp_m4 |
| 2026-05-15T09:26:30Z  | Marked idea routeweaver outcome=negative; status=DEAD; recorded evidence + post-mortem |
| 2026-05-15T09:27:00Z  | Added 3 follow-up ideas: routeweaver_early_train, lossbound_ep, hybridep_collective (all stage=proposed) |
| 2026-05-15T10:15:00Z  | Tested cross-layer routing MI hypothesis on existing M1+M4 traces (offline, 8s) → **STRONG support**: Qwen MI−null 1.534 nat, DeepSeek 0.686 nat, both ≫ 0.3-nat threshold |
| 2026-05-15T10:15:30Z  | New idea cross_layer_routing_predictability (status=ALIVE), new claim C3 (status=supported), new experiment exp:m6_cross_layer_mi, new gap G6, ingested paper:hwang2024_pregated_moe |
| 2026-05-15T11:08:00Z  | Round-2 robustness: added within-sequence shuffle null (strict), per-source entropy reduction histogram, OLMoE triangulation. STRICT null gives ≤0.03 nat for all 3 models → sequence-level locality contributes ~0%. Per-source: 92-100% of (L, source) pairs reduce entropy by ≥0.5 nat. |
| 2026-05-15T11:10:00Z  | Triangulated with OLMoE-1B-7B (top-8 aux-loss): MI-null=1.212 nat, much closer to Qwen than DeepSeek. Conclusion: balancing scheme dominates over top-k. New gap G7: why does aux-loss-free reduce cross-layer MI? |
| 2026-05-15T11:11:00Z  | Added exp:m7_olmoe_wikitext, paper:muennighoff2024_olmoe; updated claim C3, idea cross_layer_routing_predictability, gap_map. |
| 2026-05-15T11:30:00Z  | Round-3 multi-domain stability test: ran Qwen + OLMoE + DeepSeek × {code (codeparrot), math (gsm8k), nl (wikitext)} = 9 cells. **H1a (domain-invariance) rejected** but every cell still > 0.3 nat strong threshold. **Code is lowest-MI domain in all 3 models.** DeepSeek shows smallest domain spread (0.14 nat) consistent with G7 flattening hypothesis. |
| 2026-05-15T11:32:00Z  | Added exp:m8_multidomain_mi, claim:C4 (supported), gap:G8 (why code?). Updated wiki. |
| 2026-05-15T11:55:00Z  | Round-4 disambiguation: ran 9 entropy probes (full-softmax router entropy per token per layer) + filtered-MI test. exp:m9. **G8 partially resolved**: code IS slightly higher entropy in all 3 models (G8a partially supported), but the residual structural gap persists especially on OLMoE (G8b). Mixed verdict, model-dependent dominance. |
| 2026-05-15T11:56:00Z  | Added exp:m9_router_entropy, idea:confidence_aware_prefetcher (proposed pivot from G8 verdict). Updated G8, edges, index. |
| 2026-05-15T13:40:00Z  | Round-4 follow-up: looked up published training-data code shares. **OLMoE = ~2.5% code** (StarCoder 101B / 4060B; OLMoE paper Table 2; explicitly *reduced* from OLMo 1.7's 15.4%). Qwen1.5-MoE-A2.7B and DeepSeek-V2-Lite do not publish exact code ratios (DeepSeek-V2 corpus is described as "natural language portion only" by Coder-V2 paper). Falsifies prior wiki claim that OLMoE's structural residual is due to "code being more in-distribution for OLMoE training mix" — OLMoE has the LOWEST published code share. Replaced with saturated-router hypothesis (OLMoE NL median entropy 3.812 nat vs Qwen 3.583 / DeepSeek 3.679 → OOD/confidence axis is already saturated → structural residual must carry the code-vs-NL difference). |
| 2026-05-15T13:40:30Z  | New gap G9 added (aux-loss saturated-router structural residual mechanism unknown). Revised exp:m9, idea:confidence_aware_prefetcher, gap_map. Pulled missing remote data from Cloudlab before node shutdown: 6 m8 trace dirs (~833 MB), 9 router_entropy npz (~129 MB), m0_olmoe — all present locally now. |
| 2026-05-18T15:00:00Z  | Exp 1 + Exp 2 completed: permutation null retains 98-99% of observed MI in 9/9 cells; placement simulator shows 41-86% incremental cross-rank reduction (PL metric) over per-layer-optimal baseline. See `outputs/H1_DECISION.md`. |
| 2026-05-18T15:15:00Z  | Pivot brainstorm: `idea-stage/IDEA_REPORT_v2_pipelined_placement.md` — convert H1 measurement into a system-paper grafting on COMET/MegaScale-Infer. Flagged 3 premises requiring lit verification before code. |
| 2026-05-18T15:25:00Z  | Lit lookup (MoETuner + ExFlow + topology follow-up): cross-layer-aware placement already published. ExFlow Jan 2024 (arXiv 2401.08383) — IP-based cross-layer placement, up to 67% latency reduction. MoETuner Feb 2025 (arXiv 2502.06643) — two-stage ILP, 9.3% / 17.5% Mixtral speedup. Cluster topology Aug 2025 (arXiv 2508.09229) — adds datacenter topology cost. Killed `idea:pipelined_runtime_placement` BEFORE any new code was written. |
| 2026-05-18T15:30:00Z  | Added paper:wei2024_exflow, paper:go2025_moetuner, paper:cluster_topology_placement_2025. Created idea:pipelined_runtime_placement with status=DEAD; recorded post-mortem + invalidation edges. Added gap:G10 (cross-model placement-gain audit — what we still potentially have). |
