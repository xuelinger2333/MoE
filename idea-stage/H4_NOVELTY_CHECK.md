# Novelty Check Report — H4: Multi-turn conversation expert stickiness

**Date**: 2026-05-18
**Idea source**: user proposal (PLAN.md-style cheap test idea)
**Reviewer**: novelty-check skill via WebSearch (Codex cross-model review not run — see note below)

---

## Proposed idea (verbatim from user)

**Phenomenon**: In multi-turn conversations (same user session, multiple requests), the expert activation sets across turns of the same conversation have higher overlap than across different conversations.

**Why priori-alive**: Load balancing loss optimizes per-token marginal uniformity, not cross-request correlations. Conversation has semantic continuity (topic, language, user style invariant), which may cause expert reuse.

**Why not-dead**: Existing MoE systems work focuses on intra-request optimization. Multi-request / cross-session expert behavior has not been measured.

**Cheap test**: ShareGPT/WildChat/UltraChat, 3-5 turn conversations. Measure Jaccard(expert_set_turn_i, expert_set_turn_j) within vs between conversations. ~3h, mostly dataset processing.

**System pivot if alive**: Session-aware expert affinity scheduling — route same user's requests to same GPU group (which has their commonly-used experts cached). Structural analog of KV-cache session affinity (Mooncake, SGLang) applied to MoE expert weights.

---

## Core claims to verify

| # | Claim | Type |
|---|-------|------|
| C1 | Within-conversation expert Jaccard > between-conversation Jaccard, by a non-trivial margin | Measurement |
| C2 | This pattern is robust across MoE architectures (≥2 model families, ≥2 datasets) | Measurement |
| C3 | Session-affinity scheduling for MoE expert cache is structurally novel (no prior MoE-specific implementation) | System |
| C4 | The end-to-end gain from session-affinity placement justifies system complexity | System (out of scope for cheap test) |

---

## Closest prior work

| Paper | Year | Overlap type | What they did | What's NOT in their work (your gap) |
|---|---|---|---|---|
| **MoE-Infinity** ([arXiv 2401.14361](https://arxiv.org/abs/2401.14361)) | Jan 2024 | **PARTIAL OVERLAP — closest competitor** | Sequence-level expert activation tracing; Expert Activation Matrix Collection (EAMC) clustered via offline K-Means; online cosine-distance matching of new requests against EAMC for prefetching/caching | (a) Unit is "request" not "conversation/session"; (b) matching is by *content similarity*, never by *session ID*; (c) EAMC is global, not user-partitioned; (d) no measurement of within-session vs between-session overlap. |
| **arXiv 2604.17182** ("Layer-wise MoE Routing Locality under Shared-Prefix Code Generation") | 2026 (recent) | **PARTIAL OVERLAP — same methodology, different setting** | Measures Jaccard similarity of expert sets between branched generations from a shared prefix (parallel sampling / beam search). Reports J=0.649 at same-token positions (40× random) | (a) Their setting is *parallel branching* from one prefix, not *sequential turns* in a conversation; (b) measurement is per-token-position-aligned, not per-conversation-aggregated; (c) no system implication for cross-request serving — their goal is offloading-aware caching for parallel sampling |
| **DanceMoE / Prism** ([arXiv 2508.12851](https://arxiv.org/abs/2508.12851)) | Aug 2025 | **PARTIAL OVERLAP — server-level workload locality** | Per-server activation-frequency-aware expert placement; periodic 5-minute re-evaluation and migration. "Servers exhibit distinct patterns of expert activation due to varying user tasks and input distributions" | The "server" is a workload/geographic locality unit, not a *user/conversation* unit. No per-session routing decision; no measurement of within-conversation reuse. |
| **Rewiring Experts on the Fly** ([arXiv 2510.14853](https://arxiv.org/abs/2510.14853)) | Oct 2025 | **ADJACENT — multi-turn MoE but different goal** | Studies routing changes across multi-turn context shifts; proposes continuous rerouting to *adapt* routing weights | They study how routing *changes* across turns; we'd study how routing *stays similar* across turns. Orthogonal angles. They never report Jaccard overlap. |
| **MoE-Infinity insight** | Jan 2024 | **ADJACENT — temporal locality** | "Decoder activation exhibits strong temporal locality. An expert may be active for several consecutive batches, then go inactive again." | This is *within-request* decode temporal locality. We'd extend to *cross-request within-session* temporal locality. |
| **Read-ME** ([NeurIPS 2024](https://utns.cs.utexas.edu/assets/papers/neurips24-readme.pdf)) | 2024 | **POTENTIAL COUNTER-ARGUMENT** | "Expert caches are shared across multiple requests, making cache policies relying on per-request traits suboptimal. A global view across all requests is necessary for effective caching." | Read-ME's claim *could* contradict H4's premise: if global view dominates per-request, per-session might not add signal. But Read-ME is offline-batch-prefetching; doesn't address the streaming serving case where session-level information arrives free. |
| **SGLang RadixAttention** ([LMSYS](https://www.lmsys.org/blog/2024-01-17-sglang/)) | Jan 2024 | **ADJACENT — KV cache session analog** | Automatic KV cache reuse via radix tree of cached prefixes. 75-95% cache hit on multi-turn conversations | KV cache, not expert weights. SGLang explicitly exploits prefix-based session affinity. The "session affinity for MoE experts" is the structural analog they did NOT do (no MoE-specific component). |
| **Mooncake** ([arXiv 2407.00079](https://arxiv.org/abs/2407.00079)) | Jul 2024 | **ADJACENT — KV cache session routing** | KVCache-centric scheduling with session-affinity routing. Cloudflare reports 60% → 80% cache hit rate increase via `x-session-affinity` header | KV cache, not experts. Confirms the **principle** that session affinity is a real serving lever; doesn't address MoE expert-side. |
| **ExFlow** ([arXiv 2401.08383](https://arxiv.org/abs/2401.08383)) | Jan 2024 | **ALREADY IN OUR WIKI — orthogonal** | Inter-layer expert affinity ILP for *intra-request* placement | Doesn't touch cross-request / session. |
| **DuoServe-MoE, Pre-gated MoE, Read-ME, FineMoE, AdapMoE, SiDA-MoE, MoE-Beyond, PopFetcher, HOBBIT** | 2024-2026 | NO OVERLAP — single-request prefetch/cache | All within-request expert prefetch/cache strategies | None reason about cross-request session continuity. |

---

## Novelty assessment per claim

| Claim | Score | Closest prior | Notes |
|---|---|---|---|
| **C1**: within-conv Jaccard > between-conv | **7/10** | arXiv 2604.17182 (shared-prefix Jaccard) | Methodology is published; setting (multi-turn dialogue) and analysis unit (per-conversation) are unstudied. Honest position: "The shared-prefix paper does Jaccard on parallel branches; we do it on sequential turns — different temporal structure of the conversation graph." |
| **C2**: cross-architecture robustness | **8/10** | None | Nobody has run this measurement; can leverage existing 3-model trace stack (Qwen / DeepSeek / OLMoE) for free |
| **C3**: session-affinity MoE expert routing as a system | **6/10** | SGLang RadixAttention (for KV cache), MoE-Infinity (for global EAMC) | The KV-cache analog is well-established. "Apply X-for-KV-cache to MoE experts" is a reviewer's lazy-novelty-attack target. Mitigation: argue the MoE-specific implementation challenges differ from KV cache (per-layer routing decisions ≠ prefix matching). |
| **C4**: end-to-end gain | **7/10 (conditional)** | DanceMoE (server-level migration), Mooncake (KV session affinity gains) | Out of scope for the 3h cheap test. Real evaluation needs a serving stack. |

---

## Overall verdict

**Score: 6.5/10 — PROCEED WITH CAUTION**

**Key differentiator**: Measurement at the **conversation/session** level (not request, not server) is genuinely unstudied. The cheap test directly probes whether session is a meaningful unit of locality for MoE experts.

**Biggest risk**: The system contribution (session-affinity scheduling for MoE) is a structural analog of well-known KV-cache session affinity. A reviewer's first instinct: "trivial application of SGLang/Mooncake to MoE." Mitigation requires:
1. Showing the MoE expert affinity unit has *qualitatively different properties* from KV cache prefixes (e.g., expert sets are sparse and overlap probabilistically; KV cache is bit-exact and shared as prefixes). The implementation actually differs.
2. Showing the gain is real and *not subsumed* by what MoE-Infinity's EAMC already provides via content-similarity matching.

**The cheap test's actual decision-making value**:
- If within-conv J ≥ 0.4 and between-conv J ≤ 0.15 → STRONG signal, paper has legs. The 2-3× gap is the kind of number that survives reviewer skepticism.
- If within ≈ 0.25, between ≈ 0.15 → WEAK signal (1.5× gap). The phenomenon exists but is dominated by other locality forms (global popularity, layer affinity). System gain unlikely to clear the "trivial KV-analog" bar.
- If within ≈ between → DEAD. Conversation continuity doesn't drive cross-request expert reuse. Kill before any system work.

---

## Suggested positioning (if cheap test gives strong signal)

**Title direction**: "Conversation-level Expert Locality in MoE Serving"

**Pitch**:
> "MoE inference systems exploit three forms of locality: within-token (router decisions are sparse), within-request (decoder temporal locality, MoE-Infinity), and within-server (workload-level patterns, DanceMoE). A fourth — *within-conversation* — has been overlooked. We show that the expert sets activated by consecutive turns of the same conversation have Jaccard overlap of [X]× the cross-conversation baseline, across three MoE families. This locality is a free signal for serving systems: routing a user's session to the same GPU group amortizes expert weight migrations across the conversation's lifetime, analogously to how SGLang RadixAttention amortizes KV-cache prefill across turns."

**Required differentiation paragraphs**:
- vs. MoE-Infinity EAMC: "Session ID is a cheaper, more reliable signal than EAMC's cosine-distance content matching; no online K-Means; no calibration set needed."
- vs. SGLang/Mooncake: "KV cache reuses bit-exact prefixes; expert cache reuses statistically overlapping sets. Different scheduling and admission policies follow."
- vs. shared-prefix Jaccard paper (arXiv 2604.17182): "Their setting is parallel branches from one prefix; ours is sequential turns in a conversation. Different temporal structure, different system implications."

---

## Recommended next actions

1. ✅ **Run the 3-hour cheap test as proposed.** The decision tree above gives clear go/no-go thresholds.
2. 📋 **Use ShareGPT or WildChat as primary dataset.** UltraChat is mostly synthetic and may overstate the signal artificially. Prefer real human-user multi-turn conversations (≥5 turns/conversation average).
3. 📊 **Compute three Jaccard variants for diagnostic depth:**
   - per-layer (J at each MoE layer separately — likely differs across model depth per shared-prefix paper finding)
   - top-1 vs union-of-top-k (sparser unit gives sharper signal)
   - aggregated over conversation vs. between adjacent turns only
4. 🔬 **Include a calibration check**: compute Jaccard for randomly-paired turn pairs from the SAME conversation across all conversations — this is the within-domain-but-cross-conversation null, sharper than fully random pairing.
5. 📝 **If signal is positive**, the next gate before writing is a **Codex MCP cross-model verification** (skill convention) — run the full novelty check + measurement results through GPT-5.4 xhigh, ask for the strongest possible rejection memo.
6. 💀 **If signal is negative**, kill the idea in the research-wiki with the same kill record format as `idea:pipelined_runtime_placement`.

---

## What I didn't run (and why)

- **Codex MCP cross-model verification** — skipped because (a) the literature lookup is unambiguous on the key collision points (MoE-Infinity, shared-prefix Jaccard, DanceMoE), (b) the cheap test (3h) is itself the decisive evidence, and (c) the user prefers low-overhead checks at this stage. Recommend running Codex after the cheap test, before committing to a paper.
- **arXiv full-text reads of MoE-Infinity and arXiv 2604.17182** — the search-snippet content is detailed enough to characterize the gap. If the cheap test confirms the phenomenon, full reads become mandatory before paper-writing.

---

## Sources

- [MoE-Infinity: Activation-Aware Expert Offloading](https://arxiv.org/abs/2401.14361)
- [Layer-wise MoE Routing Locality under Shared-Prefix Code Generation (arXiv 2604.17182)](https://arxiv.org/html/2604.17182)
- [DanceMoE / Prism: Distributed MoE Inference in Edge Systems](https://arxiv.org/abs/2508.12851)
- [Rewiring Experts on the Fly](https://arxiv.org/abs/2510.14853)
- [Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts (NeurIPS 2024)](https://utns.cs.utexas.edu/assets/papers/neurips24-readme.pdf)
- [SGLang / RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
- [DuoServe-MoE: Dual-Phase Expert Prefetch and Caching](https://arxiv.org/abs/2509.07379)
- [ExFlow](https://arxiv.org/abs/2401.08383) — already in our research-wiki
