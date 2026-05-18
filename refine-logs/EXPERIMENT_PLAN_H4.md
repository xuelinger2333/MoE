# H4 Cheap Test — Experiment Plan

**Idea**: `idea:multi_turn_expert_stickiness` (H4) — Within-conversation expert overlap > between-conversation overlap, exploitable as session-affinity expert routing.

**Status**: Pre-commitment cheap test, ~5-6h total (4h compute + 1-2h analysis writeup).

**Decision gate** (all three must pass for "strong finding" verdict):
1. `J_multi_turn / J_shared_prefix ≥ 0.4` — H4 signal is at least 40% as strong as the published shared-prefix signal
2. `J_within_conv / J_between_conv ≥ 2.5×` — Conversation is a meaningful unit of locality
3. `session_ID_hit_rate ≥ EAMC_hit_rate × 0.9` — Session-ID is a competitive cheap signal vs MoE-Infinity's content matching

If any one fails, decision tree branches (see §4 Decision Criteria).

---

## 1. Inputs needed (NOT yet on disk)

| Input | Source | Size | Status |
|---|---|---|---|
| Multi-turn traces | ShareGPT v3 or WildChat, ≥5-turn convs × 3 MoE models | ~50-100 convs × 5 turns × ~256 tokens = ~2M tokens/model | **NOT collected** |
| Shared-prefix traces | Self-instructed prefixes + N=8 parallel sampled completions × 3 MoE models | ~50 prefixes × 8 branches × ~256 tokens = ~1M tokens/model | **NOT collected** |
| Existing m8_* traces | NL/code/math single-turn | Already on disk | Reused for the "no-session, single-turn" between-conv null |

**Forward-pass cost**: ~4-6 GPU-hours on 1× A100 80GB or ~8-10 hours on 2× P100 16GB (Qwen + OLMoE only; DeepSeek-V2-Lite needs >16GB → skip on P100 or quantize to int8).

## 2. Models

| Model | Status on 2× P100 16GB | Plan |
|---|---|---|
| Qwen1.5-MoE-A2.7B (60 experts, top-4) | ✅ fits in 1× P100 bf16 (~14 GB) | Primary |
| OLMoE-1B-7B (64 experts, top-8) | ✅ fits in 1× P100 bf16 (~7 GB) | Primary |
| DeepSeek-V2-Lite (64 experts, top-6) | ⚠️ tight, may OOM at ~16 GB | **Decision needed**: skip / int8 quantize / use Cloudlab |

Recommendation: start cheap test with Qwen + OLMoE on 2× P100; reserve DeepSeek for follow-up after the cheap test confirms the phenomenon.

## 3. Milestones

### M0 — Sanity (15 min, no GPU)
- Run analysis modules on **synthetic traces** with hand-crafted properties:
  - Synthetic A: 100% within-conversation overlap → expect J_within = 1.0, J_between = expected random
  - Synthetic B: 0% within-conversation overlap (turns are independent) → expect J_within ≈ J_between
  - Synthetic C: Mixture (50% conversations have high within-overlap, 50% don't) → expect J_within > J_between but less than A
- **Pass criterion**: numerical values match analytic expectations within 0.02
- **Files exercised**: `analysis/jaccard_overlap.py`, `analysis/eamc_baseline.py`, `analysis/affinity_evaluation.py`

### M1 — Multi-turn trace collection (3-4h compute, 2× P100)
- **Dataset**: ShareGPT v3 (or WildChat fallback if licensing concerns) — 200 conversations with ≥5 turns/conversation
- **Models**: Qwen1.5-MoE-A2.7B + OLMoE-1B-7B
- **Protocol** (each conversation):
  - Concatenate turns with conversation template; mark `(conversation_id, turn_id, token_idx_in_turn)`
  - Forward pass; collect routing trace
- **Output**: `outputs/traces/h4_multiturn_{qwen,olmoe}/` with extended schema (conversation_id, turn_id added)
- **Pass criterion**: ≥150 conversations × 5 turns each = ≥750 turn-traces per model

### M2 — Shared-prefix trace collection (1-2h compute, 2× P100)
- **Reproduces**: arXiv 2604.17182 setting
- **Prefixes**: 50 prompts of ~100 tokens each (code-gen + general dialogue mix)
- **Per prefix**: 8 parallel sampled completions, T=0.7, top_p=0.9, max_new=256
- **Models**: same Qwen + OLMoE
- **Output**: `outputs/traces/h4_shared_prefix_{qwen,olmoe}/` with extended schema (prefix_id, branch_id added)
- **Pass criterion**: 50 prefixes × 8 branches per model = 400 branch-traces/model

### M3 — H4 analysis pipeline (1h, no GPU)
- **3 settings × 2 models = 6 main results**:
  1. Within-conversation Jaccard (multi-turn trace)
  2. Between-conversation Jaccard (multi-turn trace, randomly paired)
  3. Within-prefix Jaccard (shared-prefix trace)
  4. Between-prefix Jaccard (shared-prefix trace, randomly paired)
- **EAMC reproduction** (multi-turn trace): for each turn, EAMC matching predicts expert set → compute prefetch hit rate. Compare to:
  - Session-ID affinity (cache experts active in last turn of same conv) hit rate
  - Random baseline hit rate
- **Outputs**:
  - `outputs/h4/jaccard_summary.json` — all J values
  - `outputs/h4/affinity_summary.json` — hit rates
  - `outputs/h4/F_h4_main.png` — within vs between vs prefix bar plot
  - `outputs/h4/F_h4_affinity.png` — hit rate comparison
  - `outputs/h4/H4_VERDICT.md` — explicit go/no-go memo

## 4. Decision criteria (verbatim from user, with branch logic)

**Strong finding** (paper viable):
- J_multi_turn / J_shared_prefix ≥ 0.4
- AND J_within_conv / J_between_conv ≥ 2.5×
- AND session_ID_hit_rate ≥ EAMC_hit_rate × 0.9

**Partial finding** (1 of 3 fails):
- Only criterion-1 fails (signal exists but weaker than shared-prefix) — write as smaller observation; system claim weakens
- Only criterion-2 fails (Jaccard exists but cross-conv is comparable) — H4 dead, conversation isn't the right unit
- Only criterion-3 fails (session-ID is significantly worse than EAMC) — H4 becomes a "cheap trade-off" not a novel direction

**Dead** (≥2 fail): kill in research-wiki with same format as `idea:pipelined_runtime_placement`.

## 5. Risk mitigation

| Risk | Mitigation |
|---|---|
| ShareGPT licensing | Fallback to WildChat or UltraChat. UltraChat is synthetic → may overstate signal — flag explicitly in writeup. |
| Short conversations (<5 turns) → weak signal | Filter dataset to ≥5-turn convs; report avg turn count per dataset. |
| DeepSeek OOM on P100 | Skip for cheap test; if signal confirms, retry with int8 or Cloudlab. |
| EAMC reproduction subtleties (K-Means seed, cluster count) | Sweep K ∈ {50, 100, 200, 500} per MoE-Infinity paper; report sensitivity. |
| Within-conv overlap dominated by **system prompt** tokens | Compute Jaccard on **assistant turn tokens only**, EXCLUDING system + user. Report both variants for diagnostic. |

## 6. Budget tracking

| Milestone | Est. compute | Est. wall-clock | Cumulative |
|---|---|---|---|
| M0 sanity | 0 GPU | 15 min | 15 min |
| M1 multi-turn (Qwen + OLMoE) | ~4 GPU-h | 3-4 h (2× P100 parallel) | 4 h |
| M2 shared-prefix | ~2 GPU-h | 1-2 h | 5-6 h |
| M3 analysis | 0 GPU | 1 h | 6-7 h |

Total: **~6-7 hours**, **~6 GPU-hours on P100** (or 3 GPU-hours on A100 80GB).

## 7. Out of scope for cheap test (deferred)

- DeepSeek-V2-Lite measurement (do after cheap test confirms)
- Full WildChat ablation (do if ShareGPT signal looks promising)
- End-to-end serving simulation
- Multi-tenant traffic mix (post-paper)
- Adversarial conversations (e.g., topic shifts mid-conv) — useful for paper §5 robustness

## 8. Pre-flight (BEFORE GPU run)

User decisions needed:
1. ShareGPT vs WildChat vs UltraChat as primary dataset
2. DeepSeek on P100 (skip / int8 / Cloudlab)
3. Are 2× P100 actually available, or do we need to budget Cloudlab time?
4. Approve the success criteria thresholds (J_ratio ≥ 0.4, within/between ≥ 2.5×, hit-rate ≥ 0.9×)?
