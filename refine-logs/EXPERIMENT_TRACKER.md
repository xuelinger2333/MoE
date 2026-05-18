# Experiment Tracker

| Run ID | Milestone | Setup | Status | Key result | Trace file | Notes |
|---|---|---|---|---|---|---|
| M0 | Sanity (DeepSeek) | DeepSeek-V2-Lite, bs=2, seq=64, 1 step | DONE | 26 hooks registered, 19,968 rows, all asserts pass | `outputs/traces/m0_smoke/` | — |
| M1 | Single-domain DeepSeek | WikiText-103 val, bs=4, seq=1024, 50 batches | DONE | 31.9M routing events; **C1 fail** top-20%=22.7%; **C2 fail** silhouette=0.169 | `outputs/traces/m1_deepseek_wikitext/` | Run in 29.2s |
| M2 | Multi-domain DeepSeek | 3 datasets × 15 batches (wikitext / c4 / mmlu) | DONE | per-domain top-20% all ~22%; CDFs overlap | `outputs/traces/m2_deepseek_multidomain/` | `stack` dropped (HF-gated); MMLU exhausted at 7 batches (small dev split). Run in 27s. |
| M3 | DeepSeek analysis | Offline | DONE | F1–F6 + claim_summary.json | `outputs/figures/` | C1 & C2 both fail clearly |
| M4 | Qwen1.5-MoE backup | WikiText, bs=4, seq=1024, 50 batches | DONE | 19.7M routing events; **C1 fail** top-20%=21.5%; **C2 fail** silhouette=0.186 | `outputs/traces/m4_qwen_wikitext/` | Promoted from NICE-TO-HAVE → MUST-RUN after DeepSeek failed. Run in 32.5s. |
| M5 | Cross-model comparison | DeepSeek vs Qwen, offline | DONE | F7 (heatmaps), F8 (CDF), `model_compare.json` | `outputs/figures/F7*` `outputs/figures/F8*` | Both models match the uniform reference within ~2 pp |

## Status legend
- PENDING — not yet started
- RUNNING — submitted / in flight
- DONE — completed and verified
- FAILED — completed but failed verification
- BLOCKED — cannot proceed (note blocker)

## Headline outcome

Both empirical claims (C1 traffic concentration, C2 co-activation cluster structure) **failed cleanly on both models**. RouteWeaver as originally specified does not have measurable headroom on modern open-source MoE checkpoints. See `EXPERIMENT_RESULTS.md` for interpretation and recommended pivots.
