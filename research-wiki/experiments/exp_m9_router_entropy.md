---
type: experiment
node_id: exp:m9_router_entropy
title: "M9 — Router entropy histogram + filtered-MI test (G8 OOD disambiguation)"
status: completed
date: 2026-05-15T06:57:51Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
models: ["Qwen/Qwen1.5-MoE-A2.7B", "deepseek-ai/DeepSeek-V2-Lite", "allenai/OLMoE-1B-7B-0924"]
domains: ["nl (wikitext)", "code (codeparrot-clean-valid)", "math (gsm8k)"]
trace_dirs: ["outputs/router_entropy/<tag>_<domain>.npz × 9"]
n_tokens_per_run: 204800
elapsed_total_minutes: ~5
tested_gaps: ["gap:G8"]
verdicts:
  G8: partially_resolved
tags: ["MoE", "router-entropy", "OOD", "disambiguation"]
notes: "Each entropy probe captures one float per (token, layer) — the Shannon entropy of the FULL post-softmax distribution over all experts. For DeepSeek's MoEGate (which renormalizes top-k weights), the hook recomputes the full softmax from inputs[0] and module.weight."
---

# M9 — Router entropy + filtered-MI

## Question

G8 (from exp:m8): code is the universally lowest-MI domain across 3 architecturally
diverse MoE LLMs. Two competing explanations:

- **G8a (OOD artifact)**: code is OOD vs the predominantly-NL training mix → router
  less confident on code → P(expert | token) more spread out → top-1 less stable
  → cross-layer MI estimated from top-1 is lower.
- **G8b (structural)**: router is *equally* confident on code, but uses
  *different* expert combinations → the joint structure itself differs.

## Test

For each (model, domain) cell, compute per-token, per-layer router entropy
`H(P(e | token))` from the FULL post-softmax distribution over all experts
(not just top-k). Then:

1. **Histogram test (G8a direct).** Compare entropy distributions across
   domains. If code's distribution is right-shifted vs NL → router is less
   confident on code (G8a partially supported).
2. **Filtered-MI test (the killer).** Restrict to "confident" tokens
   (entropy < NL median) at BOTH layers L and L+1. Recompute MI on the
   filtered subset. If the code-vs-NL MI gap collapses → G8a fully explains.
   If gap persists → G8b structural component remains.

## Headline results

### Median entropy by (model, domain)  [in nats]

| Model | nl | code | math | code−nl |
|---|---|---|---|---|
| Qwen | 3.583 | **3.783** | 3.688 | +0.20 |
| DeepSeek | 3.679 | **3.804** | 3.773 | +0.13 |
| OLMoE | 3.812 | **3.873** | 3.845 | +0.06 |

Code IS slightly higher entropy than NL in every model — G8a is **partially**
supported. But the magnitude is small (only +0.06 to +0.20 nat, i.e. 1.6%–5.6%
of the uniform baseline).

### Effective experts (= exp(H_median))

| Model | nl | code | math |
|---|---|---|---|
| Qwen | 36.0 | 43.9 | 40.0 |
| DeepSeek | 39.6 | 44.9 | 43.5 |
| OLMoE | 45.2 | 48.1 | 46.8 |

Routing on code "spreads across" 3-8 more effective experts than on NL.

### Filtered-MI (the disambiguation)

Threshold = NL median entropy per model. Tokens kept if BOTH `e_L < thr` AND `e_{L+1} < thr`.

| Model | code-vs-nl gap unfiltered | code-vs-nl gap filtered | gap shrinkage | dominant explanation |
|---|---|---|---|---|
| Qwen | 0.339 nat | 0.140 nat | **−59%** | mostly G8a (OOD) |
| DeepSeek | 0.138 nat | 0.091 nat | **−34%** | mixed G8a+G8b |
| OLMoE | 0.572 nat | **0.486 nat** | **−15%** | mostly G8b (structural) |

For Qwen, filtering to confident tokens cuts the code-vs-NL gap by ~60% —
most of the gap WAS the OOD effect. For OLMoE, the gap barely moves —
even among equally-confident routing decisions, code's joint structure is
genuinely different.

## Verdict on G8

**Both G8a and G8b are real.** The relative weight depends on the model:

- **OLMoE-1B-7B**: structural (G8b) dominates. Even at matched confidence,
  code's cross-layer MI is 0.49 nat below NL's.
- **Qwen1.5-MoE-A2.7B**: OOD (G8a) dominates. After matching confidence,
  the gap shrinks to 0.14 nat. Most of code's MI deficit was the router
  being less confident.
- **DeepSeek-V2-Lite**: mixed. Both effects of comparable size.

### Why is OLMoE the structural-residual outlier? (revised post-hoc)

An earlier draft of this page hypothesized that OLMoE's large structural
residual was because *code is more in-distribution* for OLMoE's training
mix. **Looking up the published training data falsifies this**:

| Model | Code share in pretraining | Source |
|---|---|---|
| OLMoE-1B-7B-0924 | **~2.5%** (StarCoder 101B / 4060B; explicitly *reduced* from OLMo 1.7's 15.4%) | OLMoE paper, arXiv:2409.02060, Table 2 |
| Qwen1.5-MoE-A2.7B | not disclosed; upcycled from Qwen-1.8B (web docs + code, EN/CN focus) | Qwen blog; no exact ratio |
| DeepSeek-V2-Lite | not disclosed; DeepSeek-Coder-V2 paper treats V2 corpus as the "natural language portion only" → V2 base is mostly NL with modest code | DeepSeek-V2 tech report; DeepSeek-Coder-V2 §2.1 |

OLMoE has the **lowest** published code share. So if anything, code is
*more* OOD for OLMoE, not less — yet the OOD/confidence channel barely
moves the gap (15% shrinkage). The "code is in-distribution" explanation
is **wrong**.

**Revised explanation (saturated-router hypothesis).** OLMoE's median NL
router entropy is **3.812 nat**, the highest of the three (Qwen 3.583,
DeepSeek 3.679; effective experts ≈ 45 of 64 on plain NL). Its router is
already close to uniform on NL, so the OOD/confidence axis has little
dynamic range left to widen on code. The MI deficit on code therefore
*has* to be carried by structural joint patterns — there is no
confidence "headroom" for it to live in. Possibly compounded by:

- **top-k=8 routing** (vs Qwen 4, DeepSeek 6) → each routing decision
  encodes more bits in the joint, giving systematic structural
  differences more visible signal.
- **stronger aux-loss balancing pressure** flattening per-layer marginals
  → conditional structure is the only place left for code-specific
  patterns to show up.

This opens a new gap (gap:G9) — for aux-loss-balanced models with
saturated routers, the code-vs-NL structural difference is independent
of router confidence; the mechanism is unknown.

## Implication

For a system paper:
- A simple confidence-aware prefetcher (skip prefetch when entropy is high)
  recovers a substantial part of the cross-domain stability — esp. on
  Qwen-class models where OOD dominates.
- For OLMoE-class models (or any model with strong code training), structural
  domain-conditional predictors are still required.
- "Domain-aware prefetcher" is a real engineering surface, not just an
  evaluation question.

## Reproducibility

```bash
# Probes (3 models × 3 domains, ~5 min total + ~1 min model swaps):
python scripts/22_probe_router_entropy.py --model deepseek-ai/DeepSeek-V2-Lite --tag deepseek --domains nl code math
# (delete deepseek cache, download Qwen)
python scripts/22_probe_router_entropy.py --model Qwen/Qwen1.5-MoE-A2.7B --tag qwen --domains nl code math
# (delete Qwen cache, download OLMoE)
python scripts/22_probe_router_entropy.py --model allenai/OLMoE-1B-7B-0924 --tag olmoe --domains nl code math

# Analysis (offline, ~1 min):
python scripts/70_router_entropy.py
# Renders F17 (entropy histogram), F18 (per-layer entropy), F19 (filtered-MI bars)
```

## Connections

- partially_resolves: gap:G8
- supports: claim:C4 (the residual structural difference is the "real" code-vs-nl finding)
- extends: exp:m8_multidomain_mi (joins entropy with M8 routing traces for filtered MI)
- inspires: idea:confidence_aware_prefetcher (new follow-up idea)
