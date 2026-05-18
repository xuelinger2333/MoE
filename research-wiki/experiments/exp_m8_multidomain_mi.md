---
type: experiment
node_id: exp:m8_multidomain_mi
title: "M8 — Multi-domain stability of cross-layer MI (Qwen + DeepSeek + OLMoE × {code, math, nl})"
status: completed
date: 2026-05-15T06:31:53Z
hardware: "1 node × 4 A100-SXM4-40GB (Cloudlab d8545-10s10501)"
mode: inference
seed: 42
models: ["Qwen/Qwen1.5-MoE-A2.7B", "deepseek-ai/DeepSeek-V2-Lite", "allenai/OLMoE-1B-7B-0924"]
domains: ["nl (wikitext)", "code (codeparrot-clean-valid)", "math (gsm8k)"]
trace_dirs:
  - "outputs/traces/m1_deepseek_wikitext/  (nl)"
  - "outputs/traces/m4_qwen_wikitext/  (nl)"
  - "outputs/traces/m7_olmoe_wikitext/  (nl)"
  - "outputs/traces/m8_qwen_code/"
  - "outputs/traces/m8_qwen_math/"
  - "outputs/traces/m8_olmoe_code/"
  - "outputs/traces/m8_olmoe_math/"
  - "outputs/traces/m8_deepseek_code/"
  - "outputs/traces/m8_deepseek_math/"
config:
  batch_size: 4
  seq_len: 1024
  n_batches: 50
  ep_size: 4
n_tokens_per_run: 204800
total_routing_events: ~211M
elapsed_total_minutes: ~5
tested_claims: ["claim:C4"]
tags: ["MoE", "multidomain", "cross-layer", "mutual-information", "stability"]
notes: "Required Phase 1 (Qwen+OLMoE both cached, run their code+math) then Phase 2 (delete Qwen+OLMoE, re-download DeepSeek, run its code+math)."
---

# M8 — Multi-domain stability of cross-layer MI

## What was measured

For each of 3 models × 3 domains (= 9 cells), run an M1-style 50-batch probe
and compute MI(L, L+1) − within-sequence-shuffle null. Domains were chosen
to span structurally different distributions: encyclopedic prose (WikiText),
Python source (CodeParrot-clean-valid), and chain-of-thought arithmetic
(GSM8K).

## Headline result table (MI − strict null, mean over adjacent layer pairs)

| Model | code | math | nl | range |
|---|---|---|---|---|
| **Qwen1.5-MoE-A2.7B** | **1.190** | 1.580 | 1.529 | 0.39 |
| **DeepSeek-V2-Lite** | **0.545** | 0.652 | 0.683 | **0.14** |
| **OLMoE-1B-7B** | **0.640** | 1.077 | 1.212 | **0.57** |

(All in nats. Strong threshold = 0.3 nat. **Every cell** clears the strong threshold by 1.7×–5.3×.)

Per-source frac ≥0.5 nat shows the same pattern even more sharply:

| Model | code | math | nl |
|---|---|---|---|
| Qwen | 82.5% | 97.9% | 97.3% |
| DeepSeek | 58.0% | 68.7% | 72.1% |
| OLMoE | 46.9% | 78.7% | 92.5% |

## Three findings

### Finding 1 — H1a (domain-invariant) is REJECTED

Every model shows non-trivial domain dependence (range 0.14-0.57 nat). Code is
**always** the lowest-MI domain. The phenomenon does not vanish, but it
attenuates substantially under domain shift.

### Finding 2 — Code is the universal "least predictable" domain

Across all three architectures (different `top_k`, different balancing scheme,
different scale), code consistently sits below NL and math. The most plausible
reasons:
- Code has more identifier diversity / less semantic redundancy per token.
- Code may be slightly OOD relative to the predominantly-NL training mix of
  these models, making router decisions less confident → top-1 less stable.
- Code's strict syntactic regularity is mediated by *different* expert
  combinations than prose, reducing the sequential coupling.

### Finding 3 — DeepSeek (loss-free balancing) is the most domain-stable

DeepSeek's domain range is just 0.14 nat — half of Qwen's 0.39 and a quarter
of OLMoE's 0.57. Combined with the round-2 finding that DeepSeek has the
lowest absolute cross-layer MI of the three (~half of Qwen/OLMoE), this
suggests **auxiliary-loss-free balancing flattens BOTH absolute MI AND its
domain variation**. Two effects, one mechanism.

## Implications for system design

A prefetcher / placement scheme tuned on NL workloads **will underperform on
code workloads**, with degradation magnitude depending on the model:
- Qwen: −22% MI on code vs NL
- DeepSeek: −20%
- OLMoE: −47% (worst)

So domain-conditional predictors are not optional for production-deployable
systems. But the *phenomenon* survives — even the worst (model, domain) cell
(OLMoE × code at 0.640 nat) is still 2× the strong threshold.

## Reproducibility

```bash
# Probes:
python scripts/21_probe_multidomain_mi.py --model Qwen/Qwen1.5-MoE-A2.7B  --tag qwen     --domains code math
python scripts/21_probe_multidomain_mi.py --model allenai/OLMoE-1B-7B-0924 --tag olmoe    --domains code math
python scripts/21_probe_multidomain_mi.py --model deepseek-ai/DeepSeek-V2-Lite --tag deepseek --domains code math
# Reuses existing m1/m4/m7 nl traces.

# Analysis:
python scripts/60_multidomain_mi.py
# Renders F14, F15, F16 + multidomain_mi.json
```

## Connections

- tests claim: claim:C4
- supports: claim:C4, idea:cross_layer_routing_predictability (extends with domain analysis)
- extends: exp:m6_cross_layer_mi, exp:m7_olmoe_wikitext (reuses their nl traces)
- opens: gap:G8
