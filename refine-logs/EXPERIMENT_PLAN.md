# Experiment Plan — RouteWeaver Motivation Probes

**Source idea.** `idea-stage/IDEA_REPORT.md`
**Source plan.** `PLAN.md` (探测A + 探测B)
**Hardware.** Cloudlab `d8545-10s10501.wisc.cloudlab.us`, 4× A100-80GB, single node.
**Mode.** Inference only.
**Total budget.** ≤2 GPU-hours.

---

## Claims under test

| ID | Claim | Probe | Success criterion |
|---|---|---|---|
| C1 | Router-induced cross-rank traffic is concentrated | 探测A | Top 20% of `(src_rank, dst_expert)` pairs carry ≥50% of cross-rank tokens |
| C2 | Experts co-activate in clustered patterns | 探测B | At least one layer with silhouette > 0.2 on hierarchical clustering of co-activation |

Both must hold for RouteWeaver to be worth building. Either failing → pivot.

---

## Compared "systems"

This is a measurement-only experiment. There is **no system under test** vs baselines —
the comparison is *data → claim verdict*. We do however report the same statistics across:

- Two model checkpoints (DeepSeek-V2-Lite primary; Qwen1.5-MoE-A2.7B as backup if M0 fails)
- Two scope levels (single-domain WikiText; multi-domain WikiText+C4+TheStack+MMLU)
- Three layer depth bins (early / middle / late MoE layers)

---

## Setup details

### Model

- **Primary:** `deepseek-ai/DeepSeek-V2-Lite`
- HF `trust_remote_code=True` required.
- Load with `torch_dtype=bfloat16`, `device_map="auto"` (let HF spread layers across 4 GPUs).
- 64 routed experts + 2 shared (we ignore shared since they're always activated).
- `top_k = 6` per token.
- 27 MoE layers (every layer except the first dense layer).

### Data

| Source | HF id | Slice | Tokens after pack |
|---|---|---|---|
| WikiText-103 | `Salesforce/wikitext`, `wikitext-103-raw-v1` | `validation` | ~25k |
| C4 (English) | `allenai/c4`, `en` | `train`, first 200 docs | ~25k |
| The Stack (Python) | `bigcode/the-stack-smol` | `data/python` first 200 | ~25k |
| MMLU dev | `cais/mmlu`, `all` | `dev` (~285 examples concatenated) | ~5k |

Tokenize with the model's own tokenizer; pack to `seq_len=2048`.

### Probe configuration

```yaml
batch_size: 8
seq_len: 2048
ep_size: 4              # simulated number of expert-parallel ranks
num_experts: 64         # DeepSeek-V2-Lite routed experts
top_k: 6
seed: 42
torch_deterministic: true
```

### Random seeds

- Single seed (42) for the primary run; reproducibility comes from determinism flags +
  fixed dataset slices, not seed sweep. (Multi-seed adds nothing for a measurement study.)

---

## Run order

| ID | Goal | Setup | Output | Estimated time |
|---|---|---|---|---|
| **M0 — Sanity** | Hooks fire, trace well-formed | DeepSeek-V2-Lite, batch=2, seq=64, 1 step | `outputs/traces/m0_smoke.parquet` | <5 min |
| **M1 — Single-domain probe** | Baseline trace for both claims | WikiText-103 val, batch=8, seq=2048, 200 batches | `outputs/traces/m1_deepseek_wikitext.parquet` | 20 min |
| **M2 — Multi-domain probe** | Diversity / cross-domain co-activation | 4 datasets × 50 batches each | `outputs/traces/m2_deepseek_multidomain.parquet` | 30 min |
| **M3 — Analysis & figures** | 4–6 motivation figures + summary | Offline | `outputs/figures/*.{png,pdf}` + `EXPERIMENT_RESULTS.md` | 30 min |
| **M4 — Backup model (NICE-TO-HAVE)** | Cross-architecture validation | Qwen1.5-MoE-A2.7B, M1 setup | `outputs/traces/m4_qwen_wikitext.parquet` | 25 min |

**M4 is dropped from default scope** per user decision; run only if claims fail on DeepSeek
and we need to check whether it's a model-specific artifact.

---

## Metrics

### Traffic-level (for C1)

- **Cross-rank ratio per layer.** `cross_rank_tokens / total_tokens` per MoE layer.
- **`(src_rank, dst_expert)` pair distribution.** Lorenz curve + Gini coefficient.
- **Top-k pair share.** Fraction of cross-rank tokens carried by top 5%, 10%, 20%, 50% of pairs.

### Routing-structure (for C2)

- **Per-layer co-activation matrix `M[i][j]`** (64×64).
- **Hierarchical clustering** with `scipy.cluster.hierarchy.linkage(1 - M_norm, 'average')`.
- **Silhouette scores** for `k ∈ {2,4,8,16}`.
- **Domain divergence.** Frobenius distance between per-domain `M` matrices (M2 only).

---

## Figures (publishable artifacts)

| # | Figure | Source data | Insight it carries |
|---|---|---|---|
| F1 | Traffic CDF over `(src_rank, dst_expert)` pairs (per layer + aggregate) | M1 trace | C1 — concentration / locality headroom |
| F2 | Cross-rank vs intra-rank ratio per layer (bar) | M1 trace | C1 — where cross-rank cost lives in the network |
| F3 | Per-domain traffic CDF overlay | M2 trace | Argues for *dynamic* (not static) placement |
| F4 | Co-activation heatmap, 3 representative layers | M1 trace | C2 — visual block structure |
| F5 | Cluster dendrogram of best-clustered layer | M1 trace | C2 — natural expert groups |
| F6 (optional) | Silhouette score vs `k`, vs layer depth | M1 trace | Quantitative C2 + how depth affects cluster quality |

---

## Verification path

Each milestone has a clear pass/fail check (see `verification` section in
`C:\Users\13178\.claude\plans\deep-research-report-md-plan-md-ssh-che-gleaming-kite.md`).

After M3, the empirical summary in `EXPERIMENT_RESULTS.md` must answer three questions:

1. Does C1 hold? (yes/no/marginal — cite top-20% share number)
2. Does C2 hold? (yes/no/marginal — cite best silhouette + show heatmap)
3. Should we proceed to RouteWeaver design? (yes / pivot / iterate)
