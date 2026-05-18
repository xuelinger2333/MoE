# Experiment Results — RouteWeaver Motivation Probes

**Date:** 2026-05-15
**Compute:** Cloudlab `d8545-10s10501.wisc.cloudlab.us`, 4× A100-SXM4-40GB, single node, inference-only
**Models probed:** DeepSeek-V2-Lite (primary), Qwen1.5-MoE-A2.7B (backup, added because DeepSeek showed near-uniform routing)
**Total wall-clock:** ~30 minutes (env + 4 probe runs + analysis)

---

## Headline finding

**Both claims FAIL** on both modern open-source MoE models. Router-induced traffic is **near-uniform** at the `(src_rank, dst_expert)` level, and co-activation has only **weak** cluster structure (silhouette ~0.17–0.19, threshold was 0.20).

This is a **negative result for RouteWeaver as originally specified** — but it is a *clean, reportable, motivation-shifting* negative result, not a measurement artifact.

| Metric | DeepSeek-V2-Lite | Qwen1.5-MoE-A2.7B | Threshold | Verdict |
|---|---|---|---|---|
| `(src,dst)` pairs (ep_size=4) | 4×64=256 | 4×60=240 | — | — |
| Top-5% pairs share | 6.1% | 5.7% | — | ≈ uniform (5%) |
| Top-10% pairs share | 11.9% | 11.1% | — | ≈ uniform (10%) |
| **Top-20% pairs share** | **22.7%** | **21.5%** | ≥ 50% | **❌ FAIL** |
| Top-50% pairs share | 53.4% | 52.2% | — | ≈ uniform (50%) |
| Best silhouette score | 0.169 (layer 25, k=16) | 0.186 (layer 15, k=16) | ≥ 0.20 | **❌ FAIL** |
| Cross-rank token ratio | ~75% (= (ep-1)/ep) | ~75% | — | matches uniform routing |

The CDFs of both models lie within ~2 percentage points of the uniform reference diagonal at every cut-off (see `outputs/figures/F8_compare_cdf.png`).

---

## Per-figure findings

| File | What it shows | Reading |
|---|---|---|
| `F1_traffic_cdf.png` | DeepSeek per-layer CDF over `(src, dst)` pairs | All 26 layers cluster tightly around the same near-diagonal curve. No layer is meaningfully more concentrated than another. |
| `F2_locality_ratio.png` | DeepSeek per-layer cross-rank vs intra-rank | Cross-rank ≈ 75% per layer (= (ep-1)/ep for uniform routing under ep=4). Confirms experts are spread evenly across the 4 simulated ranks. |
| `F3_per_domain_cdf.png` | DeepSeek M2: WikiText vs C4 vs MMLU | All three domain curves overlap to within line width. The router does not skew differently for prose, web text, or structured Q&A. |
| `F4_coactivation_heatmap.png` | DeepSeek co-activation matrices for layers 0/13/25 | Scattered bright pixels with no contiguous blocks. The brightest off-diagonal pairs (~0.7 normalized) exist but are isolated. |
| `F5_dendrogram.png` | DeepSeek hierarchical clustering of best layer (25) | Mostly flat dendrogram; cuts at any reasonable height produce many small singletons rather than 4–8 distinct groups. |
| `F6_silhouette_vs_k.png` | DeepSeek silhouette vs k for sampled layers | All curves stay below the 0.2 threshold; no layer/k combination crosses it. |
| `F7_compare_heatmap.png` | DeepSeek vs Qwen best-layer co-activation, side by side | Qualitatively identical sparse-scatter pattern. Qwen is *not* more clustered than DeepSeek. |
| `F8_compare_cdf.png` | DeepSeek vs Qwen aggregate CDF, with uniform reference | Both lines essentially coincide with the gray uniform-reference diagonal across the full 0..1 range. |

---

## Why the claims failed (interpretation)

1. **Modern MoE training explicitly minimizes the very skew RouteWeaver was hoping to exploit.** DeepSeek-V2 uses auxiliary-loss-free balancing; Qwen1.5-MoE uses an auxiliary load-balance loss. Both *succeed* — by their training objectives — at producing routing that is statistically uniform across the expert set. From the network's point of view, every expert receives ~the same volume.

2. **The simulated EP layout assumes contiguous blocks of experts per rank.** With uniform routing across experts, *any* such layout produces the same near-uniform `(src, dst)` distribution. Permuting the placement changes which pairs are "intra" vs "cross" but not how skewed the *distribution* is.

3. **Co-activation does have visible non-uniformity at the pair level** (the brightest off-diagonal cells reach ~0.6–0.9 normalized), but the structure is scattered rather than block-clustered. Hierarchical clustering returns one big cluster + many singletons, hence silhouette < 0.2.

4. **The 4-domain comparison rules out the "dynamic per-domain re-grouping" rescue.** All three open domains (encyclopedic prose / web text / structured Q&A) produce CDFs that differ by < 1 percentage point.

This combination — uniform load balance + scattered (not clustered) co-activation + domain-invariant routing — leaves RouteWeaver-as-originally-designed with very little headroom.

---

## What this implies for the research direction

The probe **kills the original RouteWeaver pitch** but **opens three more focused directions**, all consistent with the prior literature surveyed in `deep-research-report.md`:

1. **Move the locality target from experts to TOKENS / SAMPLES.** NetMoE @ ICLR'25 already won on this axis. With routing already uniform across experts, the only locality lever left is *which tokens land on which DP rank*. RouteWeaver-style ideas should target sample placement (and possibly *sample-level* batch reordering), not expert grouping.

2. **Target regimes where balance breaks.** Uniform routing is only the equilibrium of converged training. Probe early checkpoints (first 10–20% of training), or fine-tuning-on-narrow-domain regimes — both should show much stronger skew, and a domain-aware re-grouping scheme could matter there.

3. **Pivot to 方案一 (LossBound-EP) or 方案三 (HybridEP-Collective).** Neither requires routing concentration. 方案一 exploits *per-token semantic value*; 方案三 builds a hierarchical exact/approx primitive. Both still benefit from the trace infrastructure built here.

A small 4th option worth flagging: the *non-uniformity that does exist* — bright off-diagonal co-activation pairs — is real (some pair counts are 4–10× the uniform expectation). It's just too scattered to feed hierarchical clustering. A more targeted method (e.g., maximum-weight matching or top-K pair pinning) could exploit those pairs without needing dense cluster structure. This is a much narrower contribution than RouteWeaver but still publishable.

---

## Verdict on next step

**Do NOT proceed with RouteWeaver as specified.**

Recommend, in order of effort:

1. **Repeat probe on an early-training checkpoint** of any open MoE model (e.g., DeepSeekMoE-16B-base intermediate checkpoint if released, or a from-scratch Tiny MoE you train for 1k steps). If skew is much higher early, RouteWeaver becomes an "early-training accelerator" — a narrower but real contribution.
2. **Re-scope to NetMoE-style sample placement** with the existing trace infrastructure as the empirical foundation.
3. **Switch to 方案一 (LossBound-EP) or 方案三 (HybridEP-Collective).** Reuse the entire trace/hook stack — it captures token-level routing + weights, exactly what those directions need.

---

## Reproducibility

All artifacts are under `d:/DevProjects/MoE/`:

- `outputs/traces/m1_deepseek_wikitext/` — DeepSeek M1 trace (172 MB, 31.9M routing events)
- `outputs/traces/m2_deepseek_multidomain/{wikitext,c4,mmlu}/` — DeepSeek M2 traces (127 MB total)
- `outputs/traces/m4_qwen_wikitext/` — Qwen M1 trace (~118 MB, 19.7M routing events)
- `outputs/figures/` — F1–F8 PNG+PDF + `claim_summary.json` + `model_compare.json`

Reproduce with:

```bash
# Sanity
python scripts/00_smoke.py
# Single-domain DeepSeek (the C1+C2 baseline)
python scripts/10_probe_single.py
# Multi-domain DeepSeek (the per-domain comparison)
python scripts/20_probe_multidomain.py --domains wikitext c4 mmlu
# Qwen cross-check
python scripts/10_probe_single.py --model Qwen/Qwen1.5-MoE-A2.7B --out outputs/traces/m4_qwen_wikitext
# Render F1-F6 + claim verdicts
python scripts/30_analyze.py
# Render F7-F8 cross-model comparison
python scripts/40_compare_models.py
```

Run config snapshot for each trace lives in `<trace_dir>/run_meta.json`. Single seed (42) with `torch.backends.cudnn.deterministic = True`.
