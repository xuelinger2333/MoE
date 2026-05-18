# Idea Report — RouteWeaver

**Source.** Selected from `deep-research-report.md` (方案二).
**Stage.** Idea formulation; this document defines the idea and the empirical preconditions
that the current measurement experiment validates.

---

## 1. Statement

**RouteWeaver** is an online runtime for MoE training/inference that closes the loop between
three signals that current systems treat independently:

1. **Router output** — `(token, top-k experts, weight)` produced per layer per step.
2. **Expert placement / replication** — which expert lives on which rank (and whether replicated).
3. **Physical interconnect topology** — NVLink-domain vs cross-node bandwidth, current congestion, p99.

Today these are decoupled. NetMoE moves *samples* but not experts; Occult/C2R adjust *router/collocation*
but ignore live congestion; SmartMoE plans *placement* offline. RouteWeaver unifies them in a
**three-timescale runtime**:

- **Fast path (per step):** batch permutation + dispatch plan generation that respects current
  topology cost matrix.
- **Medium path (per N steps):** expert grouping / replica placement updates from a live
  co-activation matrix.
- **Slow path (per checkpoint):** light router regularization to keep co-activation locality
  stable enough that the medium path's plans remain valid.

Why this is publishable today: the three components have each been validated individually
(NetMoE @ ICLR'25, Occult @ ICML'25, MoNTA arXiv'24), but **no system jointly optimizes them
in a closed loop**. That gap defines the contribution.

---

## 2. Empirical Preconditions (what we measure now)

The idea is only worth building if two empirical claims hold on real MoE models. The current
experiment measures both.

### Claim C1 — Router-induced cross-rank traffic is concentrated.

If a few `(src_rank, dst_expert)` pairs carry most of the AllToAll traffic, then locality-aware
placement / replication can shrink those high-volume pairs to intra-rank moves. If the traffic
is uniform across pairs, RouteWeaver's medium path has no headroom.

- **Measure.** For every routed token in every layer, record `(layer, step, src_rank, dst_expert)`.
  EP rank assignments simulated offline (`expert_id // (num_experts // ep_size)`,
  `token_global_idx // (batch_size // ep_size)`).
- **Statistic.** Cumulative distribution over `(src_rank, dst_expert)` pairs ordered by traffic.
- **Success threshold.** Top **20%** of pairs carry **≥50%** of cross-rank tokens.
  Motivation-strong if ≥70%.

### Claim C2 — Experts co-activate in clustered (not uniform) patterns.

If certain experts are repeatedly co-selected by the same tokens, hierarchical clustering
will reveal expert *groups*. Those groups become the natural unit for placement and replication.
If co-activation is near-uniform, expert grouping has no signal to exploit.

- **Measure.** Per layer, accumulate `M[i][j] += 1` whenever experts `i` and `j` are both in a
  token's top-k set.
- **Statistic.** Block structure in `M` after row/col reordering; silhouette score of best
  hierarchical clustering.
- **Success threshold.** At least one layer shows visible block structure
  (silhouette > 0.2 on best `k`).

---

## 3. Failure Modes (what would kill the idea early)

| Symptom | Diagnosis | Pivot |
|---|---|---|
| Traffic is uniform across all `(src, dst)` pairs | Either the model is too small / routing is already balanced, or our DP/EP assumption mismatches the real workload | Try the larger / sparser backup model (Qwen-MoE), or report negative result and pivot to 方案一 (semantic transport) |
| Co-activation matrix is dense / no clusters | Top-k is too small to exhibit grouping, or experts are too few to specialize | Increase top-k view (record all selected, not just top-2), try a different model |
| Strong cluster structure but no traffic concentration | Means co-activation exists but EP layout already places clustered experts together — RouteWeaver-medium has nothing to do | Pivot focus toward the fast path only (batch permutation), still publishable but narrower |
| Both claims hold but only weakly | Idea is alive but motivation figures will be weaker | Add a third probe: per-domain skew (multi-domain vs single-domain CDFs) to argue dynamic re-grouping is needed |

---

## 4. Compute Cost & Risk

| Item | Value |
|---|---|
| Model | DeepSeek-V2-Lite (15.7B, 64 routed experts, top-6, 27 MoE layers) |
| Hardware | 1× node, 4× A100-80GB (Cloudlab `d8545-10s10501`) |
| Mode | Inference only (forward pass on calibration data) |
| Total tokens | ~80k across 4 domains |
| Estimated GPU-time | ~1.5 hours including buffer |
| Estimated calendar time | 3–4 hours including env setup, downloads, debugging |

---

## 5. Downstream Use of These Figures

- **Paper motivation section.** All four core figures (CDF, locality ratio, co-activation heatmap,
  cluster dendrogram) are designed to be paper-ready.
- **Method design.** Cluster boundaries from C2 directly seed RouteWeaver's medium-path placement
  algorithm. Concentration profile from C1 sets the per-layer locality target.
- **Reviewer defense.** Pre-empts "is this problem real?" with an empirical baseline measured on
  an open, well-known MoE model.
