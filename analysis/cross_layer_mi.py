"""Cross-layer mutual information between adjacent MoE expert selections.

Hypothesis (G6): per-layer load balancing only constrains the marginal
P(expert | layer) to be uniform. It does NOT constrain the joint
P(expert_L, expert_{L+1}) — so cross-layer structure may persist even when
within-layer balance is perfect.

We use top-1 expert per token per layer (the highest-weight selection)
and treat each token as one (e_L, e_{L+1}) sample.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def top1_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a routing trace to one row per (token, layer) carrying top-1 expert.

    Returns a wide DataFrame indexed by ``token_idx_in_run`` with one column per layer.
    """
    top1 = df[df["topk_rank"] == 0]
    wide = top1.pivot(index="token_idx_in_run", columns="layer", values="expert_id")
    wide = wide.dropna(how="any")  # drop tokens that don't have a value at every layer
    return wide.astype(np.int32)


def joint_counts(x: np.ndarray, y: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    """Build the [n_x, n_y] joint count matrix."""
    flat = x.astype(np.int64) * n_y + y.astype(np.int64)
    counts = np.bincount(flat, minlength=n_x * n_y).reshape(n_x, n_y)
    return counts


def mutual_information_nats(C: np.ndarray) -> Tuple[float, float, float, float]:
    """MI(X;Y), H(X), H(Y), H(X|Y) — all in nats — from joint count matrix C."""
    N = C.sum()
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0
    P = C / N
    Px = P.sum(axis=1, keepdims=True)
    Py = P.sum(axis=0, keepdims=True)

    # MI: only sum over nonzero P(x,y)
    nz = P > 0
    log_ratio = np.zeros_like(P)
    denom = Px @ Py
    safe = nz & (denom > 0)
    log_ratio[safe] = np.log(P[safe] / denom[safe])
    mi = float((P * log_ratio).sum())

    # H(X), H(Y)
    px = Px.flatten()
    py = Py.flatten()
    Hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    Hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())

    Hx_given_y = Hx - mi  # equivalent to H(X) - I(X;Y)
    return mi, Hx, Hy, Hx_given_y


def per_source_kl(C: np.ndarray) -> np.ndarray:
    """KL(P(Y|X=i) || P(Y)) for each i, in nats. Returns a length-n_x array."""
    N = C.sum()
    if N == 0:
        return np.zeros(C.shape[0])
    Py = C.sum(axis=0) / N  # marginal P(Y)
    Px = C.sum(axis=1)       # count of X=i
    kls = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        if Px[i] == 0:
            continue
        Py_given_xi = C[i] / Px[i]
        nz = (Py_given_xi > 0) & (Py > 0)
        kls[i] = float((Py_given_xi[nz] * np.log(Py_given_xi[nz] / Py[nz])).sum())
    return kls


def shuffled_null_mi(x: np.ndarray, y: np.ndarray, n_x: int, n_y: int,
                     n_repeats: int = 5, seed: int = 42) -> float:
    """Mean MI under the WEAK (i.i.d.) null: globally shuffle Y across all tokens.

    Breaks every form of structure including topic/sequence-level coupling.
    Use as a finite-sample bias estimate, not as a tight comparator.
    """
    rng = np.random.default_rng(seed)
    mis = []
    for _ in range(n_repeats):
        y_shuf = rng.permutation(y)
        C_null = joint_counts(x, y_shuf, n_x, n_y)
        mi, *_ = mutual_information_nats(C_null)
        mis.append(mi)
    return float(np.mean(mis))


def within_sequence_shuffle_null_mi(
    x: np.ndarray, y: np.ndarray, sequence_ids: np.ndarray,
    n_x: int, n_y: int, n_repeats: int = 3, seed: int = 42,
) -> float:
    """Mean MI under the STRICT null: shuffle Y only WITHIN each sequence.

    Preserves the sequence's local expert preferences (topic-level coupling)
    while breaking the per-token (e_L, e_{L+1}) pairing. The gap
    ``MI(observed) − MI(within-seq null)`` is the per-token cross-layer
    predictability that cannot be explained by sequence-level locality.
    """
    rng = np.random.default_rng(seed)
    mis = []
    # Pre-build per-sequence index lists once
    unique_seqs, inverse = np.unique(sequence_ids, return_inverse=True)
    seq_index_lists = [np.where(inverse == i)[0] for i in range(len(unique_seqs))]

    for r in range(n_repeats):
        y_shuf = y.copy()
        for indices in seq_index_lists:
            if indices.size > 1:
                shuf = rng.permutation(indices)
                y_shuf[indices] = y[shuf]
        C_null = joint_counts(x, y_shuf, n_x, n_y)
        mi, *_ = mutual_information_nats(C_null)
        mis.append(mi)
    return float(np.mean(mis))


def per_source_entropy_reduction(C: np.ndarray) -> dict:
    """For each source X=i, compute H(Y|X=i) and the reduction H(Y) - H(Y|X=i).

    Returns dict with arrays:
        H_marginal: float — entropy of marginal P(Y) in nats
        H_conditional: ndarray[n_x] — entropy of P(Y|X=i) for each i (nan if unobserved)
        reduction:    ndarray[n_x] — H_marginal - H_conditional (nan if unobserved)
        effective_y:  ndarray[n_x] — exp(H_conditional), interpretable count
        effective_y_marginal: float — exp(H_marginal)
        n_per_source: ndarray[n_x] — sample count for each source
    """
    N = C.sum()
    n_x = C.shape[0]
    Py = C.sum(axis=0) / max(1, N)
    H_marg = float(-(Py[Py > 0] * np.log(Py[Py > 0])).sum())

    H_cond = np.full(n_x, np.nan)
    n_per = C.sum(axis=1)
    for i in range(n_x):
        if n_per[i] == 0:
            continue
        p = C[i] / n_per[i]
        H_cond[i] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    reduction = H_marg - H_cond
    eff_y = np.exp(H_cond)
    return {
        "H_marginal": H_marg,
        "H_conditional": H_cond,
        "reduction": reduction,
        "effective_y": eff_y,
        "effective_y_marginal": float(np.exp(H_marg)),
        "n_per_source": n_per.astype(np.int64),
    }


def all_layer_pairs_mi(
    wide: pd.DataFrame,
    num_experts: int,
    sequence_ids: np.ndarray | None = None,
    distances: Tuple[int, ...] = (1, 2, 4, 8),
    null_repeats: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """For each (L, L+d) with d ∈ distances, compute observed MI and BOTH nulls.

    If ``sequence_ids`` is given (one entry per row in ``wide``, aligned by
    insertion order), also compute the strict within-sequence shuffle null.
    """
    layers = sorted(int(c) for c in wide.columns)
    out: List[Dict] = []
    for d in distances:
        for L in layers:
            L2 = L + d
            if L2 not in wide.columns:
                continue
            x = wide[L].to_numpy()
            y = wide[L2].to_numpy()
            C = joint_counts(x, y, num_experts, num_experts)
            mi, Hx, Hy, _ = mutual_information_nats(C)
            null_iid = shuffled_null_mi(x, y, num_experts, num_experts,
                                        n_repeats=null_repeats, seed=seed + L)
            row = {
                "L": L, "d": d, "L_plus_d": L2,
                "n_tokens": int(C.sum()),
                "MI_nats": mi,
                "H_L": Hx, "H_Lpd": Hy,
                "null_iid_MI_nats": null_iid,
                "MI_minus_null_iid": mi - null_iid,
                "MI_norm_by_H_Lpd": mi / Hy if Hy > 0 else 0.0,
            }
            if sequence_ids is not None:
                null_seq = within_sequence_shuffle_null_mi(
                    x, y, sequence_ids, num_experts, num_experts,
                    n_repeats=null_repeats, seed=seed + L,
                )
                row["null_within_seq_MI_nats"] = null_seq
                row["MI_minus_null_within_seq"] = mi - null_seq
            out.append(row)
    return out
