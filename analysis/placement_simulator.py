"""Trace-driven placement simulator: quantify cross-rank bytes saved.

Question: given the empirical (token, layer, expert) trace, what fraction of
inter-rank dispatches can we avoid by exploiting cross-layer routing
correlation, vs. a strawman random placement?

Model
-----

We have ``E`` experts per layer, ``ep`` expert-parallel ranks (each holding
``E/ep`` experts). For each routed event (token t, layer L, expert e):

    - The token's *source rank* is ``src_rank(t) = (t mod tokens_per_step) // tokens_per_rank``.
      This is fixed by the data-parallel partition.

    - The *destination rank* is determined by the placement map of layer L:
      ``dst_rank(L, e) = P_L(e) ∈ {0, ..., ep-1}``.

    - The event incurs cross-rank traffic iff ``src_rank(t) != P_L(e)``.

A strategy is an algorithm that produces ``{P_L : L in layers}``. We require
each ``P_L`` to be balanced: each rank hosts exactly ``E/ep`` experts.

Strategies
----------

S0 random
    Random balanced partition per layer. Asymptotic cross-rank rate is
    ``1 - 1/ep``. Trivial lower bound.

S1 per-layer frequency  (Occult-style, per-layer optimal placement)
    For each layer L independently, count tokens routed to expert e from
    rank r: ``N_L[e, r]``. Solve a balanced assignment: assign experts to
    ranks to maximise ``sum over (e, r) of N_L[e, r] * 1[P_L(e) == r]``.
    Greedy approximation with capacity constraints is sufficient; the
    optimum is small (60 experts × 4 ranks) and amenable to exact solve.

S2 cross-layer trajectory  (the proposed method)
    Layer 0: same as S1.
    Layer L>0: encourage placement to follow the previous layer.
    Concretely, compute a "virtual demand" matrix:
        D_L[e, r] = sum over t of  1[routed(t, L) == e] * 1[placed(t, L-1) at r]
    i.e. the demand on rank r for expert e of layer L is dominated by
    tokens that were physically located at rank r right after layer L-1's
    expert step. Then solve the same balanced-assignment problem with D_L.
    This propagates cross-layer correlation into placement decisions.

S3 oracle
    Per-token optimal placement assuming infinite reconfiguration. Equivalent
    to S1 with a tight bound (since per-layer optimal *is* the per-layer
    oracle). We do not implement a more aggressive oracle; instead we use
    S1 as the "oracle" lower bound on within-layer placement reduction.

All strategies must agree on the same ``src_rank`` function (DP partition).

Reported metric
---------------

For each (strategy, ep, model, domain):
    * Cross-rank rate (fraction of routed events with src != dst). This
      directly maps to bytes moved if all dispatched events carry the same
      hidden dim.
    * Saving over random = (random_rate - strategy_rate) / random_rate
      (relative reduction in cross-rank traffic).
    * Saving over S1 (trajectory − layer-internal): incremental contribution
      of cross-layer information beyond Occult-style baseline. This is
      the **X** the user asked about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Source rank computation
# ---------------------------------------------------------------------------

def compute_src_rank(token_idx_in_run: np.ndarray, tokens_per_step: int, ep_size: int) -> np.ndarray:
    """Fixed DP partition: token's source rank = (intra-step token id) // (E_per_rank)."""
    if tokens_per_step % ep_size != 0:
        raise ValueError(f"tokens_per_step={tokens_per_step} not divisible by ep_size={ep_size}")
    tokens_per_rank = tokens_per_step // ep_size
    intra = token_idx_in_run.astype(np.int64) % tokens_per_step
    return (intra // tokens_per_rank).astype(np.int32)


# ---------------------------------------------------------------------------
# Assignment solver: assign N items to K buckets with capacity = N/K each.
# We use a greedy approximation: at each step, pick the (item, bucket) pair
# with highest weight whose bucket still has capacity, and assign.
# For N=60-64, K=2-8 this is well within optimal.
# ---------------------------------------------------------------------------

def balanced_assignment_greedy(weights: np.ndarray, ep_size: int) -> np.ndarray:
    """Assign rows (experts) to columns (ranks) with equal column capacity.

    weights[e, r] = score for placing expert e on rank r.
    Returns: array of length n_experts giving the chosen rank for each.
    """
    n_experts, n_ranks = weights.shape
    if n_ranks != ep_size:
        raise ValueError(f"weights shape {weights.shape} does not match ep_size {ep_size}")
    if n_experts % ep_size != 0:
        raise ValueError(f"n_experts={n_experts} not divisible by ep_size={ep_size}")
    cap = n_experts // ep_size

    flat_idx = np.argsort(-weights, axis=None)  # high to low
    assignment = np.full(n_experts, -1, dtype=np.int32)
    rank_count = np.zeros(n_ranks, dtype=np.int32)

    for fi in flat_idx:
        e, r = divmod(int(fi), n_ranks)
        if assignment[e] != -1:
            continue
        if rank_count[r] >= cap:
            continue
        assignment[e] = r
        rank_count[r] += 1
        if (assignment >= 0).sum() == n_experts:
            break

    if (assignment < 0).any():
        # rare edge: fill remaining experts into the least-full rank
        for e in np.where(assignment < 0)[0]:
            r = int(np.argmin(rank_count))
            assignment[e] = r
            rank_count[r] += 1
    return assignment


# ---------------------------------------------------------------------------
# Strategy: S0 random
# ---------------------------------------------------------------------------

def placement_random(layers: List[int], n_experts: int, ep_size: int, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    cap = n_experts // ep_size
    for L in layers:
        perm = rng.permutation(n_experts)
        assignment = np.empty(n_experts, dtype=np.int32)
        for i, e in enumerate(perm):
            assignment[e] = i // cap
        out[L] = assignment
    return out


# ---------------------------------------------------------------------------
# Strategy: S1 per-layer frequency (Occult-style)
# ---------------------------------------------------------------------------

def placement_layer_frequency(
    df: pd.DataFrame, layers: List[int], n_experts: int, ep_size: int
) -> Dict[int, np.ndarray]:
    """For each layer L, maximize within-layer match between expert and src_rank."""
    out: Dict[int, np.ndarray] = {}
    for L in layers:
        sub = df[df["layer"] == L]
        if sub.empty:
            out[L] = np.arange(n_experts) % ep_size  # fallback
            continue
        e_arr = sub["expert_id"].to_numpy()
        r_arr = sub["src_rank"].to_numpy()
        weights = np.zeros((n_experts, ep_size), dtype=np.int64)
        # weights[e, r] = count of events with this expert at this src_rank
        np.add.at(weights, (e_arr, r_arr), 1)
        out[L] = balanced_assignment_greedy(weights.astype(np.float64), ep_size)
    return out


# ---------------------------------------------------------------------------
# Strategy: S2 cross-layer trajectory
# ---------------------------------------------------------------------------

def placement_trajectory(
    df: pd.DataFrame, layers: List[int], n_experts: int, ep_size: int
) -> Dict[int, np.ndarray]:
    """Use the previous layer's placement to update the demand for the next.

    Concretely, after layer L-1 expert step, the token's "current" rank is
    the placement of its layer L-1 expert. For layer L, we count demand
    using the previous-layer placement as the rank-of-origin.
    Layer 0 is identical to S1.
    """
    out: Dict[int, np.ndarray] = {}
    # We need (token, layer, expert) and src_rank to bootstrap.
    layers_sorted = sorted(layers)
    # Build a per-(token, layer) expert lookup table from top-1 routing.
    # Assume df already filtered to topk_rank == 0 by caller.

    # Pre-index df by layer for speed
    by_layer: Dict[int, pd.DataFrame] = {L: df[df["layer"] == L] for L in layers_sorted}

    # token-indexed map to track the *current* physical rank
    # initial rank = src_rank, then updated after each layer.
    # We use the first layer's DataFrame to set per-token initial.
    first_L = layers_sorted[0]
    base = by_layer[first_L].drop_duplicates("token_idx_in_run")
    token_rank = pd.Series(
        base["src_rank"].to_numpy(),
        index=base["token_idx_in_run"].to_numpy(),
        dtype=np.int32,
    )

    for L in layers_sorted:
        sub = by_layer[L]
        if sub.empty:
            out[L] = np.arange(n_experts) % ep_size
            continue
        # Get the current rank for each token in this layer.
        # Tokens not yet seen fall back to src_rank.
        tids = sub["token_idx_in_run"].to_numpy()
        cur_rank = token_rank.reindex(tids).to_numpy()
        # Fallback to src_rank for any missing entries
        miss = np.isnan(cur_rank)
        if miss.any():
            cur_rank[miss] = sub["src_rank"].to_numpy()[miss]
        cur_rank = cur_rank.astype(np.int32)

        e_arr = sub["expert_id"].to_numpy()
        weights = np.zeros((n_experts, ep_size), dtype=np.int64)
        np.add.at(weights, (e_arr, cur_rank), 1)
        assignment = balanced_assignment_greedy(weights.astype(np.float64), ep_size)
        out[L] = assignment

        # Update token_rank: each token's rank is now the placement of the
        # expert it was routed to at this layer.
        new_ranks = assignment[e_arr]
        # Some tokens may appear multiple times if topk > 1 wasn't filtered;
        # take last assignment.
        token_rank.loc[tids] = new_ranks
    return out


# ---------------------------------------------------------------------------
# Metric: cross-rank rate
# ---------------------------------------------------------------------------

def apply_placement(df: pd.DataFrame, placement: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute dst_rank array for every routing event in df."""
    layers = df["layer"].to_numpy()
    experts = df["expert_id"].to_numpy()
    dst = np.empty(len(df), dtype=np.int32)
    for L, asn in placement.items():
        mask = layers == L
        dst[mask] = asn[experts[mask]]
    return dst


def cross_rank_rate_dispatch_return(df: pd.DataFrame, placement: Dict[int, np.ndarray]) -> float:
    """Standard MoE metric: token always starts at src_rank for every layer.

    Each MoE layer dispatches from src_rank to expert_rank, processes, then
    returns to src_rank. Cross-layer info CANNOT reduce this metric: each
    layer's cost depends only on (src_rank, expert_rank) for that layer.
    """
    dst = apply_placement(df, placement)
    src = df["src_rank"].to_numpy()
    return float((src != dst).mean())


def cross_rank_rate_pipelined(df: pd.DataFrame, placement: Dict[int, np.ndarray]) -> float:
    """Pipelined metric: token's location after layer L is expert_rank for that layer.

    Layer 0 dispatch: src_rank -> placement[0][e_0]
    Layer L>0 dispatch: placement[L-1][e_{L-1}] -> placement[L][e_L]

    Under this metric, cross-layer-aware placement can amortize hops: if
    placement(e_L+1) frequently equals placement(e_L) for tokens routed
    through (e_L, e_L+1), the L+1 dispatch is free.

    Requires df sorted by token_idx_in_run, then by layer.
    """
    df_sorted = df.sort_values(["token_idx_in_run", "layer"], kind="stable")
    tids = df_sorted["token_idx_in_run"].to_numpy()
    layers = df_sorted["layer"].to_numpy()
    experts = df_sorted["expert_id"].to_numpy()
    src = df_sorted["src_rank"].to_numpy()
    dst = apply_placement(df_sorted, placement)

    # prev_loc[i] = location of token tids[i] just before this dispatch.
    # For each token's first event (layer 0 or its lowest layer), prev_loc = src_rank.
    # Otherwise prev_loc = dst of the previous event for the same token.
    is_first = np.empty(len(tids), dtype=bool)
    is_first[0] = True
    is_first[1:] = tids[1:] != tids[:-1]

    prev_loc = np.empty_like(dst)
    prev_loc[is_first] = src[is_first]
    # Forward fill: for non-first events, prev_loc = dst at previous row (same token)
    prev_loc[~is_first] = dst[:-1][~is_first[1:]]  # pair up i with i-1 within token
    # Above line: if is_first[i] is False, then prev_loc[i] = dst[i-1]
    # We need to construct that more directly:
    prev_loc = np.where(is_first, src, np.concatenate([[0], dst[:-1]]))
    return float((prev_loc != dst).mean())


# Backwards-compatible alias used by the older runner.
cross_rank_rate = cross_rank_rate_dispatch_return


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

@dataclass
class PlacementResult:
    model: str
    domain: str
    ep_size: int
    n_experts: int
    n_events: int
    # Standard dispatch-and-return metric (cross-layer cannot help here)
    rate_random_dr: float
    rate_layer_freq_dr: float
    rate_trajectory_dr: float
    # Pipelined metric (cross-layer CAN help here)
    rate_random_pl: float
    rate_layer_freq_pl: float
    rate_trajectory_pl: float
    # Headline savings (under pipelined model, since that's where X lives)
    saving_layer_freq_vs_random_pl: float
    saving_trajectory_vs_random_pl: float
    incremental_trajectory_vs_layer_freq_pl: float


def run_all_strategies(
    df_top1: pd.DataFrame,
    n_experts: int,
    tokens_per_step: int,
    ep_size: int,
    seed: int = 42,
) -> PlacementResult:
    """``df_top1`` must contain rows where topk_rank == 0, with src_rank computed."""
    layers = sorted(df_top1["layer"].unique().tolist())

    p_rand = placement_random(layers, n_experts, ep_size, seed=seed)
    p_freq = placement_layer_frequency(df_top1, layers, n_experts, ep_size)
    p_traj = placement_trajectory(df_top1, layers, n_experts, ep_size)

    rate_rand_dr = cross_rank_rate_dispatch_return(df_top1, p_rand)
    rate_freq_dr = cross_rank_rate_dispatch_return(df_top1, p_freq)
    rate_traj_dr = cross_rank_rate_dispatch_return(df_top1, p_traj)
    rate_rand_pl = cross_rank_rate_pipelined(df_top1, p_rand)
    rate_freq_pl = cross_rank_rate_pipelined(df_top1, p_freq)
    rate_traj_pl = cross_rank_rate_pipelined(df_top1, p_traj)

    def _safe_rel(a: float, b: float) -> float:
        return float((a - b) / a) if a > 1e-12 else 0.0

    return PlacementResult(
        model="",
        domain="",
        ep_size=int(ep_size),
        n_experts=int(n_experts),
        n_events=int(len(df_top1)),
        rate_random_dr=rate_rand_dr,
        rate_layer_freq_dr=rate_freq_dr,
        rate_trajectory_dr=rate_traj_dr,
        rate_random_pl=rate_rand_pl,
        rate_layer_freq_pl=rate_freq_pl,
        rate_trajectory_pl=rate_traj_pl,
        saving_layer_freq_vs_random_pl=_safe_rel(rate_rand_pl, rate_freq_pl),
        saving_trajectory_vs_random_pl=_safe_rel(rate_rand_pl, rate_traj_pl),
        incremental_trajectory_vs_layer_freq_pl=_safe_rel(rate_freq_pl, rate_traj_pl),
    )
