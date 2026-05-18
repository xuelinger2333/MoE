"""Expert-set Jaccard overlap analysis for H4 multi-turn stickiness.

Three protocols share the same downstream Jaccard math, differing only in
how we group trace rows into "units" (turn / branch / conversation) and
how we pair units (within vs between):

    multi-turn within  : pairs of turns in the same conversation
    multi-turn between : pairs of turns from different conversations
    shared-prefix within : pairs of branches forked from the same prefix
    shared-prefix between: pairs of branches from different prefixes

The headline H4 number is the ratio
    J_multi_turn_within / J_shared_prefix_within

This module is independent of trace collection — it expects DataFrames
with extended schema:

    (token_idx_in_run, layer, expert_id, topk_rank, weight,
     conversation_id, turn_id, branch_id, prefix_id)

Conversation/turn IDs are populated by ``scripts/100_probe_multiturn.py``;
prefix/branch IDs by ``scripts/110_probe_shared_prefix.py``. Either set may
be NaN depending on the trace type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expert-set extraction
# ---------------------------------------------------------------------------

def expert_set_per_unit(
    df: pd.DataFrame,
    unit_cols: Sequence[str],
    layer: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Dict[Tuple, np.ndarray]:
    """Return ``unit_value -> sorted_unique_expert_ids`` mapping.

    Args:
        df: Trace DataFrame with at least ``layer``, ``expert_id``, ``topk_rank``,
            plus the columns in ``unit_cols``.
        unit_cols: e.g. ``("conversation_id", "turn_id")`` for multi-turn,
            or ``("prefix_id", "branch_id")`` for shared-prefix.
        layer: If given, restrict to one MoE layer (per-layer analysis).
        top_k: If given, only use top-k routes with ``topk_rank < top_k``.
            Default: use all available top-k slots (model's native top_k).
    """
    sub = df
    if layer is not None:
        sub = sub[sub["layer"] == layer]
    if top_k is not None:
        sub = sub[sub["topk_rank"] < top_k]

    out: Dict[Tuple, np.ndarray] = {}
    for key, grp in sub.groupby(list(unit_cols), sort=False):
        out[tuple(key) if isinstance(key, tuple) else (key,)] = (
            np.unique(grp["expert_id"].to_numpy())
        )
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / |A ∪ B|; returns 0 if both sets are empty."""
    if a.size == 0 and b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = np.union1d(a, b).size
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def adjacent_within_group_pairs(
    units: Dict[Tuple, np.ndarray],
    group_index: int = 0,
    order_index: int = 1,
) -> List[Tuple[Tuple, Tuple]]:
    """For multi-turn: pairs (turn_t, turn_{t+1}) within the same conversation.

    For shared-prefix: pairs of branches sharing the same prefix
    (use ``group_index=0`` = prefix_id, ``order_index=1`` = branch_id, and
    set ``adjacent_only=False`` via ``all_within_group_pairs`` instead).
    """
    by_group: Dict[Tuple, List[Tuple]] = {}
    for k in units:
        g = (k[group_index],)
        by_group.setdefault(g, []).append(k)
    pairs: List[Tuple[Tuple, Tuple]] = []
    for g, ks in by_group.items():
        # sort by order_index
        ks_sorted = sorted(ks, key=lambda k: k[order_index])
        for i in range(len(ks_sorted) - 1):
            pairs.append((ks_sorted[i], ks_sorted[i + 1]))
    return pairs


def all_within_group_pairs(
    units: Dict[Tuple, np.ndarray],
    group_index: int = 0,
    max_pairs_per_group: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[Tuple, Tuple]]:
    """All unordered pairs of units within the same group (e.g., branches sharing a prefix)."""
    rng = np.random.default_rng(seed)
    by_group: Dict[Tuple, List[Tuple]] = {}
    for k in units:
        g = (k[group_index],)
        by_group.setdefault(g, []).append(k)
    pairs: List[Tuple[Tuple, Tuple]] = []
    for g, ks in by_group.items():
        n = len(ks)
        all_pairs = [(ks[i], ks[j]) for i in range(n) for j in range(i + 1, n)]
        if max_pairs_per_group is not None and len(all_pairs) > max_pairs_per_group:
            idx = rng.choice(len(all_pairs), max_pairs_per_group, replace=False)
            all_pairs = [all_pairs[i] for i in idx]
        pairs.extend(all_pairs)
    return pairs


def between_group_pairs(
    units: Dict[Tuple, np.ndarray],
    group_index: int = 0,
    n_pairs: int = 1000,
    seed: int = 42,
) -> List[Tuple[Tuple, Tuple]]:
    """Random pairs across different groups (different conversations or different prefixes)."""
    rng = np.random.default_rng(seed)
    by_group: Dict[Tuple, List[Tuple]] = {}
    for k in units:
        g = (k[group_index],)
        by_group.setdefault(g, []).append(k)
    groups = sorted(by_group.keys())
    if len(groups) < 2:
        return []
    pairs: List[Tuple[Tuple, Tuple]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        g1, g2 = rng.choice(len(groups), 2, replace=False)
        u1 = by_group[groups[g1]][rng.integers(len(by_group[groups[g1]]))]
        u2 = by_group[groups[g2]][rng.integers(len(by_group[groups[g2]]))]
        pairs.append((u1, u2))
    return pairs


# ---------------------------------------------------------------------------
# Headline analyses
# ---------------------------------------------------------------------------

@dataclass
class JaccardStats:
    n_pairs: int
    mean: float
    median: float
    p10: float
    p90: float
    std: float


def _stats(jvalues: List[float]) -> JaccardStats:
    if not jvalues:
        return JaccardStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(jvalues, dtype=np.float64)
    return JaccardStats(
        n_pairs=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    )


def compute_jaccards(
    units: Dict[Tuple, np.ndarray],
    pairs: List[Tuple[Tuple, Tuple]],
) -> JaccardStats:
    js = [jaccard(units[a], units[b]) for a, b in pairs if a in units and b in units]
    return _stats(js)


@dataclass
class MultiTurnResult:
    n_conversations: int
    n_turns: int
    within_adjacent: JaccardStats   # turn_t vs turn_{t+1} same conv
    within_all: JaccardStats        # any-pair same conv
    between: JaccardStats           # random pair across conversations
    ratio_within_adj_over_between: float


def analyze_multi_turn(
    df: pd.DataFrame,
    layer: Optional[int] = None,
    top_k: Optional[int] = None,
    n_between_pairs: int = 2000,
    seed: int = 42,
) -> MultiTurnResult:
    """Compute headline within / between Jaccard for the multi-turn protocol.

    DataFrame is expected to have ``conversation_id`` and ``turn_id`` columns.
    """
    units = expert_set_per_unit(df, ("conversation_id", "turn_id"), layer=layer, top_k=top_k)
    n_conv = len(set(k[0] for k in units))
    n_turn = len(units)

    within_adj_pairs = adjacent_within_group_pairs(units, group_index=0, order_index=1)
    within_all_pairs = all_within_group_pairs(units, group_index=0, max_pairs_per_group=10, seed=seed)
    between_pairs = between_group_pairs(units, group_index=0, n_pairs=n_between_pairs, seed=seed)

    w_adj = compute_jaccards(units, within_adj_pairs)
    w_all = compute_jaccards(units, within_all_pairs)
    b = compute_jaccards(units, between_pairs)
    ratio = w_adj.mean / b.mean if b.mean > 1e-12 else float("inf")
    return MultiTurnResult(
        n_conversations=n_conv, n_turns=n_turn,
        within_adjacent=w_adj, within_all=w_all, between=b,
        ratio_within_adj_over_between=float(ratio),
    )


@dataclass
class SharedPrefixResult:
    n_prefixes: int
    n_branches: int
    within: JaccardStats           # all-pair within same prefix
    between: JaccardStats          # random pair across prefixes
    ratio_within_over_between: float


def analyze_shared_prefix(
    df: pd.DataFrame,
    layer: Optional[int] = None,
    top_k: Optional[int] = None,
    n_between_pairs: int = 2000,
    seed: int = 42,
) -> SharedPrefixResult:
    """Reproduce arXiv 2604.17182's shared-prefix Jaccard setting."""
    units = expert_set_per_unit(df, ("prefix_id", "branch_id"), layer=layer, top_k=top_k)
    n_pref = len(set(k[0] for k in units))
    n_br = len(units)

    within_pairs = all_within_group_pairs(units, group_index=0, max_pairs_per_group=20, seed=seed)
    between_pairs = between_group_pairs(units, group_index=0, n_pairs=n_between_pairs, seed=seed)

    w = compute_jaccards(units, within_pairs)
    b = compute_jaccards(units, between_pairs)
    ratio = w.mean / b.mean if b.mean > 1e-12 else float("inf")
    return SharedPrefixResult(
        n_prefixes=n_pref, n_branches=n_br,
        within=w, between=b,
        ratio_within_over_between=float(ratio),
    )


# ---------------------------------------------------------------------------
# Headline H4 ratio
# ---------------------------------------------------------------------------

@dataclass
class H4HeadlineRatio:
    j_multi_turn_within: float
    j_shared_prefix_within: float
    ratio: float
    interpretation: str


def compute_h4_ratio(mt: MultiTurnResult, sp: SharedPrefixResult) -> H4HeadlineRatio:
    """The headline number per the user's design: J_multi_turn / J_shared_prefix.

    Uses ``within_adjacent`` for multi-turn (most directly comparable) and
    ``within`` for shared-prefix.
    """
    j_mt = mt.within_adjacent.mean
    j_sp = sp.within.mean
    ratio = j_mt / j_sp if j_sp > 1e-12 else 0.0
    if ratio >= 0.5:
        interp = "STRONG: multi-turn locality approaches prefix-level without bit-exact prefix"
    elif ratio >= 0.4:
        interp = "MARGINAL-STRONG: meets the user-defined ≥0.4 success threshold"
    elif ratio >= 0.2:
        interp = "WEAK: signal exists but is much weaker than the shared-prefix baseline"
    elif ratio >= 1.0:
        interp = "ALARM: multi-turn signal is stronger than shared-prefix — verify protocol"
    else:
        interp = "DEAD: multi-turn signal is < 20% of shared-prefix; H4 system claim doubtful"
    return H4HeadlineRatio(
        j_multi_turn_within=float(j_mt),
        j_shared_prefix_within=float(j_sp),
        ratio=float(ratio),
        interpretation=interp,
    )
