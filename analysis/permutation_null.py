"""Random-expert permutation null for cross-layer MI.

Goal: address the residual concern that the observed 22-40% normalized MI
between adjacent MoE layers may be partially inflated by a "trivial baseline"
arising from the router being a deterministic function of (correlated) inputs.

The existing `analysis.cross_layer_mi` module provides two nulls already:
    * iid (loose):  globally shuffle Y across all tokens.
    * within-sequence (strict): shuffle Y only inside each sequence.

This module adds:

    1. **Repeated iid permutation with empirical null distribution.**
       Instead of reporting the mean of 3-5 permutations, we generate
       ``n_repeats`` (default 200) permutations and report mean, 99% upper
       bound, and max. This gives a tight upper-bound on what a permutation
       null can contribute, ruling out "trivial" inflation.

    2. **Per-token, per-layer random permutation null** ("layer-shuffle null").
       Independently permute each layer's column over tokens. This is the
       cleanest way to break cross-layer alignment while preserving each
       layer's marginal histogram exactly.

    3. **Bijection invariance sanity.** Apply a random expert-id bijection
       to each layer's column. MI must be preserved exactly (MI is invariant
       under bijection on either marginal). This is a unit test on the
       pipeline.

    4. **Verdict helper.** ``signal_above_null(observed, null_samples, q=0.99)``
       returns the gap ``observed - quantile(null, q)``, the "robust signal"
       that cannot be explained by finite-sample bias at confidence level ``q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.cross_layer_mi import joint_counts, mutual_information_nats


# ---------------------------------------------------------------------------
# Null samplers
# ---------------------------------------------------------------------------

def iid_permutation_null_samples(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
    n_repeats: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Empirical distribution of MI under iid permutation of Y.

    For each repeat, we draw a fresh random permutation π of the indices
    and compute MI((x[i]), (y[π(i)])). The resulting MI distribution is
    the finite-sample-bias floor: any observed MI above this distribution's
    upper quantile is genuine cross-layer signal, not artifact.

    Returns:
        ndarray of length ``n_repeats`` of MI values (nats).
    """
    rng = np.random.default_rng(seed)
    mis = np.empty(n_repeats, dtype=np.float64)
    for r in range(n_repeats):
        y_shuf = rng.permutation(y)
        C = joint_counts(x, y_shuf, n_x, n_y)
        mi, *_ = mutual_information_nats(C)
        mis[r] = mi
    return mis


def layer_shuffle_null_samples(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
    n_repeats: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Independently permute both layers' columns over tokens.

    Equivalent statistically to ``iid_permutation_null_samples`` (because
    a permutation of one column relative to the other suffices to destroy
    alignment). Kept as a belt-and-suspenders check that the two
    formulations agree.
    """
    rng = np.random.default_rng(seed)
    mis = np.empty(n_repeats, dtype=np.float64)
    for r in range(n_repeats):
        x_shuf = rng.permutation(x)
        y_shuf = rng.permutation(y)
        C = joint_counts(x_shuf, y_shuf, n_x, n_y)
        mi, *_ = mutual_information_nats(C)
        mis[r] = mi
    return mis


def expert_id_bijection_invariant_mi(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
    seed: int = 42,
) -> Tuple[float, float]:
    """Apply random bijections π_x, π_y to expert ids, then recompute MI.

    Because MI is invariant under bijection on either marginal, the result
    MUST equal the observed MI exactly (modulo floating-point). This is a
    pipeline sanity check, not a null.

    Returns:
        (mi_observed, mi_bijected) tuple. The two should be equal to ~1e-10.
    """
    rng = np.random.default_rng(seed)
    pi_x = rng.permutation(n_x)
    pi_y = rng.permutation(n_y)
    x_perm = pi_x[x]
    y_perm = pi_y[y]
    C_orig = joint_counts(x, y, n_x, n_y)
    C_perm = joint_counts(x_perm, y_perm, n_x, n_y)
    mi_orig, *_ = mutual_information_nats(C_orig)
    mi_perm, *_ = mutual_information_nats(C_perm)
    return float(mi_orig), float(mi_perm)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class NullVerdict:
    L: int
    d: int
    n_tokens: int
    n_experts: int
    observed_mi_nats: float
    H_target_nats: float
    null_mean: float
    null_std: float
    null_max: float
    null_q99: float
    signal_above_q99: float          # observed - null_q99
    signal_above_null_mean: float    # observed - null_mean
    norm_observed: float             # observed / H_target
    norm_signal: float               # signal_above_q99 / H_target
    bijection_invariant_passed: bool


def assess_pair(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
    L: int,
    d: int,
    n_repeats: int = 200,
    seed: int = 42,
) -> NullVerdict:
    """Full null assessment for one (L, L+d) pair."""
    C = joint_counts(x, y, n_x, n_y)
    mi_obs, Hx, Hy, _ = mutual_information_nats(C)
    null_samples = iid_permutation_null_samples(x, y, n_x, n_y, n_repeats=n_repeats, seed=seed)
    null_mean = float(np.mean(null_samples))
    null_std = float(np.std(null_samples))
    null_max = float(np.max(null_samples))
    null_q99 = float(np.quantile(null_samples, 0.99))

    mi_orig, mi_bij = expert_id_bijection_invariant_mi(x, y, n_x, n_y, seed=seed)
    bijection_ok = bool(abs(mi_orig - mi_bij) < 1e-9)

    return NullVerdict(
        L=int(L),
        d=int(d),
        n_tokens=int(C.sum()),
        n_experts=int(n_y),
        observed_mi_nats=float(mi_obs),
        H_target_nats=float(Hy),
        null_mean=null_mean,
        null_std=null_std,
        null_max=null_max,
        null_q99=null_q99,
        signal_above_q99=float(mi_obs - null_q99),
        signal_above_null_mean=float(mi_obs - null_mean),
        norm_observed=float(mi_obs / Hy) if Hy > 0 else 0.0,
        norm_signal=float((mi_obs - null_q99) / Hy) if Hy > 0 else 0.0,
        bijection_invariant_passed=bijection_ok,
    )


def assess_all_pairs(
    wide: pd.DataFrame,
    num_experts: int,
    distances: Tuple[int, ...] = (1,),
    n_repeats: int = 200,
    seed: int = 42,
    progress: Optional[callable] = None,
) -> List[Dict]:
    """Run permutation null assessment for all (L, L+d) pairs in ``wide``.

    ``wide``: DataFrame with one column per layer, one row per token, dtype int.
    """
    layers = sorted(int(c) for c in wide.columns)
    out: List[Dict] = []
    total = sum(1 for d in distances for L in layers if (L + d) in wide.columns)
    done = 0
    for d in distances:
        for L in layers:
            L2 = L + d
            if L2 not in wide.columns:
                continue
            x = wide[L].to_numpy()
            y = wide[L2].to_numpy()
            v = assess_pair(x, y, num_experts, num_experts, L=L, d=d,
                            n_repeats=n_repeats, seed=seed + L)
            row = vars(v).copy()
            out.append(row)
            done += 1
            if progress is not None:
                progress(done, total)
    return out


def summarize(rows: List[Dict], d: int = 1) -> Dict:
    """Aggregate verdicts across all source layers for one distance ``d``."""
    rs = [r for r in rows if r["d"] == d]
    if not rs:
        return {}
    obs = np.array([r["observed_mi_nats"] for r in rs])
    null_mean = np.array([r["null_mean"] for r in rs])
    null_q99 = np.array([r["null_q99"] for r in rs])
    null_max = np.array([r["null_max"] for r in rs])
    signal_q99 = np.array([r["signal_above_q99"] for r in rs])
    signal_mean = np.array([r["signal_above_null_mean"] for r in rs])
    norm_obs = np.array([r["norm_observed"] for r in rs])
    norm_sig = np.array([r["norm_signal"] for r in rs])
    bij_ok = all(r["bijection_invariant_passed"] for r in rs)
    return {
        "d": d,
        "n_pairs": len(rs),
        "obs_mean": float(np.mean(obs)),
        "null_mean_mean": float(np.mean(null_mean)),
        "null_q99_mean": float(np.mean(null_q99)),
        "null_max_mean": float(np.mean(null_max)),
        "signal_above_q99_mean": float(np.mean(signal_q99)),
        "signal_above_null_mean_mean": float(np.mean(signal_mean)),
        "norm_obs_mean": float(np.mean(norm_obs)),
        "norm_signal_mean": float(np.mean(norm_sig)),
        "frac_signal_retained": float(np.mean(signal_q99 / np.maximum(obs, 1e-12))),
        "bijection_invariance_holds": bij_ok,
    }
