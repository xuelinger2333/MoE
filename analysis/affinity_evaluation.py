"""Hit-rate evaluation: session-ID affinity vs EAMC vs random.

For each conversation in a multi-turn trace, simulate prefetching the
predicted expert set for turn ``t`` based on what we know at the end of
turn ``t-1`` (or earlier). Score = fraction of expert activations at
turn ``t`` that fall in the predicted set (cache hit rate).

Three predictors compared:

  * **session_id**: predicted set = experts active in turn ``t-1`` of THIS conversation
  * **eamc**: predicted set = experts active in the EAMC entry whose flat
              vector is most similar to the conversation's running EAM
  * **random**: uniform sample

All three share the same prediction budget (``top_pct`` of experts per layer,
or union-across-layers — controlled per evaluation).

For H4 to be a publishable system claim:
    hit_rate(session_id) >= hit_rate(eamc) * 0.9   (within 10%)

If session_id beats eamc, H4 has a real differentiator (cheaper signal,
no online K-Means, no calibration set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.eamc_baseline import (
    EAMC,
    build_eam,
    predict_eamc_match,
    predict_random,
    predict_session_id,
)


@dataclass
class HitRateResult:
    predictor: str
    n_predictions: int
    hit_rate_mean: float
    hit_rate_std: float
    avg_predicted_set_size: float
    avg_actual_set_size: float


def turn_expert_set(
    df_turn: pd.DataFrame, n_layers: int, n_experts: int, top_k: Optional[int] = None,
) -> np.ndarray:
    """Set of all expert IDs activated during one turn (union across layers + tokens)."""
    sub = df_turn
    if top_k is not None:
        sub = sub[sub["topk_rank"] < top_k]
    return np.unique(sub["expert_id"].to_numpy()) if not sub.empty else np.array([], dtype=np.int64)


def per_layer_turn_expert_sets(
    df_turn: pd.DataFrame, n_layers: int, top_k: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Per-layer hot-expert sets in one turn."""
    sub = df_turn
    if top_k is not None:
        sub = sub[sub["topk_rank"] < top_k]
    out: Dict[int, np.ndarray] = {}
    for L, grp in sub.groupby("layer"):
        out[int(L)] = np.unique(grp["expert_id"].to_numpy())
    return out


def evaluate_hit_rates(
    multi_turn_df: pd.DataFrame,
    n_layers: int,
    n_experts: int,
    top_k: Optional[int] = None,
    top_pct: float = 0.3,
    eamc_capacity: int = 100,
    calibration_frac: float = 0.3,
    seed: int = 42,
) -> Dict[str, HitRateResult]:
    """Simulate session-ID, EAMC, and random predictors on a multi-turn trace.

    Pipeline:
      1. Split conversations into calibration (``calibration_frac``) and eval.
      2. Build EAMC offline from calibration set's turn-level EAMs.
      3. For each conversation in eval set:
         For each turn ``t >= 2``:
           - actual_set = experts in turn ``t``
           - session_pred = predict_session_id(turn t-1 EAM)
           - eamc_pred = predict_eamc_match(EAMC, running_eam_up_to_t-1)
           - random_pred = predict_random
           - hit_rate = |pred ∩ actual| / |actual|
      4. Average hit rates per predictor.
    """
    rng = np.random.default_rng(seed)
    convs = sorted(multi_turn_df["conversation_id"].unique().tolist())
    rng.shuffle(convs)
    n_calib = max(1, int(calibration_frac * len(convs)))
    calib_convs = set(convs[:n_calib])
    eval_convs = [c for c in convs if c not in calib_convs]

    # Build calibration EAMs: one per turn
    calib_eams: List[np.ndarray] = []
    for c in calib_convs:
        conv_df = multi_turn_df[multi_turn_df["conversation_id"] == c]
        if top_k is not None:
            conv_df = conv_df[conv_df["topk_rank"] < top_k]
        for t, grp in conv_df.groupby("turn_id"):
            calib_eams.append(build_eam(grp, n_layers, n_experts))

    eamc = EAMC.fit_offline(
        calib_eams, n_layers, n_experts, capacity=eamc_capacity, seed=seed,
    )

    session_hits: List[float] = []
    eamc_hits: List[float] = []
    random_hits: List[float] = []
    pred_sizes: Dict[str, List[int]] = {"session_id": [], "eamc": [], "random": []}
    actual_sizes: List[int] = []

    for c in eval_convs:
        conv_df = multi_turn_df[multi_turn_df["conversation_id"] == c]
        if top_k is not None:
            conv_df = conv_df[conv_df["topk_rank"] < top_k]
        turns = sorted(conv_df["turn_id"].unique().tolist())
        if len(turns) < 2:
            continue

        # Build per-turn DataFrames and EAMs once
        turn_dfs = {t: conv_df[conv_df["turn_id"] == t] for t in turns}
        turn_eams = {t: build_eam(turn_dfs[t], n_layers, n_experts) for t in turns}

        # Running EAM accumulator (for EAMC matching, simulates "what we've seen so far")
        running_eam = np.zeros((n_layers, n_experts), dtype=np.float32)

        for i, t in enumerate(turns):
            if i == 0:
                # Cold start: no prior turn to predict from. Skip.
                running_eam = turn_eams[t].copy()
                continue

            prev = turns[i - 1]
            actual = turn_expert_set(turn_dfs[t], n_layers, n_experts, top_k=top_k)
            if actual.size == 0:
                continue
            actual_sizes.append(int(actual.size))

            # Session-ID predictor: experts from last turn's EAM
            s_pred = predict_session_id(turn_eams[prev], top_pct=top_pct)
            # EAMC predictor: match running EAM against EAMC
            e_pred = predict_eamc_match(eamc, running_eam, top_pct=top_pct)
            # Random predictor (per-turn fresh seed)
            r_pred = predict_random(n_experts, top_pct=top_pct, seed=seed + 1000 * i)

            session_hits.append(_hit(actual, s_pred))
            eamc_hits.append(_hit(actual, e_pred))
            random_hits.append(_hit(actual, r_pred))
            pred_sizes["session_id"].append(s_pred.size)
            pred_sizes["eamc"].append(e_pred.size)
            pred_sizes["random"].append(r_pred.size)

            running_eam = running_eam + turn_eams[t]

    out: Dict[str, HitRateResult] = {}
    for name, hits in [("session_id", session_hits), ("eamc", eamc_hits), ("random", random_hits)]:
        if not hits:
            out[name] = HitRateResult(name, 0, 0.0, 0.0, 0.0, 0.0)
            continue
        arr = np.asarray(hits)
        out[name] = HitRateResult(
            predictor=name,
            n_predictions=int(arr.size),
            hit_rate_mean=float(arr.mean()),
            hit_rate_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            avg_predicted_set_size=float(np.mean(pred_sizes[name])) if pred_sizes[name] else 0.0,
            avg_actual_set_size=float(np.mean(actual_sizes)) if actual_sizes else 0.0,
        )
    return out


def _hit(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size == 0:
        return 0.0
    return float(np.intersect1d(actual, predicted, assume_unique=True).size / actual.size)


def h4_verdict(results: Dict[str, HitRateResult]) -> Tuple[bool, str]:
    """Per the user's success criterion: session_id >= eamc * 0.9."""
    s = results.get("session_id")
    e = results.get("eamc")
    if s is None or e is None or e.hit_rate_mean <= 1e-12:
        return False, "INSUFFICIENT DATA"
    ratio = s.hit_rate_mean / e.hit_rate_mean
    passed = ratio >= 0.9
    msg = (
        f"session_id hit_rate = {s.hit_rate_mean:.3f}, "
        f"eamc hit_rate = {e.hit_rate_mean:.3f}, "
        f"ratio = {ratio:.3f} "
        f"({'PASS' if passed else 'FAIL'} vs 0.9× threshold)"
    )
    return passed, msg
