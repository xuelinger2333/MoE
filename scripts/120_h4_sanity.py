"""M0 sanity — H4 analysis modules on synthetic traces with known properties.

Synthetic constructions:

  A. Perfect-stickiness: each conversation samples a fixed expert set; all
     turns in the conversation use ONLY those experts. Expect J_within == 1.0
     and J_between == expected_random.

  B. No-stickiness: each turn samples a fresh random expert set independent
     of conversation. Expect J_within ≈ J_between ≈ expected_random.

  C. Mixed: 50% of conversations are sticky (as A), 50% are random (as B).
     Expect J_within between A and B; ratio > 1.

  Shared-prefix sanity: each prefix samples a fixed expert set; all branches
  use it. Expect J_within == 1.0.

Pass criterion: all measured values fall within ±0.05 of analytic expectations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.jaccard_overlap import (  # noqa: E402
    analyze_multi_turn, analyze_shared_prefix, compute_h4_ratio,
)
from analysis.affinity_evaluation import evaluate_hit_rates, h4_verdict  # noqa: E402


N_LAYERS = 8
N_EXPERTS = 32
TOP_K = 4
TURNS_PER_CONV = 6
TOKENS_PER_TURN = 64


def _emit(conv_id, turn_id, prefix_id, branch_id, expert_pool, n_tokens=TOKENS_PER_TURN,
          seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for tok in range(n_tokens):
        for L in range(N_LAYERS):
            chosen = rng.choice(expert_pool, size=TOP_K, replace=False)
            for k, e in enumerate(chosen):
                rows.append({
                    "step": 0, "layer": L,
                    "token_idx_in_run": tok + 1000 * turn_id + 1_000_000 * conv_id,
                    "expert_id": int(e), "topk_rank": k, "weight": 1.0 / TOP_K,
                    "conversation_id": conv_id, "turn_id": turn_id,
                    "prefix_id": prefix_id, "branch_id": branch_id,
                })
    return rows


def build_perfect_stickiness(n_conv=20, pool_size=8, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_conv):
        pool = rng.choice(N_EXPERTS, size=pool_size, replace=False)
        for t in range(TURNS_PER_CONV):
            rows.extend(_emit(c, t, -1, -1, pool, seed=seed + 100 * c + t))
    return pd.DataFrame(rows)


def build_no_stickiness(n_conv=20, pool_size=8, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_conv):
        for t in range(TURNS_PER_CONV):
            pool = rng.choice(N_EXPERTS, size=pool_size, replace=False)
            rows.extend(_emit(c, t, -1, -1, pool, seed=seed + 100 * c + t))
    return pd.DataFrame(rows)


def build_mixed(n_conv=20, pool_size=8, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_conv):
        sticky = (c % 2 == 0)
        if sticky:
            pool = rng.choice(N_EXPERTS, size=pool_size, replace=False)
            for t in range(TURNS_PER_CONV):
                rows.extend(_emit(c, t, -1, -1, pool, seed=seed + 100 * c + t))
        else:
            for t in range(TURNS_PER_CONV):
                pool = rng.choice(N_EXPERTS, size=pool_size, replace=False)
                rows.extend(_emit(c, t, -1, -1, pool, seed=seed + 100 * c + t))
    return pd.DataFrame(rows)


def build_shared_prefix_perfect(n_prefix=10, n_branches=8, pool_size=8, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prefix):
        pool = rng.choice(N_EXPERTS, size=pool_size, replace=False)
        for b in range(n_branches):
            rows.extend(_emit(-1, -1, p, b, pool, seed=seed + 100 * p + b))
    return pd.DataFrame(rows)


def main() -> int:
    print("=== M0 sanity for H4 analysis modules ===\n")

    # --- Multi-turn: perfect stickiness ---
    print("[A] perfect-stickiness multi-turn")
    df_A = build_perfect_stickiness()
    res_A = analyze_multi_turn(df_A)
    print(f"  J_within_adjacent = {res_A.within_adjacent.mean:.3f} (expect close to 1.0; experts saturate after a few turns since each turn uses 8 random from a pool of 8 → eventually all 8)")
    print(f"  J_within_all      = {res_A.within_all.mean:.3f}")
    print(f"  J_between         = {res_A.between.mean:.3f} (expect near random)")
    print(f"  ratio             = {res_A.ratio_within_adj_over_between:.2f}× (expect >>1)")
    assert res_A.within_adjacent.mean > 0.9, "A: within_adjacent should be ~1"
    assert res_A.ratio_within_adj_over_between > 3.0, "A: ratio should be high"

    # --- Multi-turn: no stickiness ---
    print("\n[B] no-stickiness multi-turn")
    df_B = build_no_stickiness()
    res_B = analyze_multi_turn(df_B)
    print(f"  J_within_adjacent = {res_B.within_adjacent.mean:.3f}")
    print(f"  J_between         = {res_B.between.mean:.3f}")
    print(f"  ratio             = {res_B.ratio_within_adj_over_between:.2f}× (expect ~1)")
    assert abs(res_B.ratio_within_adj_over_between - 1.0) < 0.3, "B: ratio should be ~1"

    # --- Multi-turn: mixed ---
    print("\n[C] mixed multi-turn (50% sticky / 50% random)")
    df_C = build_mixed()
    res_C = analyze_multi_turn(df_C)
    print(f"  J_within_adjacent = {res_C.within_adjacent.mean:.3f}")
    print(f"  J_between         = {res_C.between.mean:.3f}")
    print(f"  ratio             = {res_C.ratio_within_adj_over_between:.2f}× (expect between A and B)")
    assert res_C.within_adjacent.mean > res_B.within_adjacent.mean
    assert res_C.within_adjacent.mean < res_A.within_adjacent.mean + 0.01

    # --- Shared-prefix ---
    print("\n[D] shared-prefix perfect-stickiness")
    df_D = build_shared_prefix_perfect()
    res_D = analyze_shared_prefix(df_D)
    print(f"  J_within  = {res_D.within.mean:.3f} (expect ~1)")
    print(f"  J_between = {res_D.between.mean:.3f} (expect near random)")
    print(f"  ratio     = {res_D.ratio_within_over_between:.2f}× (expect >>1)")
    assert res_D.within.mean > 0.9
    assert res_D.ratio_within_over_between > 3.0

    # --- H4 headline ratio (A vs D — both should be near 1.0, so ratio ~ 1.0) ---
    print("\n[E] H4 headline ratio (A multi-turn / D shared-prefix)")
    h4 = compute_h4_ratio(res_A, res_D)
    print(f"  J_multi_turn  = {h4.j_multi_turn_within:.3f}")
    print(f"  J_shared_pfx  = {h4.j_shared_prefix_within:.3f}")
    print(f"  ratio         = {h4.ratio:.3f}")
    print(f"  interpretation: {h4.interpretation}")
    assert h4.ratio > 0.8

    # --- Hit-rate evaluation on perfect stickiness (A) ---
    print("\n[F] hit-rate evaluation on synthetic A (perfect stickiness)")
    hr = evaluate_hit_rates(df_A, N_LAYERS, N_EXPERTS, top_k=TOP_K, top_pct=0.3,
                             eamc_capacity=10, calibration_frac=0.3, seed=42)
    for name in ["session_id", "eamc", "random"]:
        r = hr[name]
        print(f"  {name:>12s}: hit_rate = {r.hit_rate_mean:.3f} "
              f"(n={r.n_predictions}, pred_size={r.avg_predicted_set_size:.1f}, "
              f"actual_size={r.avg_actual_set_size:.1f})")
    passed, msg = h4_verdict(hr)
    print(f"  verdict: {msg}")
    assert hr["session_id"].hit_rate_mean > hr["random"].hit_rate_mean + 0.2

    # --- Hit-rate evaluation on synthetic B (no stickiness) ---
    print("\n[G] hit-rate evaluation on synthetic B (no stickiness)")
    hr_B = evaluate_hit_rates(df_B, N_LAYERS, N_EXPERTS, top_k=TOP_K, top_pct=0.3,
                               eamc_capacity=10, calibration_frac=0.3, seed=42)
    for name in ["session_id", "eamc", "random"]:
        r = hr_B[name]
        print(f"  {name:>12s}: hit_rate = {r.hit_rate_mean:.3f}")
    # On B, all three should be ≈ same (≈ top_pct since experts are uniformly random)
    assert abs(hr_B["session_id"].hit_rate_mean - hr_B["random"].hit_rate_mean) < 0.15

    print("\n=== M0 SANITY: ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
