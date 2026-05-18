"""H4 analysis driver — runs Jaccard + EAMC + hit-rate on H4 traces.

Loads per-conversation / per-prefix traces (from scripts 100 and 110),
injects (conversation_id, turn_id) and (prefix_id, branch_id) from
directory layout, and emits the headline H4 verdict.

Usage (post-trace-collection):
    python scripts/130_h4_analyze.py \
        --multiturn outputs/traces/h4_multiturn_qwen \
        --shared_prefix outputs/traces/h4_shared_prefix_qwen \
        --out_dir outputs/h4/qwen
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.jaccard_overlap import (  # noqa: E402
    analyze_multi_turn, analyze_shared_prefix, compute_h4_ratio,
)
from analysis.affinity_evaluation import evaluate_hit_rates, h4_verdict  # noqa: E402


# ---------------------------------------------------------------------------
# Trace loaders with ID injection from directory layout
# ---------------------------------------------------------------------------

def load_multiturn(trace_root: Path) -> pd.DataFrame:
    """Load all conv_NNNN/ subdirs; inject (conversation_id, turn_id)."""
    conv_dirs = sorted(d for d in trace_root.glob("conv_*") if d.is_dir())
    parts = []
    for d in conv_dirs:
        try:
            conv_id = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for shard in sorted(d.glob("shard_*.parquet")):
            df = pq.read_table(shard).to_pandas()
            df["conversation_id"] = conv_id
            df["turn_id"] = df["step"].astype(np.int32)  # step IS turn_id
            df["prefix_id"] = -1
            df["branch_id"] = -1
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No conv_*/shard_*.parquet under {trace_root}")
    return pd.concat(parts, ignore_index=True)


def load_shared_prefix(trace_root: Path) -> pd.DataFrame:
    """Load all prefix_NNNN/ subdirs; inject (prefix_id, branch_id)."""
    pref_dirs = sorted(d for d in trace_root.glob("prefix_*") if d.is_dir())
    parts = []
    for d in pref_dirs:
        try:
            pref_id = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for shard in sorted(d.glob("shard_*.parquet")):
            df = pq.read_table(shard).to_pandas()
            df["prefix_id"] = pref_id
            df["branch_id"] = df["step"].astype(np.int32)
            df["conversation_id"] = -1
            df["turn_id"] = -1
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No prefix_*/shard_*.parquet under {trace_root}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_main(mt, sp, h4, hr, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Jaccard comparison
    cats = ["multi-turn\nwithin\n(adjacent)", "multi-turn\nbetween", "shared-prefix\nwithin", "shared-prefix\nbetween"]
    vals = [mt.within_adjacent.mean, mt.between.mean, sp.within.mean, sp.between.mean]
    errs = [mt.within_adjacent.std, mt.between.std, sp.within.std, sp.between.std]
    colors = ["#D35F5F", "#9999AA", "#5A9BD3", "#9999AA"]
    ax1.bar(cats, vals, yerr=errs, color=colors, edgecolor="black", capsize=4)
    ax1.set_ylabel("Jaccard similarity of expert set")
    ax1.set_title(f"Expert-set Jaccard: multi-turn vs shared-prefix\nH4 ratio = {h4.ratio:.2f}")
    ax1.grid(alpha=0.3, axis="y")

    # Right: hit-rate comparison
    names = ["random", "eamc", "session_id"]
    hits = [hr[n].hit_rate_mean for n in names]
    stds = [hr[n].hit_rate_std for n in names]
    colors2 = ["#9999AA", "#5A9BD3", "#D35F5F"]
    ax2.bar(names, hits, yerr=stds, color=colors2, edgecolor="black", capsize=4)
    ax2.set_ylabel("Prefetch hit rate")
    ax2.set_title("Predictor hit rate (per-turn)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("H4 — Multi-turn expert stickiness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--multiturn", type=Path, required=True)
    p.add_argument("--shared_prefix", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="If set, compute per-layer J. Default: all layers union.")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_pct", type=float, default=0.3)
    p.add_argument("--eamc_capacity", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading multi-turn from {args.multiturn}...")
    df_mt = load_multiturn(args.multiturn)
    print(f"  {len(df_mt):,} routing events, "
          f"{df_mt['conversation_id'].nunique()} convs, "
          f"{df_mt.groupby('conversation_id')['turn_id'].nunique().mean():.1f} avg turns/conv")
    mt_meta = json.loads((args.multiturn / "run_meta.json").read_text())
    n_layers = int(mt_meta["num_moe_layers"])
    n_experts = int(mt_meta["num_routed_experts"])

    print(f"Loading shared-prefix from {args.shared_prefix}...")
    df_sp = load_shared_prefix(args.shared_prefix)
    print(f"  {len(df_sp):,} routing events, "
          f"{df_sp['prefix_id'].nunique()} prefixes, "
          f"{df_sp.groupby('prefix_id')['branch_id'].nunique().mean():.1f} branches/prefix")

    print("\n--- Jaccard: multi-turn ---")
    mt = analyze_multi_turn(df_mt, layer=args.layer, top_k=args.top_k, seed=args.seed)
    print(f"  J_within_adjacent = {mt.within_adjacent.mean:.3f} ± {mt.within_adjacent.std:.3f}")
    print(f"  J_within_all      = {mt.within_all.mean:.3f}")
    print(f"  J_between         = {mt.between.mean:.3f}")
    print(f"  within/between ratio = {mt.ratio_within_adj_over_between:.2f}×")

    print("\n--- Jaccard: shared-prefix ---")
    sp = analyze_shared_prefix(df_sp, layer=args.layer, top_k=args.top_k, seed=args.seed)
    print(f"  J_within  = {sp.within.mean:.3f} ± {sp.within.std:.3f}")
    print(f"  J_between = {sp.between.mean:.3f}")
    print(f"  within/between ratio = {sp.ratio_within_over_between:.2f}×")

    h4 = compute_h4_ratio(mt, sp)
    print(f"\n--- H4 headline ratio ---")
    print(f"  J_multi_turn / J_shared_prefix = {h4.ratio:.3f}")
    print(f"  → {h4.interpretation}")

    print("\n--- Hit-rate: session-ID vs EAMC vs random ---")
    hr = evaluate_hit_rates(df_mt, n_layers, n_experts,
                             top_k=args.top_k, top_pct=args.top_pct,
                             eamc_capacity=args.eamc_capacity, seed=args.seed)
    for name in ["random", "eamc", "session_id"]:
        r = hr[name]
        print(f"  {name:>12s}: hit_rate = {r.hit_rate_mean:.3f} ± {r.hit_rate_std:.3f} "
              f"(n={r.n_predictions}, pred={r.avg_predicted_set_size:.1f})")
    passed_c3, msg = h4_verdict(hr)
    print(f"  verdict: {msg}")

    # Three success criteria
    passed_c1 = h4.ratio >= 0.4
    passed_c2 = mt.ratio_within_adj_over_between >= 2.5

    all_pass = passed_c1 and passed_c2 and passed_c3
    verdict = "STRONG FINDING — paper viable" if all_pass else (
        "PARTIAL FINDING — see decision tree" if (passed_c1 + passed_c2 + passed_c3) >= 2
        else "DEAD — kill in research-wiki"
    )

    # Persist
    summary = {
        "verdict": verdict,
        "criteria_passed": {
            "C1_J_ratio_ge_0p4": bool(passed_c1),
            "C2_within_between_ge_2p5x": bool(passed_c2),
            "C3_session_id_ge_eamc_x_0p9": bool(passed_c3),
        },
        "multi_turn": dataclasses.asdict(mt),
        "shared_prefix": dataclasses.asdict(sp),
        "h4_ratio": dataclasses.asdict(h4),
        "hit_rates": {k: dataclasses.asdict(v) for k, v in hr.items()},
        "n_layers": n_layers, "n_experts": n_experts,
        "model": mt_meta.get("model"), "tag": mt_meta.get("tag"),
        "top_k": args.top_k, "top_pct": args.top_pct,
        "eamc_capacity": args.eamc_capacity,
    }
    (args.out_dir / "h4_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    plot_main(mt, sp, h4, hr, args.out_dir / "F_h4_main.png")

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  C1 (J_ratio ≥ 0.4):           {'PASS' if passed_c1 else 'FAIL'} (got {h4.ratio:.3f})")
    print(f"  C2 (within/between ≥ 2.5×):   {'PASS' if passed_c2 else 'FAIL'} (got {mt.ratio_within_adj_over_between:.2f})")
    print(f"  C3 (session ≥ EAMC × 0.9):    {'PASS' if passed_c3 else 'FAIL'}")
    print(f"\nWrote {args.out_dir / 'h4_summary.json'}")
    print(f"Wrote {args.out_dir / 'F_h4_main.png'}")
    return 0 if all_pass else (1 if (passed_c1 + passed_c2 + passed_c3) < 2 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
