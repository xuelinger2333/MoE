"""Exp 1 — Random expert permutation null for cross-layer MI.

For each (model, domain) trace, compute:
    * observed MI for adjacent layer pairs (d=1) and a long-range pair (d=8)
    * empirical iid-permutation null distribution (n=200 permutations)
    * gap = observed - null_q99  (signal that survives at 99% confidence)
    * bijection-invariance sanity check

Output:
    outputs/permutation_null/<run_tag>.json
    outputs/permutation_null/summary.json
    outputs/permutation_null/F_permutation_null.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.cross_layer_mi import top1_pivot  # noqa: E402
from analysis.permutation_null import (  # noqa: E402
    assess_all_pairs,
    summarize,
)


# ---------------------------------------------------------------------------
# Self-contained trace loader (no torch needed)
# ---------------------------------------------------------------------------

def load_trace_parquet(trace_dir: Path) -> pd.DataFrame:
    shards = sorted(trace_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {trace_dir}")
    return pd.concat([pq.read_table(p).to_pandas() for p in shards], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-trace pipeline
# ---------------------------------------------------------------------------

def analyze_trace(
    trace_dir: Path, label: str, distances: Tuple[int, ...], n_repeats: int, seed: int
) -> Dict:
    meta = json.loads((trace_dir / "run_meta.json").read_text())
    t0 = time.time()
    df = load_trace_parquet(trace_dir)
    print(f"  loaded {len(df):,} routing events ({time.time() - t0:.1f}s)")

    wide = top1_pivot(df)
    n_experts = meta["num_routed_experts"]
    print(f"  pivoted: {len(wide):,} tokens × {wide.shape[1]} layers, n_experts={n_experts}")

    t0 = time.time()
    def _progress(done: int, total: int) -> None:
        if done == total or done % 5 == 0:
            print(f"    pair {done}/{total} ({time.time() - t0:.1f}s)")
    rows = assess_all_pairs(
        wide, num_experts=n_experts, distances=distances,
        n_repeats=n_repeats, seed=seed, progress=_progress,
    )
    print(f"  null assessment done in {time.time() - t0:.1f}s")

    out = {
        "label": label,
        "trace_dir": str(trace_dir),
        "model": meta.get("model"),
        "tag": meta.get("tag"),
        "domain": meta.get("domain"),
        "num_experts": int(n_experts),
        "num_layers": int(meta["num_moe_layers"]),
        "top_k": int(meta["top_k"]),
        "n_tokens": int(rows[0]["n_tokens"]) if rows else 0,
        "n_repeats": int(n_repeats),
        "per_pair": rows,
        "summary_by_d": [summarize(rows, d) for d in distances],
    }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_observed_vs_null(results: List[Dict], out_path: Path) -> None:
    """Bar plot: observed MI vs null_q99 for each (model, domain), d=1."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        rows_d1 = [r for r in res["per_pair"] if r["d"] == 1]
        Ls = [r["L"] for r in rows_d1]
        obs = [r["observed_mi_nats"] for r in rows_d1]
        q99 = [r["null_q99"] for r in rows_d1]
        nmax = [r["null_max"] for r in rows_d1]
        ax.plot(Ls, obs, marker="o", lw=1.7, color="black", label="observed MI")
        ax.plot(Ls, q99, marker="s", lw=1.2, color="#d35f5f", label="null 99% upper bound (n=200)")
        ax.plot(Ls, nmax, marker="x", lw=0.8, ls=":", color="#888888", label="null max")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Source layer L")
        ax.set_ylabel("MI [nats]" if ax is axes[0] else "")
        ax.set_title(
            f"{res['tag']}/{res['domain']}\n"
            f"E={res['num_experts']}, top-{res['top_k']}, N={res['n_tokens']:,}"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Exp 1 — Observed MI vs random-permutation null (d=1, adjacent layers)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", nargs="+", type=Path, default=None,
                   help="trace directories. If omitted, default = all m8_* domain traces.")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/permutation_null")
    p.add_argument("--distances", type=int, nargs="+", default=[1, 8])
    p.add_argument("--n_repeats", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="sanity mode: 1 trace, n_repeats=20, d=1 only")
    args = p.parse_args()

    if args.smoke:
        args.traces = [REPO_ROOT / "outputs/traces/m8_qwen_code"]
        args.distances = [1]
        args.n_repeats = 20
        args.out_dir = REPO_ROOT / "outputs/permutation_null_smoke"

    if args.traces is None:
        args.traces = sorted((REPO_ROOT / "outputs/traces").glob("m8_*"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Exp 1: random-expert-permutation null ==")
    print(f"   traces: {len(args.traces)}")
    print(f"   distances: {args.distances}")
    print(f"   n_repeats: {args.n_repeats}")
    print(f"   out_dir: {args.out_dir}")

    results: List[Dict] = []
    for trace_dir in args.traces:
        if not (trace_dir / "run_meta.json").exists():
            print(f"  SKIP {trace_dir.name}: no run_meta.json")
            continue
        label = trace_dir.name
        print(f"\n[{label}]")
        res = analyze_trace(trace_dir, label, tuple(args.distances), args.n_repeats, args.seed)
        results.append(res)
        out_path = args.out_dir / f"{label}.json"
        out_path.write_text(json.dumps(res, indent=2, default=float))
        print(f"  wrote {out_path}")

    summary = []
    for res in results:
        for s in res["summary_by_d"]:
            if not s:
                continue
            summary.append({
                "tag": res["tag"], "domain": res["domain"], "n_tokens": res["n_tokens"],
                **s,
            })
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_dir / 'summary.json'}")

    if results:
        plot_observed_vs_null(results, args.out_dir / "F_permutation_null.png")
        print(f"Wrote {args.out_dir / 'F_permutation_null.png'}")

    print("\n=== Verdict ===")
    print(f"{'tag':>20s} {'domain':>5s} {'d':>2s} {'obs':>7s} {'null_q99':>9s} {'signal':>8s} {'norm_sig':>9s} {'retained':>9s} {'bij_ok':>7s}")
    for s in summary:
        print(f"{s['tag']:>20s} {s['domain']:>5s} {s['d']:>2d} "
              f"{s['obs_mean']:7.3f} {s['null_q99_mean']:9.4f} "
              f"{s['signal_above_q99_mean']:8.3f} {s['norm_signal_mean']*100:8.1f}% "
              f"{s['frac_signal_retained']*100:8.1f}% "
              f"{'OK' if s['bijection_invariance_holds'] else 'FAIL':>7s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
