"""Exp 2 — Placement simulator: how much cross-rank traffic can we save?

For each (model, domain, ep_size), compute cross-rank rate under:
    S0 random        : trivial baseline, asymptotic 1 - 1/ep
    S1 layer-freq    : per-layer optimal (Occult-style)
    S2 trajectory    : cross-layer aware (the proposed method)

Report:
    * absolute cross-rank rate per strategy
    * relative saving vs random
    * incremental saving of trajectory over layer-freq  (this is the X)

Output:
    outputs/placement_sim/<run_tag>_ep<EP>.json
    outputs/placement_sim/summary.csv
    outputs/placement_sim/F_placement_savings.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.placement_simulator import (  # noqa: E402
    compute_src_rank,
    run_all_strategies,
)


def load_trace_parquet(trace_dir: Path) -> pd.DataFrame:
    shards = sorted(trace_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {trace_dir}")
    return pd.concat([pq.read_table(p).to_pandas() for p in shards], ignore_index=True)


def analyze_trace(trace_dir: Path, ep_sizes: List[int], seed: int) -> List[Dict]:
    meta = json.loads((trace_dir / "run_meta.json").read_text())
    t0 = time.time()
    df = load_trace_parquet(trace_dir)
    print(f"  loaded {len(df):,} routing events ({time.time() - t0:.1f}s)")

    # Only top-1 events: simulator counts dispatch decisions, not all top-k slots
    df_top1 = df[df["topk_rank"] == 0].copy()
    print(f"  top-1 events: {len(df_top1):,}")

    n_experts = int(meta["num_routed_experts"])
    tokens_per_step = int(meta["tokens_per_step"])

    results: List[Dict] = []
    for ep in ep_sizes:
        if n_experts % ep != 0 or tokens_per_step % ep != 0:
            print(f"  SKIP ep={ep}: divisibility fails")
            continue
        t1 = time.time()
        df_top1["src_rank"] = compute_src_rank(
            df_top1["token_idx_in_run"].to_numpy(), tokens_per_step, ep
        )
        res = run_all_strategies(df_top1, n_experts, tokens_per_step, ep, seed=seed)
        res.model = meta.get("tag", trace_dir.name)
        res.domain = meta.get("domain", "?")
        out = dataclasses.asdict(res)
        out["meta"] = {
            "trace": trace_dir.name,
            "tokens_per_step": tokens_per_step,
            "num_layers": meta["num_moe_layers"],
            "top_k": meta["top_k"],
        }
        results.append(out)
        print(f"  ep={ep}:")
        print(f"    dispatch-return:  rand={res.rate_random_dr:.3f}  freq={res.rate_layer_freq_dr:.3f}  traj={res.rate_trajectory_dr:.3f}")
        print(f"    pipelined:        rand={res.rate_random_pl:.3f}  freq={res.rate_layer_freq_pl:.3f}  traj={res.rate_trajectory_pl:.3f}")
        print(f"    X (pipelined): traj saves {res.saving_trajectory_vs_random_pl*100:+.1f}% over rand, "
              f"{res.incremental_trajectory_vs_layer_freq_pl*100:+.1f}% incremental over freq "
              f"({time.time() - t1:.1f}s)")
    return results


def plot_savings(rows: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, (ax_dr, ax_pl) = plt.subplots(1, 2, figsize=(13, 5.0))
    eps = sorted(df["ep_size"].unique())
    models = sorted(df["model"].unique())
    width = 0.8 / max(1, len(models) * 3)
    colors = {"random": "#888888", "layer_freq": "#5A9BD3", "trajectory": "#D35F5F"}
    for ax, suffix, title in [
        (ax_dr, "_dr", "Dispatch-and-return (per-layer independent)"),
        (ax_pl, "_pl", "Pipelined (cross-layer exploits trajectory)"),
    ]:
        for i, m in enumerate(models):
            sub = df[df["model"] == m]
            agg = sub.groupby("ep_size").agg({
                f"rate_random{suffix}": "mean",
                f"rate_layer_freq{suffix}": "mean",
                f"rate_trajectory{suffix}": "mean",
            }).reset_index()
            for j, ep in enumerate(eps):
                row = agg[agg["ep_size"] == ep]
                if row.empty:
                    continue
                base_x = j + (i - (len(models) - 1) / 2) * 3 * width
                ax.bar(base_x + 0 * width, row[f"rate_random{suffix}"].iloc[0], width,
                       color=colors["random"], label="random" if i == 0 and j == 0 else None)
                ax.bar(base_x + 1 * width, row[f"rate_layer_freq{suffix}"].iloc[0], width,
                       color=colors["layer_freq"], label="layer-freq" if i == 0 and j == 0 else None)
                ax.bar(base_x + 2 * width, row[f"rate_trajectory{suffix}"].iloc[0], width,
                       color=colors["trajectory"], label="trajectory" if i == 0 and j == 0 else None)
                if suffix == "_dr":
                    ax.text(base_x + 1 * width, 0.02,
                            m, ha="center", fontsize=7, rotation=90, color="white")
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels([f"ep={e}" for e in eps])
        ax.set_ylabel("Cross-rank dispatch rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Exp 2 — Placement strategies (avg over domains)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", nargs="+", type=Path, default=None)
    p.add_argument("--ep_sizes", type=int, nargs="+", default=[2, 4])
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/placement_sim")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.traces = [REPO_ROOT / "outputs/traces/m8_qwen_code"]
        args.ep_sizes = [4]
        args.out_dir = REPO_ROOT / "outputs/placement_sim_smoke"

    if args.traces is None:
        args.traces = sorted((REPO_ROOT / "outputs/traces").glob("m8_*"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Exp 2: placement simulator ==")
    print(f"   traces: {len(args.traces)}")
    print(f"   ep_sizes: {args.ep_sizes}")

    all_rows: List[Dict] = []
    for trace_dir in args.traces:
        if not (trace_dir / "run_meta.json").exists():
            print(f"  SKIP {trace_dir.name}: no meta")
            continue
        print(f"\n[{trace_dir.name}]")
        rows = analyze_trace(trace_dir, args.ep_sizes, args.seed)
        all_rows.extend(rows)
        for r in rows:
            tag = f"{trace_dir.name}_ep{r['ep_size']}.json"
            (args.out_dir / tag).write_text(json.dumps(r, indent=2, default=float))

    if not all_rows:
        print("No results.")
        return 1
    df_summary = pd.DataFrame(all_rows)
    summary_path = args.out_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    plot_savings(all_rows, args.out_dir / "F_placement_savings.png")

    print("\n=== Verdict (pipelined metric: where cross-layer info matters) ===")
    cols = ["model", "domain", "ep_size", "rate_random_pl", "rate_layer_freq_pl",
            "rate_trajectory_pl", "saving_layer_freq_vs_random_pl",
            "saving_trajectory_vs_random_pl", "incremental_trajectory_vs_layer_freq_pl"]
    print(df_summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
