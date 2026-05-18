"""G6 — Cross-layer routing MI on Qwen / DeepSeek / OLMoE traces.

Round-2: adds (a) within-sequence shuffle STRICT null to control for
sequence-level locality, (b) per-source entropy-reduction histogram to
test whether predictability is universal vs cherry-picked, (c) optional
OLMoE input for triangulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.cross_layer_mi import (
    all_layer_pairs_mi,
    joint_counts,
    per_source_entropy_reduction,
    per_source_kl,
    top1_pivot,
)
from src.probes.trace_writer import load_trace
from src.utils import get_logger

logger = get_logger("cross_layer_mi")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_sequence_ids(wide: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Group rows of ``wide`` (token_idx_in_run is the index) into sequences of ``seq_len``."""
    token_idx = wide.index.to_numpy()
    return (token_idx // seq_len).astype(np.int32)


def analyze(trace_dir: Path, label: str) -> dict:
    meta = json.loads((trace_dir / "run_meta.json").read_text())
    df = load_trace(trace_dir)
    logger.info(f"[{label}] loaded {len(df):,} routing events from {trace_dir.name}")

    wide = top1_pivot(df)
    seq_ids = _make_sequence_ids(wide, meta["seq_len"] if "seq_len" in meta else 1024)
    logger.info(
        f"[{label}] pivoted to {len(wide):,} tokens × {wide.shape[1]} layers; "
        f"{len(np.unique(seq_ids))} sequences (seq_len={meta.get('seq_len', 1024)})"
    )

    rows = all_layer_pairs_mi(
        wide,
        num_experts=meta["num_routed_experts"],
        sequence_ids=seq_ids,
        distances=(1, 2, 4, 8),
        null_repeats=3,
    )

    # Per-source entropy reduction across ALL adjacent pairs (d=1)
    n_e = meta["num_routed_experts"]
    H_marg_uniform = float(np.log(n_e))
    all_reductions: List[np.ndarray] = []
    all_eff_y: List[np.ndarray] = []
    n_layers = meta["num_moe_layers"]
    for L in range(n_layers - 1):
        if L not in wide.columns or (L + 1) not in wide.columns:
            continue
        x = wide[L].to_numpy()
        y = wide[L + 1].to_numpy()
        C = joint_counts(x, y, n_e, n_e)
        rep = per_source_entropy_reduction(C)
        red = rep["reduction"]
        eff = rep["effective_y"]
        # Drop unobserved sources (NaN)
        mask = ~np.isnan(red)
        all_reductions.append(red[mask])
        all_eff_y.append(eff[mask])
    flat_red = np.concatenate(all_reductions) if all_reductions else np.array([])
    flat_eff = np.concatenate(all_eff_y) if all_eff_y else np.array([])

    pred_share_strong = float((flat_red >= 0.5).mean()) if flat_red.size else 0.0
    pred_share_weak = float((flat_red >= 0.2).mean()) if flat_red.size else 0.0

    # Headline summary
    rows_d1 = [r for r in rows if r["d"] == 1]
    summary = {
        "label": label,
        "num_routed_experts": meta["num_routed_experts"],
        "num_layers": meta["num_moe_layers"],
        "top_k": meta["top_k"],
        "n_tokens": int(rows[0]["n_tokens"]) if rows else 0,
        "n_sequences": int(len(np.unique(seq_ids))),
        "H_marginal_uniform_nats": H_marg_uniform,
        # MI
        "d1_mi_mean_nats": float(np.mean([r["MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        # i.i.d. null (loose)
        "d1_null_iid_mean_nats": float(np.mean([r["null_iid_MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        "d1_mi_minus_null_iid_mean_nats": float(np.mean([r["MI_minus_null_iid"] for r in rows_d1])) if rows_d1 else 0.0,
        # within-sequence null (strict)
        "d1_null_within_seq_mean_nats": float(np.mean([r["null_within_seq_MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        "d1_mi_minus_null_within_seq_mean_nats": float(np.mean([r["MI_minus_null_within_seq"] for r in rows_d1])) if rows_d1 else 0.0,
        "d1_mi_minus_null_within_seq_max_nats": float(np.max([r["MI_minus_null_within_seq"] for r in rows_d1])) if rows_d1 else 0.0,
        # per-source distribution
        "n_source_pairs_observed": int(flat_red.size),
        "frac_sources_reduce_ge_0p2_nat": pred_share_weak,
        "frac_sources_reduce_ge_0p5_nat": pred_share_strong,
        "median_source_reduction_nats": float(np.median(flat_red)) if flat_red.size else 0.0,
        "p90_source_reduction_nats": float(np.percentile(flat_red, 90)) if flat_red.size else 0.0,
        "median_effective_y": float(np.median(flat_eff)) if flat_eff.size else 0.0,
        # for plotting
        "_per_pair_rows": rows,
        "_flat_reductions": flat_red.tolist(),
        "_flat_effective_y": flat_eff.tolist(),
    }
    return summary


def plot_strict_vs_loose(summaries: list[dict], out_path: Path) -> None:
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        rows_d1 = [r for r in s["_per_pair_rows"] if r["d"] == 1]
        Ls = [r["L"] for r in rows_d1]
        mi = [r["MI_nats"] for r in rows_d1]
        loose = [r["MI_minus_null_iid"] for r in rows_d1]
        strict = [r["MI_minus_null_within_seq"] for r in rows_d1]
        ax.plot(Ls, mi, marker="o", linewidth=1.6, color="black", label="raw MI(L, L+1)")
        ax.plot(Ls, loose, marker="s", linewidth=1.4, color="#5A9BD3",
                label="MI − i.i.d. null (loose)")
        ax.plot(Ls, strict, marker="^", linewidth=1.6, color="#D35F5F",
                label="MI − within-seq null (strict)")
        ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8, label="strong threshold (0.3 nat)")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_xlabel("Source layer L")
        ax.set_ylabel("MI [nats]")
        ax.set_title(
            f"{s['label']}  ({s['num_routed_experts']} expert, top-{s['top_k']}, {s['num_layers']} layer)"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("F11 — Loose (i.i.d.) vs strict (within-sequence) null at d=1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_decay_with_strict(summaries: list[dict], out_path: Path) -> None:
    """Show how MI − strict_null evolves with layer distance d."""
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        rows = s["_per_pair_rows"]
        by_d = {}
        for r in rows:
            by_d.setdefault(r["d"], []).append(r)
        for d in sorted(by_d):
            xs = [r["L"] for r in by_d[d]]
            ys = [r["MI_minus_null_within_seq"] for r in by_d[d]]
            ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"d={d}")
        ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_xlabel("Source layer L")
        ax.set_ylabel("MI(L, L+d) − within-seq null  [nats]")
        ax.set_title(f"{s['label']}  ({s['num_routed_experts']} experts, top-{s['top_k']})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("F9b — Cross-layer MI vs strict null, by distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_source_histogram(summaries: list[dict], out_path: Path) -> None:
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        red = np.asarray(s["_flat_reductions"])
        if red.size == 0:
            continue
        ax.hist(red, bins=40, color="#5A9BD3", edgecolor="white")
        ax.axvline(0.2, color="orange", linestyle="--", linewidth=1, label="weak threshold (0.2 nat)")
        ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1, label="strong threshold (0.5 nat)")
        median = float(np.median(red))
        ax.axvline(median, color="black", linewidth=1, label=f"median = {median:.2f}")
        ax.set_xlabel("H(P(e_{L+1})) − H(P(e_{L+1} | e_L = i))   [nats]")
        ax.set_ylabel("count of (L, source_expert) pairs")
        n_strong = int((red >= 0.5).sum())
        n_weak = int((red >= 0.2).sum())
        ax.set_title(
            f"{s['label']}\n"
            f"{red.size} (L, source) pairs;  "
            f"≥0.2 nat: {n_weak} ({n_weak/red.size*100:.0f}%);  "
            f"≥0.5 nat: {n_strong} ({n_strong/red.size*100:.0f}%)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("F12 — Per-source entropy reduction histogram (all adjacent pairs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_effective_y_cdf(summaries: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for s in summaries:
        eff = np.sort(np.asarray(s["_flat_effective_y"]))
        if eff.size == 0:
            continue
        cdf = np.arange(1, eff.size + 1) / eff.size
        ax.plot(eff, cdf, label=f"{s['label']}  (marg. eff. = {s['num_routed_experts']})", linewidth=1.6)
    ax.set_xlabel("Effective number of next-layer experts  exp(H(e_{L+1} | e_L = i))")
    ax.set_ylabel("CDF over (L, source_expert) pairs")
    ax.set_title("F13 — How many next-layer experts each source 'narrows down to'")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen_dir", type=Path,
                   default=REPO_ROOT / "outputs/traces/m4_qwen_wikitext")
    p.add_argument("--deepseek_dir", type=Path,
                   default=REPO_ROOT / "outputs/traces/m1_deepseek_wikitext")
    p.add_argument("--olmoe_dir", type=Path,
                   default=REPO_ROOT / "outputs/traces/m7_olmoe_wikitext")
    p.add_argument("--fig_dir", type=Path,
                   default=REPO_ROOT / "outputs/figures")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for trace_dir, label in [
        (args.qwen_dir, "Qwen1.5-MoE-A2.7B"),
        (args.deepseek_dir, "DeepSeek-V2-Lite"),
        (args.olmoe_dir, "OLMoE-1B-7B"),
    ]:
        if trace_dir.exists() and (trace_dir / "run_meta.json").exists():
            s = analyze(trace_dir, label)
            summaries.append(s)
        else:
            logger.warning(f"Skipping {label}: trace dir {trace_dir} missing")

    if not summaries:
        logger.error("No traces found. Aborting.")
        return 1

    plot_strict_vs_loose(summaries, args.fig_dir / "F11_strict_vs_loose_null.png")
    plot_decay_with_strict(summaries, args.fig_dir / "F9b_decay_strict_null.png")
    plot_source_histogram(summaries, args.fig_dir / "F12_per_source_histogram.png")
    plot_effective_y_cdf(summaries, args.fig_dir / "F13_effective_experts_cdf.png")

    # Strip large arrays from JSON output
    out = []
    for s in summaries:
        s2 = {k: v for k, v in s.items() if not k.startswith("_")}
        s2["per_pair_rows"] = s["_per_pair_rows"]
        out.append(s2)
    (args.fig_dir / "cross_layer_mi.json").write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote {args.fig_dir / 'cross_layer_mi.json'}")

    print()
    for s in summaries:
        loose = s["d1_mi_minus_null_iid_mean_nats"]
        strict = s["d1_mi_minus_null_within_seq_mean_nats"]
        verdict_loose = "STRONG" if loose >= 0.3 else ("WEAK" if loose >= 0.1 else "FAIL")
        verdict_strict = "STRONG" if strict >= 0.3 else ("WEAK" if strict >= 0.1 else "FAIL")
        print(f"=== {s['label']} (top-{s['top_k']}, {s['num_routed_experts']} experts, {s['num_layers']} layers) ===")
        print(f"  d=1 raw MI               = {s['d1_mi_mean_nats']:.3f} nat")
        print(f"  d=1 i.i.d. null (loose)  = {s['d1_null_iid_mean_nats']:.3f} nat")
        print(f"  d=1 within-seq null (strict) = {s['d1_null_within_seq_mean_nats']:.3f} nat")
        print(f"  d=1 MI − loose null      = {loose:.3f} nat   → verdict-loose: {verdict_loose}")
        print(f"  d=1 MI − strict null     = {strict:.3f} nat   → verdict-strict: {verdict_strict}")
        print(f"  per-source: median reduction = {s['median_source_reduction_nats']:.3f} nat,"
              f" frac≥0.2 = {s['frac_sources_reduce_ge_0p2_nat']*100:.1f}%,"
              f" frac≥0.5 = {s['frac_sources_reduce_ge_0p5_nat']*100:.1f}%")
        print(f"  median effective experts  = {s['median_effective_y']:.1f} (out of {s['num_routed_experts']})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
