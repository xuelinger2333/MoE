"""Compare DeepSeek-V2-Lite vs Qwen1.5-MoE on the C1/C2 claims.

Reads the two M1 traces, augments each with simulated EP ranks, and prints/dumps
a side-by-side comparison of:
  - C1: top-q share for q in {5%, 10%, 20%, 50%}
  - C2: best layer + best k + best silhouette
  - Per-layer cross-rank ratio summary
  - Co-activation heatmap for best-clustered layer of each model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.clustering import cluster_summary
from analysis.coactivation_matrix import compute_coactivation, plot_heatmap
from analysis.traffic_cdf import pair_cdf, top_k_share
from src.probes.ep_simulate import augment_with_ranks
from src.probes.trace_writer import load_trace
from src.utils import get_logger

logger = get_logger("compare")
REPO_ROOT = Path(__file__).resolve().parent.parent


def analyze_one(trace_dir: Path, label: str) -> dict:
    meta = json.loads((trace_dir / "run_meta.json").read_text())
    df = load_trace(trace_dir)
    df = augment_with_ranks(
        df,
        tokens_per_step=meta["tokens_per_step"],
        num_experts=meta["num_routed_experts"],
        ep_size=meta["ep_size"],
    )
    n_layers = meta["num_moe_layers"]

    # C1: top-q share at multiple q
    shares = {q: top_k_share(df, q=q) for q in (0.05, 0.10, 0.20, 0.50)}

    # Per-layer cross-rank ratio
    per_layer_cross = (
        df.groupby("layer")["cross_rank"].mean().to_dict()
    )

    # C2: best silhouette across layers and k
    best = (-1.0, -1, -1, None)
    sil_by_layer = {}
    for lyr in range(n_layers):
        M = compute_coactivation(df, layer=lyr, num_experts=meta["num_routed_experts"])
        for k_try in (4, 8, 16):
            s = cluster_summary(M, k=k_try)["silhouette"]
            if s > best[0]:
                best = (s, lyr, k_try, M)
            sil_by_layer.setdefault(lyr, {})[k_try] = s

    return {
        "label": label,
        "meta": meta,
        "shares": shares,
        "per_layer_cross_rank_ratio": {int(k): float(v) for k, v in per_layer_cross.items()},
        "best_silhouette": best[0],
        "best_layer": best[1],
        "best_k": best[2],
        "best_M": best[3],
        "sil_by_layer": sil_by_layer,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deepseek_dir", type=Path,
                   default=REPO_ROOT / "outputs/traces/m1_deepseek_wikitext")
    p.add_argument("--qwen_dir", type=Path,
                   default=REPO_ROOT / "outputs/traces/m4_qwen_wikitext")
    p.add_argument("--fig_dir", type=Path,
                   default=REPO_ROOT / "outputs/figures")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    deepseek = analyze_one(args.deepseek_dir, "DeepSeek-V2-Lite")
    qwen = analyze_one(args.qwen_dir, "Qwen1.5-MoE-A2.7B")

    summary = {}
    for r in (deepseek, qwen):
        summary[r["label"]] = {
            "num_routed_experts": r["meta"]["num_routed_experts"],
            "top_k": r["meta"]["top_k"],
            "num_moe_layers": r["meta"]["num_moe_layers"],
            "C1_top_share_at_5pct": r["shares"][0.05],
            "C1_top_share_at_10pct": r["shares"][0.10],
            "C1_top_share_at_20pct": r["shares"][0.20],
            "C1_top_share_at_50pct": r["shares"][0.50],
            "C2_best_layer": r["best_layer"],
            "C2_best_k": r["best_k"],
            "C2_best_silhouette": r["best_silhouette"],
            "C1_pass_at_20": r["shares"][0.20] >= 0.5,
            "C2_pass": r["best_silhouette"] >= 0.2,
        }

    out_json = args.fig_dir / "model_compare.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {out_json}")

    # Side-by-side heatmaps of best layer in each model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, r in zip(axes, (deepseek, qwen)):
        plot_heatmap(
            r["best_M"], ax=ax,
            title=f"{r['label']} layer {r['best_layer']} (silhouette={r['best_silhouette']:.3f})",
        )
    fig.suptitle("F7 — Best-clustered co-activation layer: DeepSeek-V2-Lite vs Qwen1.5-MoE")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F7_compare_heatmap.png", dpi=150)
    fig.savefig(args.fig_dir / "F7_compare_heatmap.pdf")
    plt.close(fig)

    # Side-by-side CDFs of (src,dst) pairs
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in (deepseek, qwen):
        # rebuild flat trace df for one CDF
        meta = r["meta"]
        df = load_trace(Path(REPO_ROOT) / ("outputs/traces/m1_deepseek_wikitext"
                                            if r["label"].startswith("Deep") else
                                            "outputs/traces/m4_qwen_wikitext"))
        df = augment_with_ranks(
            df,
            tokens_per_step=meta["tokens_per_step"],
            num_experts=meta["num_routed_experts"],
            ep_size=meta["ep_size"],
        )
        cdf = pair_cdf(df, group_by_layer=False)
        x = np.arange(1, cdf.size + 1) / cdf.size
        ax.plot(x, cdf, label=r["label"], linewidth=1.8)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="uniform reference")
    ax.axvline(0.20, color="crimson", linestyle="--", linewidth=1, label="top-20% mark")
    ax.set_xlabel("Fraction of (src_rank, dst_expert) pairs (descending traffic)")
    ax.set_ylabel("Cumulative fraction of cross-rank tokens")
    ax.set_title("F8 — Traffic CDF: DeepSeek vs Qwen (aggregate over all layers)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F8_compare_cdf.png", dpi=150)
    fig.savefig(args.fig_dir / "F8_compare_cdf.pdf")
    plt.close(fig)

    # Print headline
    for label, s in summary.items():
        print(f"\n=== {label} ===")
        print(f"  experts={s['num_routed_experts']}, top_k={s['top_k']}, layers={s['num_moe_layers']}")
        print(f"  C1 top-5%  share = {s['C1_top_share_at_5pct']*100:.1f}%")
        print(f"  C1 top-10% share = {s['C1_top_share_at_10pct']*100:.1f}%")
        print(f"  C1 top-20% share = {s['C1_top_share_at_20pct']*100:.1f}% (threshold 50% → pass={s['C1_pass_at_20']})")
        print(f"  C1 top-50% share = {s['C1_top_share_at_50pct']*100:.1f}%")
        print(f"  C2 best layer={s['C2_best_layer']}, k={s['C2_best_k']}, silhouette={s['C2_best_silhouette']:.3f} → pass={s['C2_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
