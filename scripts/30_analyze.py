"""M3 — Render motivation figures from M1 + M2 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safety
import matplotlib.pyplot as plt
import numpy as np

from analysis.clustering import cluster_summary, plot_dendrogram, silhouette_vs_k
from analysis.coactivation_matrix import compute_coactivation, plot_heatmap
from analysis.locality_ratio import per_layer_locality, plot_locality_bars
from analysis.traffic_cdf import (
    pair_cdf,
    plot_cdf_per_layer,
    plot_cdf_per_domain,
    top_k_share,
)
from src.probes.ep_simulate import augment_with_ranks
from src.probes.trace_writer import load_trace
from src.utils import get_logger

logger = get_logger("analyze")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args) -> int:
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- M1 single-domain ----------
    m1_meta = json.loads((args.m1_dir / "run_meta.json").read_text())
    df_m1 = load_trace(args.m1_dir)
    df_m1 = augment_with_ranks(
        df_m1,
        tokens_per_step=m1_meta["tokens_per_step"],
        num_experts=m1_meta["num_routed_experts"],
        ep_size=m1_meta["ep_size"],
    )
    logger.info(f"M1 trace: {len(df_m1):,} rows, {df_m1['layer'].nunique()} layers")

    # F1: traffic CDF per layer
    cdf_layers = pair_cdf(df_m1, group_by_layer=True)
    plot_cdf_per_layer(cdf_layers, fig_dir / "F1_traffic_cdf.png")

    # F2: locality ratio per layer
    loc = per_layer_locality(df_m1)
    plot_locality_bars(loc, fig_dir / "F2_locality_ratio.png")

    # Headline number for C1
    share_20 = top_k_share(df_m1, q=0.20)
    logger.info(f"[C1] Top-20% (src,dst) pairs carry {share_20*100:.1f}% of cross-rank tokens")

    # F4: co-activation heatmaps for early/middle/late layers
    n_layers = m1_meta["num_moe_layers"]
    layer_picks = sorted({0, n_layers // 2, n_layers - 1})
    fig, axes = plt.subplots(1, len(layer_picks), figsize=(5 * len(layer_picks), 4.5))
    if len(layer_picks) == 1:
        axes = [axes]
    for ax, lyr in zip(axes, layer_picks):
        M = compute_coactivation(df_m1, layer=lyr, num_experts=m1_meta["num_routed_experts"])
        plot_heatmap(M, ax=ax, title=f"layer {lyr}")
    fig.tight_layout()
    fig.savefig(fig_dir / "F4_coactivation_heatmap.png", dpi=150)
    fig.savefig(fig_dir / "F4_coactivation_heatmap.pdf")
    plt.close(fig)
    logger.info("F4 co-activation heatmap saved")

    # F5: dendrogram on best-clustered layer
    best_layer, best_k, best_score = -1, -1, -1.0
    for lyr in range(n_layers):
        M = compute_coactivation(df_m1, layer=lyr, num_experts=m1_meta["num_routed_experts"])
        for k_try in (4, 8, 16):
            score = cluster_summary(M, k=k_try)["silhouette"]
            if score > best_score:
                best_layer, best_k, best_score = lyr, k_try, score
    M_best = compute_coactivation(
        df_m1, layer=best_layer, num_experts=m1_meta["num_routed_experts"]
    )
    plot_dendrogram(
        M_best,
        out_path=fig_dir / "F5_dendrogram.png",
        title=f"layer {best_layer} (best silhouette={best_score:.3f} @ k={best_k})",
    )
    logger.info(
        f"[C2] Best clustering: layer={best_layer}, k={best_k}, silhouette={best_score:.3f}"
    )

    # F6 (optional): silhouette vs k vs layer
    sil_grid = silhouette_vs_k(
        df_m1,
        num_experts=m1_meta["num_routed_experts"],
        layers=list(range(0, n_layers, max(1, n_layers // 6))),
        ks=(2, 4, 8, 16),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for lyr, scores in sil_grid.items():
        ax.plot(list(scores.keys()), list(scores.values()), marker="o", label=f"layer {lyr}")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("silhouette")
    ax.set_title("Co-activation cluster quality vs k, by layer")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, label="C2 threshold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "F6_silhouette_vs_k.png", dpi=150)
    fig.savefig(fig_dir / "F6_silhouette_vs_k.pdf")
    plt.close(fig)

    # ---------- M2 multi-domain ----------
    domain_cdfs: dict[str, np.ndarray] = {}
    domain_share: dict[str, float] = {}
    if args.m2_dir.exists() and (args.m2_dir / "run_meta.json").exists():
        m2_meta = json.loads((args.m2_dir / "run_meta.json").read_text())
        for domain, info in m2_meta["domains"].items():
            df_d = load_trace(Path(info["trace_dir"]))
            df_d = augment_with_ranks(
                df_d,
                tokens_per_step=m2_meta["meta"]["tokens_per_step"],
                num_experts=m2_meta["meta"]["num_routed_experts"],
                ep_size=m2_meta["meta"]["ep_size"],
            )
            domain_cdfs[domain] = pair_cdf(df_d, group_by_layer=False)
            domain_share[domain] = top_k_share(df_d, q=0.20)
        plot_cdf_per_domain(domain_cdfs, fig_dir / "F3_per_domain_cdf.png")
        logger.info(f"[per-domain top-20% share] {domain_share}")
    else:
        logger.warning(f"No M2 traces found at {args.m2_dir}; skipping F3")

    # ---------- Write headline summary ----------
    summary = {
        "C1_top20_share_M1": share_20,
        "C1_per_domain_top20_share": domain_share,
        "C2_best_layer": best_layer,
        "C2_best_k": best_k,
        "C2_best_silhouette": best_score,
        "C1_threshold": 0.5,
        "C1_strong_threshold": 0.7,
        "C2_threshold": 0.2,
        "C1_pass": share_20 >= 0.5,
        "C2_pass": best_score >= 0.2,
    }
    (fig_dir / "claim_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote claim summary: {fig_dir / 'claim_summary.json'}")
    logger.info(f"  C1 pass={summary['C1_pass']} (top-20% share={share_20*100:.1f}%)")
    logger.info(f"  C2 pass={summary['C2_pass']} (best silhouette={best_score:.3f})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m1_dir", type=Path, default=REPO_ROOT / "outputs/traces/m1_deepseek_wikitext")
    p.add_argument(
        "--m2_dir", type=Path, default=REPO_ROOT / "outputs/traces/m2_deepseek_multidomain"
    )
    p.add_argument("--fig_dir", type=Path, default=REPO_ROOT / "outputs/figures")
    args = p.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
