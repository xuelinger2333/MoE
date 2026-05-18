"""C1 — Traffic concentration over (src_rank, dst_expert) pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pair_cdf(df: pd.DataFrame, group_by_layer: bool = False) -> dict | np.ndarray:
    """Compute the descending-sorted normalized cumulative distribution
    of token counts over (src_rank, dst_expert) pairs.

    If ``group_by_layer`` is True, returns ``{layer: cdf_array}``. Otherwise
    returns a single 1-D array.
    """
    cross = df[df["cross_rank"] == 1]
    if group_by_layer:
        out: dict[int, np.ndarray] = {}
        for lyr, sub in cross.groupby("layer"):
            counts = np.asarray(sub.groupby(["src_rank", "expert_id"]).size().to_numpy()).copy()
            counts.sort()
            counts = counts[::-1]
            out[int(lyr)] = np.cumsum(counts) / counts.sum() if counts.sum() else counts
        return out
    counts = np.asarray(cross.groupby(["src_rank", "expert_id"]).size().to_numpy()).copy()
    counts.sort()
    counts = counts[::-1]
    if counts.sum() == 0:
        return counts.astype(float)
    return np.cumsum(counts) / counts.sum()


def top_k_share(df: pd.DataFrame, q: float = 0.20) -> float:
    """Fraction of cross-rank tokens carried by the top ``q`` fraction of pairs."""
    cross = df[df["cross_rank"] == 1]
    counts = np.asarray(cross.groupby(["src_rank", "expert_id"]).size().to_numpy()).copy()
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    counts.sort()
    counts = counts[::-1]
    cutoff = max(1, int(np.ceil(q * counts.size)))
    return float(counts[:cutoff].sum() / counts.sum())


def plot_cdf_per_layer(cdfs: dict[int, np.ndarray], out_path: Path) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lyr in sorted(cdfs):
        cdf = cdfs[lyr]
        if cdf.size == 0:
            continue
        x = np.arange(1, cdf.size + 1) / cdf.size
        ax.plot(x, cdf, alpha=0.4, linewidth=0.8)
    # Bold aggregate
    if cdfs:
        agg = np.mean(
            [
                np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, c.size), c)
                for c in cdfs.values()
                if c.size > 1
            ],
            axis=0,
        )
        ax.plot(np.linspace(0, 1, 200), agg, color="black", linewidth=2.0, label="mean across layers")
    ax.axvline(0.20, color="crimson", linestyle="--", linewidth=1, label="top-20% mark")
    ax.set_xlabel("Fraction of (src_rank, dst_expert) pairs (descending traffic)")
    ax.set_ylabel("Cumulative fraction of cross-rank tokens")
    ax.set_title("F1 — Traffic CDF over (src_rank, dst_expert) pairs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_cdf_per_domain(cdfs: dict[str, np.ndarray], out_path: Path) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for domain, cdf in cdfs.items():
        if cdf.size == 0:
            continue
        x = np.arange(1, cdf.size + 1) / cdf.size
        ax.plot(x, cdf, label=domain, linewidth=1.5)
    ax.axvline(0.20, color="crimson", linestyle="--", linewidth=1, label="top-20% mark")
    ax.set_xlabel("Fraction of (src_rank, dst_expert) pairs (descending traffic)")
    ax.set_ylabel("Cumulative fraction of cross-rank tokens")
    ax.set_title("F3 — Per-domain traffic CDF (aggregated across layers)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
