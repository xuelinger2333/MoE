"""C1 helper — per-layer cross-rank vs intra-rank ratio."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def per_layer_locality(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("layer").agg(
        total=("expert_id", "size"), cross=("cross_rank", "sum")
    )
    g["intra"] = g["total"] - g["cross"]
    g["cross_ratio"] = g["cross"] / g["total"]
    g["intra_ratio"] = 1.0 - g["cross_ratio"]
    return g.reset_index()


def plot_locality_bars(loc: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = loc["layer"].astype(int).to_numpy()
    ax.bar(layers, loc["intra_ratio"], label="intra-rank", color="#5A9BD3")
    ax.bar(
        layers,
        loc["cross_ratio"],
        bottom=loc["intra_ratio"],
        label="cross-rank",
        color="#D35F5F",
    )
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel("Token fraction")
    ax.set_title("F2 — Cross-rank vs intra-rank token routing per layer (simulated EP=4)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
