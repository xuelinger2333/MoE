"""C2 — Co-activation matrix per layer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_coactivation(df: pd.DataFrame, layer: int, num_experts: int) -> np.ndarray:
    """For one layer, build the symmetric ``[num_experts, num_experts]`` matrix
    where ``M[i,j]`` is the number of tokens that selected both expert ``i`` and
    expert ``j`` in their top-k set. ``M[i,i]`` counts total activations of expert ``i``.
    """
    sub = df[df["layer"] == layer]
    if sub.empty:
        return np.zeros((num_experts, num_experts), dtype=np.int64)

    # Group by token; collect set of experts per token.
    grouped = sub.groupby("token_idx_in_run")["expert_id"].apply(np.asarray)

    M = np.zeros((num_experts, num_experts), dtype=np.int64)
    for experts in grouped.values:
        # increment all (i,j) pairs in the cartesian product (incl. diagonal)
        e = np.unique(experts)
        idx = np.array(np.meshgrid(e, e, indexing="ij")).reshape(2, -1)
        np.add.at(M, (idx[0], idx[1]), 1)
    return M


def normalize_coactivation(M: np.ndarray) -> np.ndarray:
    """Symmetric Jaccard-style normalization: ``M[i,j] / (sqrt(M[i,i] * M[j,j]))``."""
    diag = np.diag(M).astype(np.float64)
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        N = np.where(denom > 0, M / denom, 0.0)
    return N


def plot_heatmap(M: np.ndarray, ax=None, title: str = "co-activation") -> None:
    N = normalize_coactivation(M)
    np.fill_diagonal(N, 0.0)  # suppress the diagonal so off-diagonal structure pops
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(N, cmap="magma", vmin=0, vmax=N.max() if N.max() > 0 else 1.0)
    ax.set_title(title)
    ax.set_xlabel("expert j")
    ax.set_ylabel("expert i")
    plt.colorbar(im, ax=ax, fraction=0.04)
