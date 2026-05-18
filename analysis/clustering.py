"""C2 — hierarchical clustering of experts on the co-activation matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from analysis.coactivation_matrix import compute_coactivation, normalize_coactivation


def _distance_matrix(M: np.ndarray) -> np.ndarray:
    N = normalize_coactivation(M)
    np.fill_diagonal(N, 1.0)
    D = 1.0 - N
    D = (D + D.T) / 2.0   # enforce symmetry against floating noise
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    return D


def cluster_summary(M: np.ndarray, k: int) -> dict:
    if M.sum() == 0 or k >= M.shape[0]:
        return {"k": k, "silhouette": -1.0, "labels": None}
    D = _distance_matrix(M)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    if len(np.unique(labels)) < 2:
        return {"k": k, "silhouette": -1.0, "labels": labels}
    score = float(silhouette_score(D, labels, metric="precomputed"))
    return {"k": k, "silhouette": score, "labels": labels}


def silhouette_vs_k(
    df: pd.DataFrame,
    num_experts: int,
    layers: list[int],
    ks: tuple[int, ...] = (2, 4, 8, 16),
) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for lyr in layers:
        M = compute_coactivation(df, layer=lyr, num_experts=num_experts)
        out[lyr] = {k: cluster_summary(M, k=k)["silhouette"] for k in ks}
    return out


def plot_dendrogram(M: np.ndarray, out_path: Path, title: str = "") -> None:
    out_path = Path(out_path)
    D = _distance_matrix(M)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(f"F5 — Expert hierarchical clustering ({title})")
    ax.set_xlabel("expert id")
    ax.set_ylabel("co-activation distance (1 - normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
