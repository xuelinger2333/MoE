"""MoE-Infinity EAMC (Expert Activation Matrix Collection) reproduction.

Faithful reproduction of the offline+online algorithm described in
arXiv 2401.14361 (MoE-Infinity), used as a baseline for H4. Compares
against session-ID-based affinity, the proposed H4 method.

Key mechanism (paraphrased from MoE-Infinity §3):

  1. Per request, build a request-level EAM (L × E matrix of token counts
     per (layer, expert)).
  2. Offline: cluster EAMs from a calibration trace via K-Means with
     ``k = EAMC_capacity``. Centroids → EAMC.
  3. Online: for each new request's EAM, match against EAMC via cosine
     distance on flattened vectors; the closest EAM guides prefetch/cache.
  4. Adaptive update: when a new EAM is processed, optionally replace the
     most-similar existing EAMC entry (cosine-min replacement) to track
     distribution shift.

For H4 we use it as a **prediction baseline**: given the past EAM (turn
1..k-1 aggregated) of an incoming conversation's current turn, predict the
expert set most likely to be activated. Compare to:
    (a) Session-ID: assume same as last turn of THIS conversation
    (b) EAMC: match incoming request's running EAM against EAMC, predict
        experts from the matched centroid
    (c) Random: uniform sample
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# EAM construction
# ---------------------------------------------------------------------------

def build_eam(
    df: pd.DataFrame, n_layers: int, n_experts: int,
) -> np.ndarray:
    """Build an L x E request-level Expert Activation Matrix.

    ``M[l, e] = number of (token, top-k slot) events at layer l routing to expert e``.
    Per MoE-Infinity Section 3, normalisation is left as a downstream choice.
    """
    M = np.zeros((n_layers, n_experts), dtype=np.float32)
    if df.empty:
        return M
    layers = df["layer"].to_numpy()
    experts = df["expert_id"].to_numpy()
    np.add.at(M, (layers, experts), 1.0)
    return M


def flatten_eam(M: np.ndarray) -> np.ndarray:
    return M.reshape(-1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# EAMC: K-Means construction + online matching
# ---------------------------------------------------------------------------

@dataclass
class EAMCEntry:
    matrix: np.ndarray             # L x E
    flat: np.ndarray               # L*E vector cache for fast matching
    source_id: Optional[str] = None  # diagnostic: which conversation/branch produced this


class EAMC:
    """Expert Activation Matrix Collection.

    Two modes:
      * Offline construct via K-Means clustering of calibration EAMs.
      * Online insertion with cosine-min replacement (MoE-Infinity §3 update).
    """

    def __init__(self, n_layers: int, n_experts: int, capacity: int):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.capacity = capacity
        self.entries: List[EAMCEntry] = []

    @classmethod
    def fit_offline(
        cls,
        eams: List[np.ndarray],
        n_layers: int,
        n_experts: int,
        capacity: int = 100,
        seed: int = 42,
    ) -> "EAMC":
        """K-Means clustering of calibration EAMs → centroids stored in EAMC.

        Pure-numpy K-Means (no sklearn dependency to keep deps light).
        """
        rng = np.random.default_rng(seed)
        if len(eams) == 0:
            return cls(n_layers, n_experts, capacity)
        flats = np.stack([m.reshape(-1) for m in eams], axis=0)  # [N, L*E]
        n = flats.shape[0]
        k = min(capacity, n)
        # Init: k-means++ minimal
        idx0 = int(rng.integers(n))
        centroids = [flats[idx0]]
        for _ in range(1, k):
            d2 = np.min(
                np.stack([np.sum((flats - c) ** 2, axis=1) for c in centroids], axis=0),
                axis=0,
            )
            probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
            pick = int(rng.choice(n, p=probs))
            centroids.append(flats[pick])
        C = np.stack(centroids, axis=0)
        # Lloyd
        for _ in range(15):
            # assign
            d = np.stack([np.sum((flats - C[i]) ** 2, axis=1) for i in range(k)], axis=0)
            assign = np.argmin(d, axis=0)
            # update
            new_C = np.zeros_like(C)
            for i in range(k):
                mask = assign == i
                if mask.any():
                    new_C[i] = flats[mask].mean(axis=0)
                else:
                    new_C[i] = C[i]
            if np.allclose(new_C, C, atol=1e-6):
                break
            C = new_C

        eamc = cls(n_layers, n_experts, capacity)
        for i in range(k):
            M = C[i].reshape(n_layers, n_experts)
            eamc.entries.append(EAMCEntry(matrix=M, flat=C[i].copy(), source_id=f"centroid_{i}"))
        return eamc

    def match(self, eam: np.ndarray) -> Tuple[int, float]:
        """Return ``(index_of_closest, cosine_distance)`` against current EAMC."""
        if not self.entries:
            return -1, 1.0
        flat = eam.reshape(-1)
        dists = np.array([cosine_distance(e.flat, flat) for e in self.entries])
        i = int(np.argmin(dists))
        return i, float(dists[i])

    def online_update(self, eam: np.ndarray, source_id: Optional[str] = None) -> None:
        """MoE-Infinity §3 update rule: replace the most-similar entry with the new one.

        Maintains diversity by evicting near-duplicates.
        """
        flat = eam.reshape(-1)
        if len(self.entries) < self.capacity:
            self.entries.append(EAMCEntry(matrix=eam.copy(), flat=flat.copy(), source_id=source_id))
            return
        # Find most-similar (smallest cosine distance) entry and replace
        i, _ = self.match(eam)
        self.entries[i] = EAMCEntry(matrix=eam.copy(), flat=flat.copy(), source_id=source_id)


# ---------------------------------------------------------------------------
# Prediction: experts predicted to be "hot" for an incoming turn
# ---------------------------------------------------------------------------

def predict_expert_set_from_eam(
    eam: np.ndarray, layer: Optional[int] = None, top_pct: float = 0.5,
) -> np.ndarray:
    """Return the expert IDs in the top ``top_pct`` of activation count.

    If ``layer`` is given, return per-layer hot set (1D). Otherwise return
    the union across all layers.
    """
    if layer is not None:
        row = eam[layer]
        n_keep = max(1, int(np.ceil(top_pct * row.size)))
        return np.argsort(-row)[:n_keep]
    # All-layer union of hot experts
    n_per_layer = max(1, int(np.ceil(top_pct * eam.shape[1])))
    sets = [np.argsort(-eam[l])[:n_per_layer] for l in range(eam.shape[0])]
    return np.unique(np.concatenate(sets))


def predict_eamc_match(
    eamc: EAMC, running_eam: np.ndarray, top_pct: float = 0.5,
) -> np.ndarray:
    """EAMC baseline prediction: match running EAM, return matched EAM's hot experts."""
    i, _ = eamc.match(running_eam)
    if i < 0:
        # Empty EAMC: fall back to uniform / empty
        return np.array([], dtype=np.int64)
    return predict_expert_set_from_eam(eamc.entries[i].matrix, top_pct=top_pct)


def predict_session_id(
    last_turn_eam: np.ndarray, top_pct: float = 0.5,
) -> np.ndarray:
    """Session-ID baseline: predict experts active in the last turn of same conversation."""
    return predict_expert_set_from_eam(last_turn_eam, top_pct=top_pct)


def predict_random(n_experts: int, top_pct: float = 0.5, seed: int = 42) -> np.ndarray:
    """Random baseline: uniform sample without replacement."""
    rng = np.random.default_rng(seed)
    k = max(1, int(np.ceil(top_pct * n_experts)))
    return rng.choice(n_experts, size=k, replace=False)
