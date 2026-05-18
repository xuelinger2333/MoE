"""Per-token router entropy analysis (G8 OOD disambiguation).

Loads the entropy NPZ files written by scripts/22_probe_router_entropy.py and
provides:
- summary statistics per (model, domain)
- entropy distribution histograms
- 'effective experts' = exp(entropy)
- a confidence-filtered MI joint test (joined with the M8 routing traces)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analysis.cross_layer_mi import (
    joint_counts,
    mutual_information_nats,
    shuffled_null_mi,
    top1_pivot,
)
from src.probes.trace_writer import load_trace


@dataclass
class EntropyData:
    entropies: np.ndarray  # [n_tokens, n_layers]
    meta: dict


def load_entropy(npz_path: Path) -> EntropyData:
    z = np.load(npz_path, allow_pickle=False)
    ent = z["entropies"]
    # meta was serialised as a 0-d JSON string array
    meta = json.loads(str(z["meta"]))
    return EntropyData(entropies=ent, meta=meta)


def per_layer_summary(ed: EntropyData) -> pd.DataFrame:
    e = ed.entropies
    n_layers = e.shape[1]
    rows = []
    H_uniform = float(np.log(ed.meta["num_routed_experts"]))
    for L in range(n_layers):
        col = e[:, L]
        rows.append({
            "layer": L,
            "mean_H": float(col.mean()),
            "median_H": float(np.median(col)),
            "p25_H": float(np.percentile(col, 25)),
            "p75_H": float(np.percentile(col, 75)),
            "mean_eff_experts": float(np.exp(col.mean())),
            "median_eff_experts": float(np.exp(np.median(col))),
            "fraction_uniform": float(col.mean() / H_uniform),
        })
    return pd.DataFrame(rows)


def filtered_mi_per_layer_pair(
    entropies: np.ndarray,        # [n_tokens, n_layers]
    wide_top1: pd.DataFrame,      # token_idx_in_run × layer (top-1 expert id)
    num_experts: int,
    threshold: float,
    null_repeats: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """For each adjacent pair (L, L+1), restrict to tokens where BOTH entropies
    are below ``threshold``, then compute MI vs i.i.d. shuffled null on the
    filtered subset.
    """
    n_tokens, n_layers = entropies.shape
    layer_cols = sorted(int(c) for c in wide_top1.columns)
    # Build a contiguous index array aligning entropy rows to wide rows.
    # wide rows are indexed by token_idx_in_run; assume the entropy array is
    # ordered so that row i corresponds to global token index i. (Same data,
    # same seed, same loader → tokens align by construction.)
    wide_idx = wide_top1.index.to_numpy()
    rows = []
    for L in layer_cols:
        L2 = L + 1
        if L2 not in wide_top1.columns:
            continue
        eL = entropies[wide_idx, L]
        eLp1 = entropies[wide_idx, L2]
        mask = (eL < threshold) & (eLp1 < threshold)
        n_kept = int(mask.sum())
        if n_kept < 100:
            rows.append({"L": L, "n_kept": n_kept, "MI_filtered": float("nan"),
                         "null_MI_filtered": float("nan"),
                         "MI_minus_null": float("nan")})
            continue
        x = wide_top1.iloc[:, layer_cols.index(L)].to_numpy()[mask]
        y = wide_top1.iloc[:, layer_cols.index(L2)].to_numpy()[mask]
        C = joint_counts(x, y, num_experts, num_experts)
        mi, *_ = mutual_information_nats(C)
        null_mi = shuffled_null_mi(x, y, num_experts, num_experts,
                                    n_repeats=null_repeats, seed=seed + L)
        rows.append({
            "L": L,
            "n_kept": n_kept,
            "MI_filtered": mi,
            "null_MI_filtered": null_mi,
            "MI_minus_null": mi - null_mi,
        })
    return pd.DataFrame(rows)


def filtered_mi_summary(
    npz_path: Path,
    trace_dir: Path,
    threshold: float,
) -> dict:
    ed = load_entropy(npz_path)
    df = load_trace(trace_dir)
    wide = top1_pivot(df)
    fmi = filtered_mi_per_layer_pair(
        ed.entropies, wide, ed.meta["num_routed_experts"], threshold=threshold
    )
    return {
        "threshold_nats": threshold,
        "n_pairs": int(fmi["L"].nunique()),
        "mean_MI_filtered_minus_null": float(fmi["MI_minus_null"].dropna().mean()),
        "mean_n_kept": float(fmi["n_kept"].mean()),
        "n_kept_min": int(fmi["n_kept"].min()),
        "n_kept_max": int(fmi["n_kept"].max()),
        "per_pair": fmi.to_dict(orient="records"),
    }
