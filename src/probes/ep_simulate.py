"""Offline simulation of EP rank assignments from a routing trace.

We do **not** run real expert-parallel distribution. Instead, given a trace
containing ``(token_idx_in_step, expert_id)``, we deterministically assign:

- ``src_rank``: which DP rank "owns" this token, as ``token_idx_in_step // (tokens_per_rank)``
- ``dst_rank``: which EP rank hosts this expert, as ``expert_id // experts_per_rank``

This produces the same ``(src_rank, dst_expert)`` distribution that a real
EP=ep_size + DP=ep_size deployment would observe on the same model + data,
modulo that we use a fixed (not learned) sample-to-rank mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def simulate_src_rank(
    token_idx_in_step: np.ndarray, tokens_per_step: int, ep_size: int
) -> np.ndarray:
    tokens_per_rank = tokens_per_step // ep_size
    if tokens_per_rank == 0:
        raise ValueError(
            f"tokens_per_step={tokens_per_step} < ep_size={ep_size}; cannot split"
        )
    return (token_idx_in_step // tokens_per_rank).astype(np.uint8)


def simulate_dst_rank(expert_id: np.ndarray, num_experts: int, ep_size: int) -> np.ndarray:
    experts_per_rank = num_experts // ep_size
    if experts_per_rank * ep_size != num_experts:
        raise ValueError(
            f"num_experts={num_experts} not divisible by ep_size={ep_size}"
        )
    return (expert_id // experts_per_rank).astype(np.uint8)


def augment_with_ranks(
    df: "pd.DataFrame",
    tokens_per_step: int,
    num_experts: int,
    ep_size: int,
) -> "pd.DataFrame":
    """Add ``src_rank``, ``dst_rank``, ``cross_rank`` columns to a routing DataFrame."""
    df = df.copy()
    # token_idx_in_run is global across all steps; modulo to per-step idx
    token_idx_in_step = (df["token_idx_in_run"].to_numpy() % tokens_per_step).astype(np.uint32)
    df["src_rank"] = simulate_src_rank(token_idx_in_step, tokens_per_step, ep_size)
    df["dst_rank"] = simulate_dst_rank(df["expert_id"].to_numpy(), num_experts, ep_size)
    df["cross_rank"] = (df["src_rank"] != df["dst_rank"]).astype(np.uint8)
    return df
