"""Streaming Parquet writer for MoE routing events.

Schema (one row per (token, top-k slot)):

    step              uint16
    layer             uint8
    token_idx_in_run  uint32       # global within this run (i.e. unique per row's step + token offset)
    expert_id         uint16
    topk_rank         uint8
    weight            float32

The writer buffers rows in memory and flushes a parquet shard per step (or per
flush threshold) to keep peak RAM bounded. All shards live under one directory;
:func:`load_trace` re-assembles them as a single DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.utils import get_logger

logger = get_logger(__name__)


_SCHEMA = pa.schema(
    [
        ("step", pa.uint16()),
        ("layer", pa.uint8()),
        ("token_idx_in_run", pa.uint32()),
        ("expert_id", pa.uint16()),
        ("topk_rank", pa.uint8()),
        ("weight", pa.float32()),
    ]
)


@dataclass
class TraceWriter:
    out_dir: Path
    flush_every_rows: int = 2_000_000  # ~10-20 MB per shard
    _buffers: dict = field(default_factory=dict)  # column-name -> list[np.ndarray]
    _row_count: int = 0
    _shard_idx: int = 0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for col in _SCHEMA.names:
            self._buffers[col] = []

    # ------------------------------------------------------------------ public API

    def write_routing(
        self,
        step: int,
        layer: int,
        topk_idx: torch.Tensor,   # [N, K] int64
        topk_weight: torch.Tensor,  # [N, K] float32
        token_offset: int = 0,
    ) -> None:
        n_tokens, top_k = topk_idx.shape
        idx_np = topk_idx.cpu().numpy().astype(np.uint16)
        w_np = topk_weight.cpu().numpy().astype(np.float32)

        # Build per-column flat arrays of length n_tokens * top_k
        token_ids = np.repeat(
            np.arange(n_tokens, dtype=np.uint32) + np.uint32(token_offset),
            top_k,
        )
        topk_rank = np.tile(np.arange(top_k, dtype=np.uint8), n_tokens)
        flat_expert = idx_np.reshape(-1)
        flat_weight = w_np.reshape(-1)
        n_rows = flat_expert.size

        self._buffers["step"].append(np.full(n_rows, step, dtype=np.uint16))
        self._buffers["layer"].append(np.full(n_rows, layer, dtype=np.uint8))
        self._buffers["token_idx_in_run"].append(token_ids)
        self._buffers["expert_id"].append(flat_expert)
        self._buffers["topk_rank"].append(topk_rank)
        self._buffers["weight"].append(flat_weight)
        self._row_count += n_rows

        if self._row_count >= self.flush_every_rows:
            self.flush()

    def flush(self) -> Optional[Path]:
        if self._row_count == 0:
            return None
        cols = {name: np.concatenate(parts) for name, parts in self._buffers.items()}
        table = pa.table(cols, schema=_SCHEMA)
        shard_path = self.out_dir / f"shard_{self._shard_idx:04d}.parquet"
        pq.write_table(table, shard_path, compression="zstd", compression_level=3)
        logger.info(
            f"Flushed shard {self._shard_idx} -> {shard_path.name} "
            f"({self._row_count:,} rows, {shard_path.stat().st_size / 1e6:.1f} MB)"
        )
        for col in _SCHEMA.names:
            self._buffers[col] = []
        self._row_count = 0
        self._shard_idx += 1
        return shard_path

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def load_trace(out_dir: Path) -> "pd.DataFrame":  # noqa: F821 — pandas imported lazily
    import pandas as pd  # local import to keep writer module light

    out_dir = Path(out_dir)
    shards: List[Path] = sorted(out_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {out_dir}")
    return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
