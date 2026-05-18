"""M0 — Smoke test. Verify hook registration + trace shape on a tiny input.

Runs in <1 minute on a single A100. Asserts:
- Hook count equals expected MoE-layer count.
- Trace contains rows for every layer.
- Expert IDs and topk_rank are within bounds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import iter_packed_batches
from src.models import get_moe_metadata, load_moe_model
from src.probes import TraceCollector, TraceWriter
from src.probes.trace_writer import load_trace
from src.utils import get_logger, set_seed

logger = get_logger("smoke")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs" / "traces" / "m0_smoke"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    set_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    logger.info(
        f"MoE meta: experts={meta.num_routed_experts}, top_k={meta.top_k}, "
        f"moe_layers={meta.num_moe_layers}, gate_class={meta.gate_module_class}"
    )

    writer = TraceWriter(out_dir=args.out, flush_every_rows=50_000)
    collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
    n_hooks = collector.register()
    assert n_hooks == meta.num_moe_layers, (
        f"Hook count mismatch: registered {n_hooks}, expected {meta.num_moe_layers} MoE layers"
    )

    # Single batch, single step
    it = iter_packed_batches(
        "wikitext", tok, args.batch_size, args.seq_len, n_batches=1, slice_size=20
    )
    batch_idx, input_ids = next(it)

    device = next(model.parameters()).device
    logger.info(f"Forward pass on {input_ids.shape} → device {device}")
    collector.set_step(step=batch_idx, batch_offset=batch_idx * args.batch_size * args.seq_len)
    with torch.no_grad():
        _ = model(input_ids.to(device))

    collector.remove()
    writer.close()

    # Verify
    df = load_trace(args.out)
    logger.info(f"Trace rows: {len(df):,}")
    logger.info(f"Trace columns: {list(df.columns)}")
    logger.info(f"Layers seen: {sorted(df['layer'].unique().tolist())}")
    logger.info(f"Expert id range: [{df['expert_id'].min()}, {df['expert_id'].max()}]")
    logger.info(f"topk_rank range: [{df['topk_rank'].min()}, {df['topk_rank'].max()}]")

    expected_rows_per_step = args.batch_size * args.seq_len * meta.num_moe_layers * meta.top_k
    assert len(df) == expected_rows_per_step, (
        f"Expected {expected_rows_per_step} rows, got {len(df)}"
    )
    assert df["expert_id"].max() < meta.num_routed_experts
    assert df["topk_rank"].max() < meta.top_k
    assert df["layer"].nunique() == meta.num_moe_layers, "Some MoE layers had no routing events"
    logger.info("M0 SANITY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
