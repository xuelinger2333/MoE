"""M1 — Single-domain probe. Capture routing on WikiText-103 validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import iter_packed_batches
from src.models import get_moe_metadata, load_moe_model
from src.probes import TraceCollector, TraceWriter
from src.utils import get_logger, set_seed

logger = get_logger("probe.single")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs" / "traces" / "m1_deepseek_wikitext"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ep_size", type=int, default=4)  # recorded for downstream analysis
    args = p.parse_args()

    set_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    logger.info(
        f"MoE meta: experts={meta.num_routed_experts}, top_k={meta.top_k}, "
        f"moe_layers={meta.num_moe_layers}"
    )

    writer = TraceWriter(out_dir=args.out, flush_every_rows=2_000_000)
    collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
    n_hooks = collector.register()
    assert n_hooks == meta.num_moe_layers

    device = next(model.parameters()).device
    tokens_per_step = args.batch_size * args.seq_len
    t_start = time.time()

    it = iter_packed_batches(args.dataset, tok, args.batch_size, args.seq_len, args.n_batches)
    pbar = tqdm(it, total=args.n_batches, desc=f"M1[{args.dataset}]")
    for batch_idx, input_ids in pbar:
        collector.set_step(step=batch_idx, batch_offset=batch_idx * tokens_per_step)
        with torch.no_grad():
            _ = model(input_ids.to(device))
    pbar.close()

    collector.remove()
    writer.close()
    elapsed = time.time() - t_start

    # Side-car run config + metadata
    run_meta = {
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_batches": args.n_batches,
        "ep_size": args.ep_size,
        "tokens_per_step": tokens_per_step,
        "num_routed_experts": meta.num_routed_experts,
        "top_k": meta.top_k,
        "num_moe_layers": meta.num_moe_layers,
        "elapsed_s": elapsed,
    }
    (args.out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    logger.info(f"Wrote {args.out / 'run_meta.json'}")
    logger.info(f"M1 done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
