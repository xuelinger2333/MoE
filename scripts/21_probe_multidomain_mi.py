"""M8 — Multi-domain probe for cross-layer MI study (G6 round-3).

For one model, run M1-style probes on multiple domains in sequence,
each writing a separate trace dir. Re-uses the M1 probe loop;
just iterates over domains.

Output layout:
    outputs/traces/m8_<model>_<domain>/...

Usage:
    python scripts/21_probe_multidomain_mi.py \
        --model Qwen/Qwen1.5-MoE-A2.7B \
        --tag qwen \
        --domains code math
"""

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

logger = get_logger("probe.m8")
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="short label for the model used in trace dir name")
    p.add_argument("--domains", nargs="+", default=["code", "math"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--ep_size", type=int, default=4)
    p.add_argument("--out_root", type=Path, default=REPO_ROOT / "outputs" / "traces")
    args = p.parse_args()

    set_seed(42)

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    logger.info(
        f"MoE meta: experts={meta.num_routed_experts}, top_k={meta.top_k}, "
        f"moe_layers={meta.num_moe_layers}, gate_class={meta.gate_module_class}"
    )

    device = next(model.parameters()).device
    tokens_per_step = args.batch_size * args.seq_len

    for domain in args.domains:
        out_dir = args.out_root / f"m8_{args.tag}_{domain}"
        out_dir.mkdir(parents=True, exist_ok=True)

        writer = TraceWriter(out_dir=out_dir, flush_every_rows=2_000_000)
        collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
        collector.register()

        t_start = time.time()
        it = iter_packed_batches(domain, tok, args.batch_size, args.seq_len, args.n_batches)
        pbar = tqdm(it, total=args.n_batches, desc=f"M8[{args.tag}/{domain}]")
        n_completed = 0
        for batch_idx, input_ids in pbar:
            collector.set_step(step=batch_idx, batch_offset=batch_idx * tokens_per_step)
            with torch.no_grad():
                _ = model(input_ids.to(device))
            n_completed = batch_idx + 1
        pbar.close()

        collector.remove()
        writer.close()
        elapsed = time.time() - t_start

        run_meta = {
            "model": args.model,
            "tag": args.tag,
            "domain": domain,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches_requested": args.n_batches,
            "n_batches_completed": n_completed,
            "ep_size": args.ep_size,
            "tokens_per_step": tokens_per_step,
            "num_routed_experts": meta.num_routed_experts,
            "top_k": meta.top_k,
            "num_moe_layers": meta.num_moe_layers,
            "elapsed_s": elapsed,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
        logger.info(f"M8 {args.tag}/{domain}: completed {n_completed}/{args.n_batches} batches in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
