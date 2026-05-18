"""M2 — Multi-domain probe. Captures routing across 4 datasets, one trace per domain."""

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

logger = get_logger("probe.multidomain")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "outputs" / "traces" / "m2_deepseek_multidomain"

DOMAINS = ["wikitext", "c4", "stack", "mmlu"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_batches_per_domain", type=int, default=15)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ep_size", type=int, default=4)
    p.add_argument("--domains", nargs="*", default=DOMAINS)
    args = p.parse_args()

    set_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)

    device = next(model.parameters()).device
    tokens_per_step = args.batch_size * args.seq_len

    summary: dict = {"domains": {}, "meta": {
        "model": args.model,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_batches_per_domain": args.n_batches_per_domain,
        "ep_size": args.ep_size,
        "tokens_per_step": tokens_per_step,
        "num_routed_experts": meta.num_routed_experts,
        "top_k": meta.top_k,
        "num_moe_layers": meta.num_moe_layers,
    }}

    for domain in args.domains:
        domain_dir = args.out / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(out_dir=domain_dir, flush_every_rows=2_000_000)
        collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
        collector.register()

        t_start = time.time()
        it = iter_packed_batches(
            domain, tok, args.batch_size, args.seq_len, args.n_batches_per_domain
        )
        pbar = tqdm(it, total=args.n_batches_per_domain, desc=f"M2[{domain}]")
        for batch_idx, input_ids in pbar:
            collector.set_step(step=batch_idx, batch_offset=batch_idx * tokens_per_step)
            with torch.no_grad():
                _ = model(input_ids.to(device))
        pbar.close()

        collector.remove()
        writer.close()
        elapsed = time.time() - t_start
        summary["domains"][domain] = {"trace_dir": str(domain_dir), "elapsed_s": elapsed}
        logger.info(f"M2 domain={domain} done in {elapsed:.1f}s")

    (args.out / "run_meta.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"M2 multi-domain done; summary -> {args.out / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
