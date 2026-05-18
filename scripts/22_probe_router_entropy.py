"""Probe FULL-distribution router entropy per (token, layer) per (model, domain).

Same data / seed / batch sizes as scripts/21_probe_multidomain_mi.py so that
token_idx_in_run aligns with the M8 routing traces — enabling filtered-MI
joins downstream.

Output: outputs/router_entropy/<tag>_<domain>.npz containing
    entropies: float32 [n_tokens, n_layers]
    meta:      dict with model_id, num_routed_experts, top_k, etc.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data import iter_packed_batches
from src.models import get_moe_metadata, load_moe_model
from src.probes.entropy_hooks import EntropyCollector
from src.utils import get_logger, set_seed

logger = get_logger("probe.router_entropy")
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--domains", nargs="+", default=["nl", "code", "math"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--out_root", type=Path,
                   default=REPO_ROOT / "outputs" / "router_entropy")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    logger.info(
        f"MoE meta: experts={meta.num_routed_experts}, top_k={meta.top_k}, "
        f"moe_layers={meta.num_moe_layers}, gate={meta.gate_module_class}"
    )

    device = next(model.parameters()).device
    tokens_per_step = args.batch_size * args.seq_len

    domain_aliases = {"nl": "wikitext"}  # the loader name for nl is "wikitext"

    for domain_label in args.domains:
        loader_name = domain_aliases.get(domain_label, domain_label)
        out_path = args.out_root / f"{args.tag}_{domain_label}.npz"

        collector = EntropyCollector(
            model,
            gate_module_class=meta.gate_module_class,
            num_routed_experts=meta.num_routed_experts,
            num_moe_layers=meta.num_moe_layers,
        )
        collector.register()

        t_start = time.time()
        it = iter_packed_batches(loader_name, tok, args.batch_size, args.seq_len, args.n_batches)
        pbar = tqdm(it, total=args.n_batches, desc=f"entropy[{args.tag}/{domain_label}]")
        n_completed = 0
        for batch_idx, input_ids in pbar:
            collector.set_step(step=batch_idx, token_offset=batch_idx * tokens_per_step)
            with torch.no_grad():
                _ = model(input_ids.to(device))
            n_completed = batch_idx + 1
        pbar.close()

        ent = collector.consolidate()
        collector.remove()
        elapsed = time.time() - t_start

        meta_dict = {
            "model": args.model,
            "tag": args.tag,
            "domain_label": domain_label,
            "loader_name": loader_name,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_batches_requested": args.n_batches,
            "n_batches_completed": n_completed,
            "tokens_per_step": tokens_per_step,
            "num_routed_experts": meta.num_routed_experts,
            "top_k": meta.top_k,
            "num_moe_layers": meta.num_moe_layers,
            "elapsed_s": elapsed,
        }
        np.savez_compressed(out_path, entropies=ent, meta=json.dumps(meta_dict))
        logger.info(
            f"[{args.tag}/{domain_label}] {ent.shape} entropies → {out_path.name} "
            f"({out_path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
