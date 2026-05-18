"""Collect shared-prefix branched traces for H4 — reproduces arXiv 2604.17182.

Each prefix is sampled N times with temperature; each sample is a separate
"branch". Branches share the prefix's KV cache prefix but diverge after.

Output layout:
    outputs/traces/h4_shared_prefix_{model}/prefix_{NNNN}/shard_0000.parquet
                                          /run_meta.json

step field == branch_id; prefix_id is parent dir prefix_NNNN.

NOT auto-deployed — explicit GPU run needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import get_moe_metadata, load_moe_model  # noqa: E402
from src.probes import TraceCollector, TraceWriter  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("probe.shared_prefix")


# ---------------------------------------------------------------------------
# Default prefix set (mix of code-gen + dialogue per arXiv 2604.17182 spirit)
# ---------------------------------------------------------------------------

DEFAULT_PREFIXES = [
    "Write a Python function that sorts a list of integers using merge sort.",
    "Explain the difference between supervised and unsupervised learning in 3 paragraphs.",
    "Implement a binary search tree class with insert, search, and delete methods.",
    "Describe how the attention mechanism works in transformer models.",
    "Write a SQL query that returns the top 10 customers by total purchase amount.",
    "Compare and contrast TCP and UDP, with use cases for each.",
    "Implement Dijkstra's shortest-path algorithm in C++.",
    "Explain how the Linux kernel handles memory paging.",
    "Write a React component that fetches data from an API and displays it in a table.",
    "Discuss the trade-offs between gradient descent, SGD, and Adam optimizers.",
]


def load_prefixes(path: Path | None) -> List[str]:
    if path is None:
        return DEFAULT_PREFIXES
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--prefixes_file", type=Path, default=None,
                   help="text file, one prefix per line. Default: built-in set.")
    p.add_argument("--n_branches", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    tag = args.tag or (
        "qwen" if "Qwen" in args.model else
        "olmoe" if "OLMoE" in args.model else
        "deepseek" if "DeepSeek" in args.model else "model"
    )
    out = args.out or (REPO_ROOT / "outputs/traces" / f"h4_shared_prefix_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    prefixes = load_prefixes(args.prefixes_file)
    logger.info(f"Using {len(prefixes)} prefixes, {args.n_branches} branches each.")

    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    device = next(model.parameters()).device

    t0 = time.time()
    prefix_meta: List[dict] = []

    for prefix_id, prompt in enumerate(prefixes):
        pref_dir = out / f"prefix_{prefix_id:04d}"
        pref_dir.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(out_dir=pref_dir, flush_every_rows=200_000)
        collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
        collector.register()

        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        # Generate N branches with different seeds
        for branch_id in range(args.n_branches):
            torch.manual_seed(args.seed + 1000 * prefix_id + branch_id)
            collector.set_step(step=branch_id, batch_offset=branch_id * args.max_new_tokens * 4)
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                )

        collector.remove()
        writer.close()
        prefix_meta.append({"prefix_id": prefix_id, "prompt": prompt[:80]})
        logger.info(f"Prefix {prefix_id}/{len(prefixes)} done in {time.time() - t0:.0f}s")

    (out / "run_meta.json").write_text(json.dumps({
        "model": args.model,
        "tag": tag,
        "n_prefixes": len(prefixes),
        "n_branches": args.n_branches,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_routed_experts": meta.num_routed_experts,
        "top_k": meta.top_k,
        "num_moe_layers": meta.num_moe_layers,
        "prefixes": prefix_meta,
        "schema_note": "step field == branch_id; prefix_id is parent dir prefix_NNNN",
    }, indent=2))
    logger.info(f"Done in {time.time() - t0:.0f}s; out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
