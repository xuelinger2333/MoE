"""Collect multi-turn routing traces for H4.

Loads multi-turn conversation dataset (ShareGPT / WildChat / UltraChat),
filters to ≥5-turn conversations, runs each conversation through the MoE
model turn-by-turn (each turn = one ``step`` in TraceWriter), and writes
per-conversation parquet directories.

Output layout:
    outputs/traces/h4_multiturn_{model}/conv_{NNNN}/shard_0000.parquet
                                    /run_meta.json   (model + dataset info)

Each parquet shard carries the existing schema (step, layer, token_idx_in_run,
expert_id, topk_rank, weight). The ``step`` value IS the turn_id, and the
conv_id is inferred from the parent directory name.

NOT auto-deployed by /experiment-bridge — explicit GPU run needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import get_moe_metadata, load_moe_model  # noqa: E402
from src.probes import TraceCollector, TraceWriter  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("probe.multiturn")


# ---------------------------------------------------------------------------
# Conversation loader
# ---------------------------------------------------------------------------

def load_conversations(
    dataset: str, min_turns: int = 5, max_convs: int = 200,
    max_turn_tokens: int = 256, seed: int = 42,
) -> Iterator[Tuple[int, List[Dict[str, str]]]]:
    """Yield (conv_id, conversation) where conversation is a list of {role, content}.

    Supports:
      sharegpt  : anon8231489123/ShareGPT_Vicuna_unfiltered  (HF)
      wildchat  : allenai/WildChat-1M  (HF, requires gated access in some cases)
      ultrachat : HuggingFaceH4/ultrachat_200k  (HF)
    """
    from datasets import load_dataset

    if dataset == "sharegpt":
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          split="train", streaming=True)
        key_conv = "conversations"
        role_key = "from"; content_key = "value"
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    elif dataset == "wildchat":
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        key_conv = "conversation"
        role_key = "role"; content_key = "content"
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
    elif dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        key_conv = "messages"
        role_key = "role"; content_key = "content"
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    n = 0
    cid = 0
    for ex in ds:
        conv = ex.get(key_conv) or []
        # Normalise roles
        turns = []
        for msg in conv:
            r = role_map.get(msg.get(role_key, ""), "user")
            c = (msg.get(content_key) or "").strip()
            if c:
                turns.append({"role": r, "content": c})
        # Filter: ≥ min_turns and at least one assistant turn
        if len([t for t in turns if t["role"] == "assistant"]) < min_turns // 2:
            continue
        if len(turns) < min_turns:
            continue
        yield cid, turns
        cid += 1
        n += 1
        if n >= max_convs:
            return


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--dataset", default="sharegpt",
                   choices=["sharegpt", "wildchat", "ultrachat"])
    p.add_argument("--max_convs", type=int, default=200)
    p.add_argument("--min_turns", type=int, default=5)
    p.add_argument("--max_turn_tokens", type=int, default=256)
    p.add_argument("--out", type=Path, default=None,
                   help="default: outputs/traces/h4_multiturn_<tag>/")
    p.add_argument("--tag", default=None, help="model tag (qwen/olmoe/deepseek). Auto-inferred.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    tag = args.tag or (
        "qwen" if "Qwen" in args.model else
        "olmoe" if "OLMoE" in args.model else
        "deepseek" if "DeepSeek" in args.model else "model"
    )
    out = args.out or (REPO_ROOT / "outputs/traces" / f"h4_multiturn_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model {args.model}")
    model, tok = load_moe_model(args.model)
    meta = get_moe_metadata(model)
    device = next(model.parameters()).device

    # Per-conversation trace collection
    t0 = time.time()
    n_collected = 0
    convs_meta: List[Dict] = []

    for conv_id, turns in load_conversations(
        args.dataset, min_turns=args.min_turns,
        max_convs=args.max_convs, max_turn_tokens=args.max_turn_tokens,
        seed=args.seed,
    ):
        conv_dir = out / f"conv_{conv_id:04d}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(out_dir=conv_dir, flush_every_rows=200_000)
        collector = TraceCollector(model, writer, gate_module_class=meta.gate_module_class)
        collector.register()

        # Build progressive context: turn k uses tokens [turn_1..turn_k] concatenated
        running_prompt = ""
        n_turns_done = 0
        for turn_id, msg in enumerate(turns):
            running_prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
            # Tokenize ONLY the new turn's text — model still sees concat via prefix
            ids = tok(running_prompt, return_tensors="pt", truncation=True,
                      max_length=args.max_turn_tokens * (turn_id + 1)).input_ids.to(device)
            collector.set_step(step=turn_id, batch_offset=turn_id * args.max_turn_tokens * 8)
            with torch.no_grad():
                _ = model(ids)
            n_turns_done += 1
            if n_turns_done >= 10:  # cap per-conv turns for budget
                break

        collector.remove()
        writer.close()
        convs_meta.append({"conv_id": conv_id, "n_turns": n_turns_done})
        n_collected += 1
        if n_collected % 10 == 0:
            logger.info(f"Collected {n_collected} convs in {time.time() - t0:.0f}s")

    (out / "run_meta.json").write_text(json.dumps({
        "model": args.model,
        "tag": tag,
        "dataset": args.dataset,
        "num_routed_experts": meta.num_routed_experts,
        "top_k": meta.top_k,
        "num_moe_layers": meta.num_moe_layers,
        "n_conversations": n_collected,
        "max_turn_tokens": args.max_turn_tokens,
        "conversations": convs_meta,
        "schema_note": "step field == turn_id; conversation_id is parent dir conv_NNNN",
    }, indent=2))
    logger.info(f"Done: {n_collected} convs in {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
