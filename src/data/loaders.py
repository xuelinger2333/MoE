"""Calibration-data loaders.

We pack each dataset's text into fixed-length token sequences to keep batches
uniform. Yields ``input_ids`` tensors of shape ``[batch_size, seq_len]``.

C4 is HUGE (~300 GB compressed); it is loaded in streaming mode so we never
materialize more than the bytes we actually consume.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from src.utils import get_logger

logger = get_logger(__name__)


# Map of friendly name -> (HF id, config, split, text_field, slice_size, streaming)
_DATASETS = {
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation", "text", None, False),
    "c4": ("allenai/c4", "en", "train", "text", 2000, True),
    "stack": ("bigcode/the-stack-smol", "default", "train", "content", 200, False),
    "mmlu": ("cais/mmlu", "all", "dev", None, None, False),
    # Cross-domain stability probe (G6 round-3): code + math
    "code": ("codeparrot/codeparrot-clean-valid", None, "train", "content", 5000, True),
    "math": ("openai/gsm8k", "main", "train", None, None, False),
}


def get_dataset(name: str, slice_size: int | None = None):
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {list(_DATASETS)}")
    hf_id, cfg, split, text_field, default_slice, streaming = _DATASETS[name]
    n = slice_size if slice_size is not None else default_slice
    logger.info(
        f"Loading dataset {name} ({hf_id}/{cfg} split={split}, slice={n}, streaming={streaming})"
    )

    if name == "stack":
        # the-stack-smol has per-language sub-paths; pull only Python.
        ds = load_dataset(hf_id, data_dir="data/python", split=split, streaming=False)
    elif cfg is None:
        ds = load_dataset(hf_id, split=split, streaming=streaming)
    else:
        ds = load_dataset(hf_id, cfg, split=split, streaming=streaming)

    if n is not None:
        if streaming:
            ds = ds.take(n)
        else:
            ds = ds.select(range(min(n, len(ds))))

    return ds, text_field


def _iter_text(ds, text_field: str | None, name: str = "") -> Iterator[str]:
    for ex in ds:
        if text_field is not None:
            t = ex.get(text_field, "")
            if t:
                yield t
        elif name == "math":
            # GSM8K: question + chain-of-thought answer
            q = ex.get("question", "")
            a = ex.get("answer", "")
            if q and a:
                yield f"Question: {q}\nAnswer: {a}\n"
        else:
            # MMLU fallback: build "Q: ...\nA: <correct choice>"
            q = ex.get("question", "")
            choices = ex.get("choices", [])
            ans_idx = ex.get("answer", 0)
            ans = choices[ans_idx] if 0 <= ans_idx < len(choices) else ""
            yield f"Q: {q}\nChoices: {' / '.join(choices)}\nA: {ans}\n"


def iter_packed_batches(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    slice_size: int | None = None,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield ``(batch_idx, input_ids)`` of shape ``[batch_size, seq_len]``.

    Tokens from the dataset are concatenated and chopped into ``seq_len`` chunks;
    we then group ``batch_size`` chunks per batch. EOS is inserted between docs.
    """
    ds, text_field = get_dataset(name, slice_size)
    tokens_per_batch = batch_size * seq_len

    buf: list[int] = []
    batch_idx = 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for text in _iter_text(ds, text_field, name=name):
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos_id)

        while len(buf) >= tokens_per_batch:
            chunk = buf[:tokens_per_batch]
            buf = buf[tokens_per_batch:]
            input_ids = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)
            yield batch_idx, input_ids
            batch_idx += 1
            if batch_idx >= n_batches:
                return

    if batch_idx < n_batches:
        logger.warning(
            f"Dataset {name} exhausted after {batch_idx} batches "
            f"(requested {n_batches}); reduce batch_size or seq_len, or pull a larger slice"
        )
