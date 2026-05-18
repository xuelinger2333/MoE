#!/usr/bin/env python3
"""Self-contained H4 MoE locality experiment for OpenI Jupyter jobs.

Collects routing traces in memory from one MoE model, then reports:
  1. J_multi_turn / J_shared_prefix.
  2. Session-ID affinity vs MoE-Infinity-style EAMC vs random hit rate.

The default model is Qwen/Qwen1.5-MoE-A2.7B because it fits a 40GB A100.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


DEFAULT_CONVERSATIONS = [
    [
        ("user", "I am debugging a flaky CUDA memory issue in a PyTorch training loop."),
        ("assistant", "Start by logging allocated and reserved memory around each forward and backward step."),
        ("user", "The leak only appears when gradient checkpointing is enabled."),
        ("assistant", "Then isolate recomputation paths and verify no tensors are kept in closures."),
        ("user", "How should I design a minimal reproducer?"),
    ],
    [
        ("user", "Can you help write a dynamic programming solution for edit distance?"),
        ("assistant", "Use a two-dimensional table where dp[i][j] is the cost for prefixes."),
        ("user", "How do I reduce the memory usage to one row?"),
        ("assistant", "Keep the previous row and current row, updating from left to right."),
        ("user", "What edge cases should I test?"),
    ],
    [
        ("user", "Explain why TCP congestion control uses slow start."),
        ("assistant", "It probes available bandwidth conservatively after a connection begins."),
        ("user", "How does packet loss change the congestion window?"),
        ("assistant", "Traditional algorithms interpret loss as congestion and reduce the window."),
        ("user", "What about high latency satellite links?"),
    ],
    [
        ("user", "I need a SQL query for monthly active users."),
        ("assistant", "Group events by month and count distinct user identifiers."),
        ("user", "The table is partitioned by event_date."),
        ("assistant", "Add a date range predicate so the query prunes partitions."),
        ("user", "Can you add a retention cohort view too?"),
    ],
    [
        ("user", "Compare AdamW and SGD with momentum for transformer pretraining."),
        ("assistant", "AdamW is usually easier to tune, while SGD can require careful schedules."),
        ("user", "Why does decoupled weight decay matter?"),
        ("assistant", "It separates shrinkage from the adaptive gradient normalization."),
        ("user", "What learning-rate warmup would you try first?"),
    ],
    [
        ("user", "Write a Rust parser for a tiny arithmetic grammar."),
        ("assistant", "A recursive descent parser is a good fit for expressions and precedence."),
        ("user", "How do I report useful syntax errors?"),
        ("assistant", "Track spans and expected token sets at the point of failure."),
        ("user", "Can you sketch the AST types?"),
    ],
]

DEFAULT_PREFIXES = [
    "Write a Python function that sorts a list of integers using merge sort.",
    "Explain the difference between supervised and unsupervised learning in three paragraphs.",
    "Implement a binary search tree class with insert, search, and delete methods.",
    "Describe how the attention mechanism works in transformer models.",
    "Write a SQL query that returns the top 10 customers by total purchase amount.",
    "Compare and contrast TCP and UDP, with use cases for each.",
    "Implement Dijkstra's shortest-path algorithm in C++.",
    "Explain how the Linux kernel handles memory paging.",
    "Write a React component that fetches data from an API and displays it in a table.",
    "Discuss the trade-offs between gradient descent, SGD, and Adam optimizers.",
]


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


@dataclass
class UnitTrace:
    key: Tuple[int, int]
    expert_set: np.ndarray
    eam: np.ndarray


class InMemoryRouterTrace:
    def __init__(self, n_layers: int, n_experts: int):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self._events: List[Tuple[int, np.ndarray]] = []

    def clear(self) -> None:
        self._events.clear()

    def add(self, layer: int, topk_idx: torch.Tensor) -> None:
        arr = topk_idx.detach().to("cpu", dtype=torch.int64).reshape(-1).numpy()
        self._events.append((int(layer), arr))

    def finalize(self, key: Tuple[int, int]) -> UnitTrace:
        eam = np.zeros((self.n_layers, self.n_experts), dtype=np.float32)
        experts: List[np.ndarray] = []
        for layer, arr in self._events:
            if arr.size == 0:
                continue
            experts.append(arr)
            valid = arr[(arr >= 0) & (arr < self.n_experts)]
            np.add.at(eam[layer], valid, 1.0)
        if experts:
            expert_set = np.unique(np.concatenate(experts))
        else:
            expert_set = np.array([], dtype=np.int64)
        return UnitTrace(key=key, expert_set=expert_set, eam=eam)


class RouterHook:
    def __init__(self, model, trace: InMemoryRouterTrace, gate_class: str):
        self.model = model
        self.trace = trace
        self.gate_class = gate_class
        self.handles = []

    def register(self) -> int:
        layer = 0
        for module in self.model.modules():
            if module.__class__.__name__ == self.gate_class:
                self.handles.append(module.register_forward_hook(self._make_hook(layer)))
                layer += 1
        return layer

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            try:
                topk_idx = extract_topk(module, inputs, output, self.gate_class)
                self.trace.add(layer, topk_idx)
            except Exception as exc:
                log(f"WARN hook failed at layer {layer}: {type(exc).__name__}: {exc}")

        return hook


def extract_topk(module, inputs, output, gate_class: str) -> torch.Tensor:
    if gate_class == "MoEGate":
        return output[0]
    hidden_states = inputs[0]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if not hasattr(module, "gate"):
        raise ValueError(f"{gate_class} has no .gate")
    router_logits = module.gate(flat)
    top_k = getattr(module, "top_k", None) or getattr(module, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(module.config, "num_experts_per_tok", None)
    if top_k is None:
        raise ValueError(f"cannot infer top_k for {gate_class}")
    return torch.topk(torch.softmax(router_logits, dim=-1, dtype=torch.float), int(top_k), dim=-1).indices


def infer_moe_meta(model) -> Tuple[int, int, int, str]:
    cfg = model.config
    if hasattr(cfg, "n_routed_experts"):
        n_experts = int(cfg.n_routed_experts)
        top_k = int(cfg.num_experts_per_tok)
        n_layers = int(cfg.num_hidden_layers - getattr(cfg, "first_k_dense_replace", 0))
        return n_layers, n_experts, top_k, "MoEGate"
    n_experts = int(getattr(cfg, "num_experts"))
    top_k = int(getattr(cfg, "num_experts_per_tok"))
    model_type = getattr(cfg, "model_type", "")
    candidates = [
        "Qwen2MoeSparseMoeBlock",
        "Qwen3MoeSparseMoeBlock",
        "OlmoeSparseMoeBlock",
        "MixtralSparseMoeBlock",
    ]
    names = {m.__class__.__name__ for m in model.modules()}
    gate_class = next((c for c in candidates if c in names), None)
    if gate_class is None:
        gate_class = next((n for n in names if "Moe" in n and "Sparse" in n), None)
    if gate_class is None:
        raise ValueError(f"cannot find MoE block class; model_type={model_type}")
    n_layers = sum(1 for m in model.modules() if m.__class__.__name__ == gate_class)
    return n_layers, n_experts, top_k, gate_class


def apply_chat_template(tok, messages: List[Tuple[str, str]], fallback_join: bool = True) -> str:
    msgs = [{"role": r, "content": c} for r, c in messages]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    if fallback_join:
        return "\n".join(f"<|{r}|>\n{c}" for r, c in messages)
    return messages[-1][1]


def forward_unit(model, tok, trace: InMemoryRouterTrace, text: str, max_tokens: int, key: Tuple[int, int]) -> UnitTrace:
    device = next(model.parameters()).device
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
    trace.clear()
    with torch.no_grad():
        _ = model(ids)
    out = trace.finalize(key)
    trace.clear()
    return out


def generate_unit(
    model, tok, trace: InMemoryRouterTrace, prompt: str, max_input_tokens: int,
    max_new_tokens: int, temperature: float, top_p: float, seed: int, key: Tuple[int, int],
) -> UnitTrace:
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).input_ids.to(device)
    trace.clear()
    torch.manual_seed(seed)
    with torch.no_grad():
        _ = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    out = trace.finalize(key)
    trace.clear()
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    u = np.union1d(a, b).size
    if u == 0:
        return 0.0
    return float(np.intersect1d(a, b, assume_unique=True).size / u)


def stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def analyze_pairs(units: Dict[Tuple[int, int], UnitTrace], adjacent: bool, n_between: int, seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    by_group: Dict[int, List[Tuple[int, int]]] = {}
    for key in units:
        by_group.setdefault(key[0], []).append(key)
    within = []
    for keys in by_group.values():
        keys = sorted(keys)
        if adjacent:
            pairs = list(zip(keys[:-1], keys[1:]))
        else:
            pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
        for a, b in pairs:
            within.append(jaccard(units[a].expert_set, units[b].expert_set))
    groups = sorted(by_group)
    between = []
    if len(groups) >= 2:
        for _ in range(n_between):
            g1, g2 = rng.choice(groups, 2, replace=False)
            a = by_group[int(g1)][int(rng.integers(len(by_group[int(g1)])))]
            b = by_group[int(g2)][int(rng.integers(len(by_group[int(g2)])))]
            between.append(jaccard(units[a].expert_set, units[b].expert_set))
    return {"within": stats(within), "between": stats(between)}


def hot_experts(eam: np.ndarray, top_pct: float) -> np.ndarray:
    k = max(1, int(math.ceil(top_pct * eam.shape[1])))
    return np.unique(np.concatenate([np.argsort(-eam[l])[:k] for l in range(eam.shape[0])]))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def fit_kmeans(eams: List[np.ndarray], capacity: int, seed: int) -> List[np.ndarray]:
    if not eams:
        return []
    rng = np.random.default_rng(seed)
    flats = np.stack([x.reshape(-1) for x in eams])
    n = flats.shape[0]
    k = min(capacity, n)
    centroids = [flats[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(np.stack([np.sum((flats - c) ** 2, axis=1) for c in centroids]), axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        centroids.append(flats[int(rng.choice(n, p=probs))])
    C = np.stack(centroids)
    for _ in range(15):
        d = np.stack([np.sum((flats - C[i]) ** 2, axis=1) for i in range(k)])
        assign = np.argmin(d, axis=0)
        new = C.copy()
        for i in range(k):
            mask = assign == i
            if mask.any():
                new[i] = flats[mask].mean(axis=0)
        if np.allclose(new, C):
            break
        C = new
    return [C[i].reshape(eams[0].shape) for i in range(k)]


def hit(actual: np.ndarray, pred: np.ndarray) -> float:
    if actual.size == 0:
        return 0.0
    return float(np.intersect1d(actual, pred, assume_unique=True).size / actual.size)


def evaluate_affinity(mt_units: Dict[Tuple[int, int], UnitTrace], n_experts: int, capacity: int, top_pct: float, seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    convs = sorted({k[0] for k in mt_units})
    rng.shuffle(convs)
    n_cal = max(1, int(0.3 * len(convs)))
    cal = set(convs[:n_cal])
    eval_convs = [c for c in convs if c not in cal]
    centroids = fit_kmeans([u.eam for k, u in mt_units.items() if k[0] in cal], capacity, seed)
    hits = {"session_id": [], "eamc": [], "random": []}
    sizes = {"session_id": [], "eamc": [], "random": []}
    for c in eval_convs:
        keys = sorted(k for k in mt_units if k[0] == c)
        running = None
        for i, key in enumerate(keys):
            cur = mt_units[key]
            if i == 0:
                running = cur.eam.copy()
                continue
            prev = mt_units[keys[i - 1]]
            actual = cur.expert_set
            s_pred = hot_experts(prev.eam, top_pct)
            if centroids:
                flat = running.reshape(-1)
                d = [cosine_distance(cen.reshape(-1), flat) for cen in centroids]
                e_pred = hot_experts(centroids[int(np.argmin(d))], top_pct)
            else:
                e_pred = np.array([], dtype=np.int64)
            r_pred = rng.choice(n_experts, max(1, int(math.ceil(top_pct * n_experts))), replace=False)
            for name, pred in [("session_id", s_pred), ("eamc", e_pred), ("random", r_pred)]:
                hits[name].append(hit(actual, pred))
                sizes[name].append(int(pred.size))
            running = running + cur.eam
    out = {}
    for name, vals in hits.items():
        s = stats(vals)
        s["avg_predicted_set_size"] = float(np.mean(sizes[name])) if sizes[name] else 0.0
        out[name] = s
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--out", default="outputs/h4_openi")
    p.add_argument("--convs", type=int, default=12)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--prefixes", type=int, default=8)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--max-turn-tokens", type=int, default=192)
    p.add_argument("--max-prefix-tokens", type=int, default=96)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-pct", type=float, default=0.3)
    p.add_argument("--eamc-capacity", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.convs = min(args.convs, 3)
        args.turns = min(args.turns, 3)
        args.prefixes = min(args.prefixes, 3)
        args.branches = min(args.branches, 2)
        args.max_turn_tokens = min(args.max_turn_tokens, 96)
        args.max_new_tokens = min(args.max_new_tokens, 32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading tokenizer/model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_layers, n_experts, top_k, gate_class = infer_moe_meta(model)
    log(f"MoE meta: layers={n_layers}, experts={n_experts}, top_k={top_k}, hook={gate_class}")

    trace = InMemoryRouterTrace(n_layers, n_experts)
    hook = RouterHook(model, trace, gate_class)
    n_hook = hook.register()
    if n_hook <= 0:
        raise RuntimeError("No MoE hooks registered")

    mt_units: Dict[Tuple[int, int], UnitTrace] = {}
    conversations = (DEFAULT_CONVERSATIONS * ((args.convs // len(DEFAULT_CONVERSATIONS)) + 1))[: args.convs]
    log(f"Collecting multi-turn units: {len(conversations)} convs x {args.turns} turns, turn-only prompts")
    for c, conv in enumerate(conversations):
        for t, msg in enumerate(conv[: args.turns]):
            text = apply_chat_template(tok, [msg], fallback_join=False)
            mt_units[(c, t)] = forward_unit(model, tok, trace, text, args.max_turn_tokens, (c, t))
        log(f"  multi-turn conv {c + 1}/{len(conversations)} done")

    sp_units: Dict[Tuple[int, int], UnitTrace] = {}
    prefixes = DEFAULT_PREFIXES[: args.prefixes]
    log(f"Collecting shared-prefix units: {len(prefixes)} prefixes x {args.branches} branches")
    for pfx_id, prompt in enumerate(prefixes):
        for b in range(args.branches):
            sp_units[(pfx_id, b)] = generate_unit(
                model, tok, trace, prompt, args.max_prefix_tokens, args.max_new_tokens,
                temperature=0.7, top_p=0.9, seed=args.seed + 1000 * pfx_id + b, key=(pfx_id, b),
            )
        log(f"  shared-prefix {pfx_id + 1}/{len(prefixes)} done")

    hook.remove()

    mt = analyze_pairs(mt_units, adjacent=True, n_between=2000, seed=args.seed)
    sp = analyze_pairs(sp_units, adjacent=False, n_between=2000, seed=args.seed)
    j_mt = mt["within"]["mean"]
    j_sp = sp["within"]["mean"]
    h4_ratio = float(j_mt / j_sp) if j_sp > 1e-12 else 0.0
    affinity = evaluate_affinity(mt_units, n_experts, args.eamc_capacity, args.top_pct, args.seed)
    sess = affinity["session_id"]["mean"]
    eamc = affinity["eamc"]["mean"]
    sess_eamc_ratio = float(sess / eamc) if eamc > 1e-12 else 0.0

    summary = {
        "model": args.model,
        "args": vars(args),
        "moe": {"n_layers": n_layers, "n_experts": n_experts, "top_k": top_k, "gate_class": gate_class},
        "multi_turn": mt,
        "shared_prefix": sp,
        "h4_ratio": {
            "j_multi_turn_within": j_mt,
            "j_shared_prefix_within": j_sp,
            "ratio": h4_ratio,
        },
        "affinity": affinity,
        "session_id_over_eamc": sess_eamc_ratio,
    }
    path = out_dir / ("summary_smoke.json" if args.smoke else "summary.json")
    path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote {path}")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
