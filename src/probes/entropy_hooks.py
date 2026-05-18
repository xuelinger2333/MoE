"""Forward hook that captures FULL-distribution router entropy per (token, layer).

This is orthogonal to TraceCollector (which captures top-k routing decisions):
EntropyCollector recomputes the *full* softmax over all experts inside the
hook, computes its Shannon entropy in nats, and emits one float per
(token, layer). The full per-token expert distribution is never persisted —
only its entropy.

Designed to use the SAME data + seed + batch sizes as the M1/M8 routing
probes so that token_idx_in_run aligns one-to-one with those existing
traces (enables filtered-MI joins downstream).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

logger = get_logger(__name__)


def _softmax_entropy_nats(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats along the last dim. Input shape [..., n]."""
    p = probs.clamp_min(1e-30)
    return -(p * p.log()).sum(dim=-1)


class EntropyCollector:
    """Registers gate hooks; emits per-(token, layer) full-softmax entropy."""

    def __init__(
        self,
        model: nn.Module,
        gate_module_class: str,
        num_routed_experts: int,
        num_moe_layers: int,
    ) -> None:
        self.model = model
        self.gate_module_class = gate_module_class
        self.num_routed_experts = num_routed_experts
        self.num_moe_layers = num_moe_layers
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._step: int = 0
        self._token_offset: int = 0
        # Buffer of (step, layer, token_idx_in_run, entropy) as flat lists,
        # consolidated per flush into a numpy structured array.
        self._buffers: Dict[int, List[np.ndarray]] = {}  # layer -> list of [n_tokens] arrays
        self._step_token_starts: Dict[int, int] = {}  # step -> token_offset
        self._registered = 0

    # ---------- lifecycle ----------

    def set_step(self, step: int, token_offset: int) -> None:
        self._step = step
        self._token_offset = token_offset
        self._step_token_starts.setdefault(step, token_offset)

    def register(self) -> int:
        layer_idx = 0
        for module in self.model.modules():
            if module.__class__.__name__ == self.gate_module_class:
                handle = module.register_forward_hook(self._make_hook(layer_idx))
                self._handles.append(handle)
                self._buffers[layer_idx] = []
                layer_idx += 1
        self._registered = layer_idx
        logger.info(
            f"Registered {layer_idx} entropy hooks (class={self.gate_module_class})"
        )
        return layer_idx

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "EntropyCollector":
        if not self._handles:
            self.register()
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()

    # ---------- result ----------

    def consolidate(self) -> np.ndarray:
        """Return a [n_tokens, n_layers] float32 array of per-token entropy.

        Tokens are ordered by global token_idx_in_run starting at the smallest
        token_offset seen during this run.
        """
        n_layers = self._registered
        # Each layer's buffer is a list of [n_tokens_in_step] arrays in step order
        per_layer = [np.concatenate(self._buffers[L]).astype(np.float32) for L in range(n_layers)]
        n_tokens = per_layer[0].shape[0]
        for L in range(1, n_layers):
            assert per_layer[L].shape[0] == n_tokens, (
                f"Layer {L} has {per_layer[L].shape[0]} tokens, layer 0 has {n_tokens}"
            )
        out = np.stack(per_layer, axis=1)  # [n_tokens, n_layers]
        return out

    # ---------- hook ----------

    def _make_hook(self, layer_idx: int):
        gate_cls = self.gate_module_class

        def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
            try:
                entropy_per_token = self._compute_entropy(gate_cls, module, inputs, output)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Entropy hook on layer {layer_idx} ({gate_cls}) failed: {e}")
                return
            self._buffers[layer_idx].append(entropy_per_token.cpu().numpy())

        return _hook

    @staticmethod
    def _compute_entropy(gate_cls: str, module: nn.Module, inputs: Any, output: Any) -> torch.Tensor:
        """Return [n_tokens] tensor of full-softmax entropy in nats."""
        if gate_cls == "MoEGate":
            # DeepSeek-V2 MoEGate. Recompute full softmax from inputs since the
            # gate output is renormalized to top-k only.
            hidden = inputs[0]
            if hidden.dim() == 3:
                flat = hidden.reshape(-1, hidden.shape[-1])
            else:
                flat = hidden
            # Match DeepSeek-V2's casting pattern
            logits = F.linear(flat.to(module.weight.dtype), module.weight)
            scoring_func = getattr(module, "scoring_func", "softmax")
            if scoring_func == "softmax":
                probs = F.softmax(logits.float(), dim=-1)
            else:  # sigmoid scoring (some variants)
                probs = torch.sigmoid(logits.float())
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            return _softmax_entropy_nats(probs)

        elif gate_cls in ("Qwen2MoeSparseMoeBlock", "OlmoeSparseMoeBlock"):
            hidden = inputs[0]
            if hidden.dim() == 3:
                flat = hidden.reshape(-1, hidden.shape[-1])
            else:
                flat = hidden
            logits = module.gate(flat)
            probs = F.softmax(logits.float(), dim=-1)
            return _softmax_entropy_nats(probs)

        else:
            raise ValueError(f"Unsupported gate module class: {gate_cls}")
