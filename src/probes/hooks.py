"""Forward hooks that capture per-token routing decisions from MoE gate modules.

Supports DeepSeek-V2 (``MoEGate``) and Qwen2-MoE (``Qwen2MoeSparseMoeBlock``).
The hook reads the model's chosen ``(topk_indices, topk_weights)`` and emits
one row per (token, top-k slot) into the attached :class:`TraceWriter`.

Hooks are stateless w.r.t. step / batch boundaries — the surrounding probe
script must call :meth:`TraceCollector.set_step` before each forward pass.
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from src.utils import get_logger

logger = get_logger(__name__)


class TraceCollector:
    """Registers gate-output hooks across all MoE layers and routes events to a writer."""

    def __init__(
        self,
        model: nn.Module,
        writer: "TraceWriter",  # noqa: F821 — forward ref; circular-import safe
        gate_module_class: str,
    ) -> None:
        self.model = model
        self.writer = writer
        self.gate_module_class = gate_module_class
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._step: int = 0
        self._batch_offset: int = 0  # base global token idx for the current step
        self._registered_layers = 0

    # ------------------------------------------------------------------ public API

    def set_step(self, step: int, batch_offset: int = 0) -> None:
        self._step = step
        self._batch_offset = batch_offset

    def register(self) -> int:
        """Walk the model and attach a hook to every gate module. Returns count."""
        layer_idx = 0
        for module in self.model.modules():
            if module.__class__.__name__ == self.gate_module_class:
                handle = module.register_forward_hook(self._make_hook(layer_idx))
                self._handles.append(handle)
                layer_idx += 1
        self._registered_layers = layer_idx
        logger.info(
            f"Registered {layer_idx} gate hooks (class={self.gate_module_class})"
        )
        return layer_idx

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "TraceCollector":
        if not self._handles:
            self.register()
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------ internals

    def _make_hook(self, layer_idx: int):
        gate_cls = self.gate_module_class

        def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
            try:
                topk_idx, topk_weight = self._extract_routing(gate_cls, module, inputs, output)
            except Exception as e:  # pragma: no cover — runtime diagnostic
                logger.warning(
                    f"Hook on layer {layer_idx} ({gate_cls}) failed to extract routing: {e}"
                )
                return
            self.writer.write_routing(
                step=self._step,
                layer=layer_idx,
                topk_idx=topk_idx,
                topk_weight=topk_weight,
                token_offset=self._batch_offset,
            )

        return _hook

    @staticmethod
    def _extract_routing(
        gate_cls: str,
        module: nn.Module,
        inputs: Any,
        output: Any,
    ):
        """Return ``(topk_idx, topk_weight)`` as 2-D long/float tensors of shape ``[N_tokens, top_k]``."""

        if gate_cls == "MoEGate":
            # DeepSeek-V2 MoEGate.forward returns (topk_idx, topk_weight, aux_loss)
            if isinstance(output, tuple) and len(output) >= 2:
                topk_idx, topk_weight = output[0], output[1]
            else:
                raise ValueError(f"DeepSeek MoEGate hook got non-tuple output: {type(output)}")
        elif gate_cls in ("Qwen2MoeSparseMoeBlock", "OlmoeSparseMoeBlock"):
            # Qwen2 / OLMoE SparseMoeBlock both do softmax + topk inside forward;
            # we recompute from the input hidden states using the block's own gate.
            hidden_states = inputs[0]
            if hidden_states.dim() == 3:
                n_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            else:
                n_tokens = hidden_states.shape[0]
            flat = hidden_states.view(n_tokens, -1)
            router_logits = module.gate(flat)
            weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
            topk_weight, topk_idx = torch.topk(weights, module.top_k, dim=-1)
        else:
            raise ValueError(f"Unsupported gate module class: {gate_cls}")

        topk_idx = topk_idx.detach().to(torch.int64).reshape(-1, topk_idx.shape[-1])
        topk_weight = topk_weight.detach().to(torch.float32).reshape(-1, topk_weight.shape[-1])
        return topk_idx, topk_weight
