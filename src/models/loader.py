from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MoEMetadata:
    name: str
    num_routed_experts: int
    top_k: int
    num_moe_layers: int
    gate_module_class: str  # class name string for hook registration


def load_moe_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    logger.info(f"Loading tokenizer + model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Loaded {model_id}; dtype={dtype}; device_map={device_map}")
    return model, tokenizer


def get_moe_metadata(model) -> MoEMetadata:
    """Probe a loaded MoE model to discover its structural metadata.

    Works for DeepSeek-V2-Lite (gate class: ``MoEGate``) and Qwen2-MoE
    (gate class: ``Qwen2MoeSparseMoeBlock``).
    """
    cfg = model.config
    name = getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", "") or "unknown"

    # DeepSeek-V2: config has n_routed_experts + num_experts_per_tok + first_k_dense_replace
    if hasattr(cfg, "n_routed_experts"):
        num_experts = cfg.n_routed_experts
        top_k = cfg.num_experts_per_tok
        first_dense = getattr(cfg, "first_k_dense_replace", 0)
        num_moe_layers = cfg.num_hidden_layers - first_dense
        gate_cls = "MoEGate"
    # Qwen2-MoE / OLMoE: num_experts + num_experts_per_tok
    elif hasattr(cfg, "num_experts") and hasattr(cfg, "num_experts_per_tok"):
        num_experts = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        num_moe_layers = cfg.num_hidden_layers
        model_type = getattr(cfg, "model_type", "")
        if model_type == "olmoe":
            gate_cls = "OlmoeSparseMoeBlock"
        else:
            gate_cls = "Qwen2MoeSparseMoeBlock"
    else:
        raise ValueError(
            f"Unrecognized MoE config: no (n_routed_experts | num_experts) on {type(cfg).__name__}"
        )

    return MoEMetadata(
        name=name,
        num_routed_experts=num_experts,
        top_k=top_k,
        num_moe_layers=num_moe_layers,
        gate_module_class=gate_cls,
    )
