import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch


@dataclass
class AblationSpec:
    ablation_type: str  # 'heads' | 'layers' | 'emb_mask'
    severity: float  # fraction in [0,1]
    selection_policy: str  # 'random' | 'consecutive'
    seeds: List[int]
    # details populated after selection
    selection_indices: Dict[str, Any]
    component: Optional[str] = None  # for layers: 'attn'|'mlp'|'block'


def _select_indices_random(total: int, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    return sorted(idxs[:k])


def _select_indices_consecutive(total: int, k: int, start: int = 0) -> List[int]:
    return list(range(start, min(start + k, total)))


def plan_head_selection(num_layers: int, num_heads: int, severity: float, policy: str, seed: int) -> Dict[int, List[int]]:
    k = int(round(num_heads * severity))
    plan: Dict[int, List[int]] = {}
    for layer in range(num_layers):
        if policy == 'random':
            plan[layer] = _select_indices_random(num_heads, k, seed + layer)
        else:
            plan[layer] = _select_indices_consecutive(num_heads, k, start=0)
    return plan


def plan_layer_selection(num_layers: int, severity: float, policy: str, seed: int) -> List[int]:
    k = int(round(num_layers * severity))
    if policy == 'random':
        return _select_indices_random(num_layers, k, seed)
    else:
        return _select_indices_consecutive(num_layers, k, start=0)


def make_head_zero_hook(hidden_size: int, num_heads: int, heads_to_zero: List[int]):
    head_dim = hidden_size // num_heads
    channels: List[Tuple[int, int]] = []
    for h in heads_to_zero:
        channels.append((h * head_dim, (h + 1) * head_dim))

    def hook(module, inputs, output):
        # output: [bsz, seq, hidden]
        if not channels:
            return output
        out = output
        for s, e in channels:
            out[..., s:e] = 0.0
        return out

    return hook


def make_zero_output_hook():
    def hook(module, inputs, output):
        return torch.zeros_like(output)
    return hook


def make_embedding_mask_hook(frac: float):
    def hook(module, inputs, output):
        # output: [bsz, seq, hidden]
        hidden = output.size(-1)
        k = int(round(hidden * frac))
        if k <= 0:
            return output
        out = output
        out[..., :k] = 0.0
        return out
    return hook


@contextmanager
def apply_ablation_hooks(model, spec: AblationSpec):
    """
    Registers forward hooks according to the ablation spec.
    Yields the handles list; on exit, all hooks are removed.
    """
    handles = []
    try:
        if spec.ablation_type in ('none', None):
            # No ablation; leave model unmodified
            spec.selection_indices = {}
            yield handles
            return
        elif spec.ablation_type == 'heads':
            # Expect model.model.layers[*].self_attn.o_proj output to be [bsz, seq, hidden]
            num_layers = model.config.num_hidden_layers
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
            plan = plan_head_selection(num_layers, num_heads, spec.severity, spec.selection_policy, spec.seeds[0])
            spec.selection_indices = {str(k): v for k, v in plan.items()}
            for layer_idx, heads in plan.items():
                attn = model.model.layers[layer_idx].self_attn
                # Register on the attention output projection forward: hook on o_proj forward output
                o_proj = attn.o_proj
                handles.append(o_proj.register_forward_hook(make_head_zero_hook(hidden_size, num_heads, heads)))
        elif spec.ablation_type == 'layers':
            num_layers = model.config.num_hidden_layers
            layers_to_zero = plan_layer_selection(num_layers, spec.severity, spec.selection_policy, spec.seeds[0])
            spec.selection_indices = {"layers": layers_to_zero, "component": spec.component or "block"}
            for layer_idx in layers_to_zero:
                layer = model.model.layers[layer_idx]
                comp = (spec.component or 'block').lower()
                if comp == 'attn':
                    handles.append(layer.self_attn.o_proj.register_forward_hook(make_zero_output_hook()))
                elif comp == 'mlp':
                    handles.append(layer.mlp.down_proj.register_forward_hook(make_zero_output_hook()))
                else:
                    # Hook the whole block output
                    handles.append(layer.register_forward_hook(make_zero_output_hook()))
        elif spec.ablation_type == 'emb_mask':
            frac = spec.severity
            spec.selection_indices = {"masked_frac": frac}
            emb = model.model.embed_tokens
            handles.append(emb.register_forward_hook(make_embedding_mask_hook(frac)))
        else:
            raise ValueError(f"Unknown ablation_type: {spec.ablation_type}")
        yield handles
    finally:
        for h in handles:
            h.remove()
