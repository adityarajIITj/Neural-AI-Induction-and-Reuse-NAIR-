"""Transparent, documented FLOP and parameter accounting."""
from __future__ import annotations
import torch
from torch import nn

def count_params(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_active_params(model: nn.Module, kind: str) -> int:
    """Estimated active parameters per computation step."""
    if kind == "mlp":
        return count_params(model)
    elif kind == "shared":
        return count_params(model)
    elif kind == "explicit":
        # At any single step, exactly one unit is activated for a given operation
        if hasattr(model, "units") and model.units is not None and len(model.units) > 0:
            return count_params(model.units[0])
        return count_params(model)
    elif kind == "library":
        # With straight-through hard routing, one unit is active plus the router logits
        router_params = model.router_logits.numel() if hasattr(model, "router_logits") and model.router_logits is not None else 0
        unit_params = count_params(model.units[0]) if hasattr(model, "units") and model.units is not None else 0
        return unit_params + router_params
    elif kind in ("soft_library", "hypernet"):
        return count_params(model)
    return count_params(model)

def linear_forward_flops(module: nn.Module, batch: int) -> int:
    """Count matrix multiplication FLOPs for all nn.Linear layers in module.

    Convention: 1 multiply-accumulate = 2 FLOPs (1 multiply, 1 addition).
    Explicitly excludes bias additions, activations, softmax, normalization, and indexing.
    """
    total = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            total += 2 * batch * layer.in_features * layer.out_features
    return total

def estimate_forward_flops(model: nn.Module, batch_size: int, task_depth: int = 3, kind: str = "mlp") -> int:
    """Estimate forward matrix-multiplication FLOPs for a batch of depth `task_depth`."""
    if kind == "mlp":
        return linear_forward_flops(model, batch_size)

    if kind == "hypernet":
        # Hypernetwork generates W1 and W2, then performs batch matmuls
        # Gen W1: embed_dim -> n_states * hidden; Gen W2: embed_dim -> hidden * n_states
        gen_flops = linear_forward_flops(model.hyper_step, batch_size)
        # BMM 1: [B, 1, n_states] @ [B, n_states, hidden] -> 2 * B * n_states * hidden
        # BMM 2: [B, 1, hidden] @ [B, hidden, n_states] -> 2 * B * hidden * n_states
        s = model.n_states
        h = model.hyper_step.hidden
        bmm_flops = (2 * batch_size * s * h) + (2 * batch_size * h * s)
        step_flops = gen_flops + bmm_flops
        return step_flops * task_depth

    if hasattr(model, "units") and model.units is not None:
        fwd_per_unit = sum(linear_forward_flops(u, batch_size) for u in model.units)
        if kind == "shared":
            return fwd_per_unit * task_depth
        elif kind in ("explicit", "library", "soft_library"):
            # All units are evaluated in current vectorized tensor implementation
            return fwd_per_unit * task_depth

    return linear_forward_flops(model, batch_size)

def estimate_train_flops(model: nn.Module, batch_size: int, steps: int, task_depth: int = 3, kind: str = "mlp") -> int:
    """Estimate total training FLOPs (Forward + Backward passes).

    Standard ML convention: Backward pass requires approximately 2x forward FLOPs
    (weight gradients + activation gradients), so 1 training step ≈ 3x forward FLOPs.
    """
    fwd_per_step = estimate_forward_flops(model, batch_size, task_depth, kind)
    return int(steps * 3 * fwd_per_step)

def estimate_inference_flops(model: nn.Module, batch_size: int, task_depth: int = 3, kind: str = "mlp") -> int:
    """Estimate total forward inference FLOPs for an evaluation sequence of length task_depth."""
    return estimate_forward_flops(model, batch_size, task_depth, kind)
