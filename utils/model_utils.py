"""
Model utility functions.
"""

import torch
import torch.nn as nn
from typing import Dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model with detailed breakdown.

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Breakdown by component
    breakdown = {}

    if hasattr(model, 'token_embedding'):
        breakdown['token_embedding'] = sum(
            p.numel() for p in model.token_embedding.parameters()
        )

    if hasattr(model, 'layers'):
        breakdown['transformer_layers'] = sum(
            p.numel() for p in model.layers.parameters()
        )

        # Count attention vs FFN
        attn_params = 0
        ffn_params = 0
        coord_params = 0

        for layer in model.layers:
            if hasattr(layer, 'attention'):
                attn_params += sum(p.numel() for p in layer.attention.parameters())
            if hasattr(layer, 'feed_forward'):
                ffn_params += sum(p.numel() for p in layer.feed_forward.parameters())

            # Count coordinate modulation
            if hasattr(layer.attention, 'coord_mod_q'):
                coord_params += sum(p.numel() for p in layer.attention.coord_mod_q.parameters())
            if hasattr(layer.attention, 'coord_mod_k'):
                coord_params += sum(p.numel() for p in layer.attention.coord_mod_k.parameters())
            if hasattr(layer.attention, 'coord_mod_v'):
                coord_params += sum(p.numel() for p in layer.attention.coord_mod_v.parameters())
            if hasattr(layer.feed_forward, 'coord_mod_gate'):
                coord_params += sum(p.numel() for p in layer.feed_forward.coord_mod_gate.parameters())

        breakdown['attention'] = attn_params
        breakdown['feed_forward'] = ffn_params
        breakdown['coordinate_modulation'] = coord_params

    if hasattr(model, 'coordinate_predictor'):
        breakdown['coordinate_predictor'] = sum(
            p.numel() for p in model.coordinate_predictor.parameters()
        )

    if hasattr(model, 'output'):
        breakdown['output'] = sum(
            p.numel() for p in model.output.parameters()
        )

    return {
        'total': total,
        'trainable': trainable,
        'breakdown': breakdown
    }


def print_model_summary(model: nn.Module, target_params: int = 10_000_000):
    """
    Print detailed model summary.

    Args:
        model: The model to summarize
        target_params: Target parameter count (default 10M)
    """
    params = count_parameters(model)

    print("="*70)
    print("Geometric Intelligence Model Summary")
    print("="*70)

    print(f"\nTotal Parameters: {params['total']:,}")
    print(f"Trainable Parameters: {params['trainable']:,}")
    print(f"Target: {target_params:,}")
    print(f"Difference: {params['total'] - target_params:,} ({(params['total'] - target_params) / target_params * 100:.2f}%)")

    print("\n" + "-"*70)
    print("Parameter Breakdown:")
    print("-"*70)

    breakdown = params['breakdown']
    for component, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = count / params['total'] * 100
        print(f"{component:30s}: {count:12,} ({percentage:5.2f}%)")

    print("="*70)

    # Memory estimate
    memory_fp32 = params['total'] * 4 / (1024**2)  # MB
    memory_fp16 = params['total'] * 2 / (1024**2)  # MB

    print(f"\nMemory Estimates:")
    print(f"  FP32: {memory_fp32:.2f} MB")
    print(f"  FP16: {memory_fp16:.2f} MB")

    print("="*70)


def verify_parameter_budget(model: nn.Module, target: int = 10_000_000, tolerance: float = 0.05):
    """
    Verify that model is within parameter budget.

    Args:
        model: Model to check
        target: Target parameter count
        tolerance: Acceptable deviation (default 5%)

    Returns:
        True if within budget, False otherwise
    """
    params = count_parameters(model)
    total = params['total']

    deviation = abs(total - target) / target

    if deviation <= tolerance:
        print(f"✓ Model within budget: {total:,} / {target:,} ({deviation*100:.2f}% deviation)")
        return True
    else:
        print(f"✗ Model exceeds budget: {total:,} / {target:,} ({deviation*100:.2f}% deviation)")
        return False


def get_model_memory_usage(model: nn.Module, batch_size: int = 1, seq_len: int = 512) -> Dict[str, float]:
    """
    Estimate model memory usage.

    Returns:
        Dictionary with memory estimates in MB
    """
    params = count_parameters(model)

    # Parameter memory
    param_memory = params['total'] * 2 / (1024**2)  # FP16

    # Activation memory (rough estimate)
    # Depends on batch size, sequence length, hidden dim
    if hasattr(model, 'dim'):
        hidden_dim = model.dim
        num_layers = model.num_layers

        # Attention activations
        attn_memory = batch_size * seq_len * seq_len * num_layers * 2 / (1024**2)

        # Hidden state activations
        hidden_memory = batch_size * seq_len * hidden_dim * num_layers * 2 / (1024**2)

        activation_memory = attn_memory + hidden_memory
    else:
        activation_memory = 0

    # KV cache memory
    if hasattr(model, 'num_kv_heads'):
        kv_cache_memory = (
            batch_size * seq_len * model.num_kv_heads * model.head_dim *
            model.num_layers * 2 * 2 / (1024**2)  # K and V, FP16
        )
    else:
        kv_cache_memory = 0

    # Optimizer state (AdamW: 2x params for momentum + variance)
    optimizer_memory = params['total'] * 4 * 2 / (1024**2)  # FP32

    return {
        'parameters': param_memory,
        'activations': activation_memory,
        'kv_cache': kv_cache_memory,
        'optimizer': optimizer_memory,
        'total_inference': param_memory + activation_memory + kv_cache_memory,
        'total_training': param_memory + activation_memory + optimizer_memory
    }
