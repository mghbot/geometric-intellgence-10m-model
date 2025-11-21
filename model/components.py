"""
Core components for Llama 3.3 architecture:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Positional Embeddings
- SwiGLU: Swish-Gated Linear Unit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in Llama models.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from Su et al. 2021.
    Encodes position information through rotation matrices.
    More effective than absolute positional embeddings for long sequences.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        # theta_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for max sequence length
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency"""
        # Create position indices
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # Compute freqs: outer product of positions and frequencies
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate for applying to even/odd dimensions
        emb = torch.cat([freqs, freqs], dim=-1)
        # Cache cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for given sequence length.

        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length (if None, uses x.shape[1])
        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Reshape for broadcasting: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Split into first and second half for rotation
    # RoPE applies 2D rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    Used in Llama FFN layers. More effective than standard ReLU/GELU.

    Formula: SwiGLU(x) = Swish(xW) ⊙ (xV)
             where Swish(x) = x * sigmoid(x)
             and ⊙ is element-wise multiplication
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        # Two linear projections: one for gate, one for values
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # Values
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # Output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Output tensor of same shape
        """
        # Apply SwiGLU gating
        gate = F.silu(self.w1(x))  # Swish activation (SiLU in PyTorch)
        values = self.w2(x)
        hidden = gate * values
        # Project back to input dimension
        return self.w3(hidden)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention (GQA).

    In GQA, we have fewer KV heads than Q heads. This function repeats
    each KV head n_rep times to match the number of Q heads.

    Args:
        x: Tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        n_rep: Number of times to repeat each KV head
    Returns:
        Tensor of shape (batch, seq_len, num_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x

    batch, seq_len, num_kv_heads, head_dim = x.shape

    # Expand and reshape to repeat each head n_rep times
    x = x[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)
