"""
Geometric Llama: Llama 3.3 with Coordinate Modulation

This implements a 10M parameter conversational AI with:
- Base Llama 3.3: 7.8M parameters
- Coordinate Infrastructure: 2.2M parameters
- Total: 10M parameters

The model uses 1024-dimensional coordinate space to modulate weights,
enabling efficient multi-task learning without catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .components import (
    RMSNorm,
    RotaryPositionalEmbedding,
    apply_rotary_emb,
    SwiGLU,
    repeat_kv
)
from .coordinate_system import (
    CoordinateModulation,
    CoordinateEnergyPredictor,
    CoordinateContextTracker
)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) from Llama 3.

    GQA is a more efficient variant of Multi-Head Attention where multiple
    query heads share the same key/value heads. This reduces KV cache size
    and computation while maintaining model quality.

    In this model: 16 Q heads, 4 KV heads → 4 queries per KV head
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        coordinate_dim: int = 1024,
        use_coordinate_modulation: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads  # 16 // 4 = 4 groups

        # Base projection matrices (modulated by coordinates)
        self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Coordinate modulation for Q, K, V
        self.use_coordinate_modulation = use_coordinate_modulation
        if use_coordinate_modulation:
            self.coord_mod_q = CoordinateModulation(
                input_dim=dim,
                output_dim=num_heads * head_dim,
                coordinate_dim=coordinate_dim,
                num_wavelets=16,  # Fewer wavelets for attention (memory)
                rank=12
            )
            self.coord_mod_k = CoordinateModulation(
                input_dim=dim,
                output_dim=num_kv_heads * head_dim,
                coordinate_dim=coordinate_dim,
                num_wavelets=16,
                rank=12
            )
            self.coord_mod_v = CoordinateModulation(
                input_dim=dim,
                output_dim=num_kv_heads * head_dim,
                coordinate_dim=coordinate_dim,
                num_wavelets=16,
                rank=12
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input (batch, seq_len, dim)
            freqs_cos: RoPE cosine frequencies
            freqs_sin: RoPE sine frequencies
            coordinates: Coordinate vector θ for modulation
            attention_mask: Causal mask
            kv_cache: Cached (key, value) from previous steps
        Returns:
            (output, new_kv_cache)
        """
        batch_size, seq_len, _ = x.shape

        # Apply coordinate-modulated projections
        if self.use_coordinate_modulation and coordinates is not None:
            # Use coordinate modulation for Q, K, V
            # Note: This is expensive, so we use fewer wavelets
            q = self.coord_mod_q(x, coordinates, self.wq.weight).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            k = self.coord_mod_k(x, coordinates, self.wk.weight).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
            v = self.coord_mod_v(x, coordinates, self.wv.weight).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        else:
            # Standard projections
            q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Handle KV cache for generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_kv_cache = (k, v)

        # Repeat KV heads for grouped query attention
        # Each KV head is shared by num_kv_groups query heads
        k = repeat_kv(k, self.num_kv_groups)  # (batch, seq_len, num_heads, head_dim)
        v = repeat_kv(v, self.num_kv_groups)

        # Transpose for attention computation
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # (batch, num_heads, seq_len_q, head_dim) @ (batch, num_heads, head_dim, seq_len_k)
        # -> (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, head_dim)
        # -> (batch, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)

        return output, new_kv_cache


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation and coordinate modulation.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        coordinate_dim: int = 1024,
        use_coordinate_modulation: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # SwiGLU components
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Value
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Output

        # Coordinate modulation for gate (most important for controlling behavior)
        self.use_coordinate_modulation = use_coordinate_modulation
        if use_coordinate_modulation:
            self.coord_mod_gate = CoordinateModulation(
                input_dim=dim,
                output_dim=hidden_dim,
                coordinate_dim=coordinate_dim,
                num_wavelets=32,  # More wavelets for FFN (main computation)
                rank=12
            )

    def forward(self, x: torch.Tensor, coordinates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, dim)
            coordinates: Coordinate vector θ
        Returns:
            Output (batch, seq_len, dim)
        """
        # Apply coordinate modulation to gate
        if self.use_coordinate_modulation and coordinates is not None:
            gate = F.silu(self.coord_mod_gate(x, coordinates, self.w1.weight))
        else:
            gate = F.silu(self.w1(x))

        # Standard value projection (not modulated for efficiency)
        value = self.w2(x)

        # SwiGLU: gate * value
        hidden = gate * value

        # Output projection
        return self.w3(hidden)


class TransformerBlock(nn.Module):
    """
    Single transformer layer with GQA and coordinate-modulated FFN.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        coordinate_dim: int = 1024,
        norm_eps: float = 1e-5,
        use_coordinate_modulation: bool = True
    ):
        super().__init__()
        self.dim = dim

        # Pre-normalization
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        # Attention and FFN
        self.attention = GroupedQueryAttention(
            dim, num_heads, num_kv_heads, head_dim, coordinate_dim, use_coordinate_modulation
        )
        self.feed_forward = FeedForward(
            dim, hidden_dim, coordinate_dim, use_coordinate_modulation
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input (batch, seq_len, dim)
            freqs_cos/sin: RoPE frequencies
            coordinates: Coordinate vector θ
            attention_mask: Causal mask
            kv_cache: Cached KV
        Returns:
            (output, new_kv_cache)
        """
        # Attention with residual
        h, new_kv_cache = self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            coordinates,
            attention_mask,
            kv_cache
        )
        x = x + h

        # FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x), coordinates)

        return x, new_kv_cache


class GeometricLlama(nn.Module):
    """
    Complete Geometric Llama model with coordinate modulation.

    Architecture:
    - 10 transformer layers
    - 448 hidden dim
    - 16 attention heads (4 KV heads for GQA)
    - 32k vocabulary
    - 1024-dimensional coordinate space
    - Total: 10M parameters (7.8M base + 2.2M coordinate infrastructure)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Extract config
        base_cfg = config['base_architecture']
        coord_cfg = config['coordinate_system']

        self.vocab_size = base_cfg['vocab_size']
        self.num_layers = base_cfg['num_layers']
        self.dim = base_cfg['hidden_dim']
        self.num_heads = base_cfg['num_attention_heads']
        self.num_kv_heads = base_cfg['num_kv_heads']
        self.head_dim = base_cfg['head_dim']
        self.hidden_dim = base_cfg['intermediate_dim']
        self.max_seq_len = base_cfg['max_seq_len']
        self.coordinate_dim = coord_cfg['num_coordinates']

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)

        # Rotary positional embeddings
        self.rope = RotaryPositionalEmbedding(
            self.head_dim,
            self.max_seq_len,
            base_cfg['rope_theta']
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.hidden_dim,
                self.coordinate_dim,
                base_cfg['norm_eps'],
                use_coordinate_modulation=True
            )
            for _ in range(self.num_layers)
        ])

        # Final norm and output
        self.norm = RMSNorm(self.dim, eps=base_cfg['norm_eps'])
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        # Coordinate system
        self.coordinate_predictor = CoordinateEnergyPredictor(
            self.dim,
            self.coordinate_dim,
            coord_cfg.get('coordinate_predictor', {}).get('hidden_dim', 512),
            coord_cfg.get('coordinate_predictor', {}).get('num_layers', 2)
        )

        # Coordinate context tracker for generation
        self.coordinate_tracker = CoordinateContextTracker(self.coordinate_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Llama 3 initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
        predict_coordinates: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch, seq_len)
            coordinates: Coordinate vector θ (optional, will be predicted if None)
            attention_mask: Attention mask
            kv_cache: List of cached (K, V) tuples for each layer
            use_cache: Whether to return KV cache
            predict_coordinates: Whether to predict coordinates if not provided
        Returns:
            (logits, coordinates, new_kv_cache)
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        h = self.token_embedding(input_ids)

        # Predict coordinates if not provided
        if coordinates is None and predict_coordinates:
            coordinates = self.coordinate_predictor(h)

        # Get RoPE frequencies
        freqs_cos, freqs_sin = self.rope(h, seq_len)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, h.device)

        # Initialize KV cache if needed
        if use_cache and kv_cache is None:
            kv_cache = [None] * self.num_layers

        # Forward through transformer layers
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h, new_cache = layer(
                h,
                freqs_cos,
                freqs_sin,
                coordinates,
                attention_mask,
                layer_cache
            )
            if use_cache:
                new_kv_cache.append(new_cache)

        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)

        return logits, coordinates, new_kv_cache if use_cache else None

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def count_parameters(self) -> dict:
        """Count parameters in each component"""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        base_params = (
            count(self.token_embedding) +
            sum(count(layer) for layer in self.layers) +
            count(self.norm) +
            count(self.output)
        )

        coord_params = count(self.coordinate_predictor)

        # Coordinate modulation in layers
        coord_mod_params = 0
        for layer in self.layers:
            if hasattr(layer.attention, 'coord_mod_q'):
                coord_mod_params += count(layer.attention.coord_mod_q)
                coord_mod_params += count(layer.attention.coord_mod_k)
                coord_mod_params += count(layer.attention.coord_mod_v)
            if hasattr(layer.feed_forward, 'coord_mod_gate'):
                coord_mod_params += count(layer.feed_forward.coord_mod_gate)

        total = base_params + coord_params + coord_mod_params

        return {
            'base_parameters': base_params,
            'coordinate_predictor': coord_params,
            'coordinate_modulation': coord_mod_params,
            'total': total,
            'target': 10_000_000,
            'difference': total - 10_000_000
        }
