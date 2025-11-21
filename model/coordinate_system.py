"""
Coordinate Modulation System for Geometric Intelligence

This module implements the core innovation: modulating neural network weights
through a 1024-dimensional coordinate space using trigonometric functions.

Key formula:
W_effective(θ) = W_base + Σ_{i=1}^{64} α_i(x) × [cos(θ·c_i) × U_i V_i^T + sin(θ·c_i) × P_i Q_i^T]

where:
- θ is the coordinate vector in 1024D space
- α_i(x) is learned gating based on input
- cos/sin create spectral modulation patterns
- U_i, V_i, P_i, Q_i are low-rank modulation matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpectralWavelet(nn.Module):
    """
    A single cos/sin wavelet pair that modulates weight matrices.

    This implements one term in the coordinate modulation sum:
    α(x) × [cos(θ·c) × UV^T + sin(θ·c) × PQ^T]
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int,
        rank: int = 12
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coordinate_dim = coordinate_dim
        self.rank = rank

        # Coordinate direction vector (what to project θ onto)
        self.coord_direction = nn.Parameter(torch.randn(coordinate_dim) / math.sqrt(coordinate_dim))

        # Low-rank factorization for cos modulation: U × V^T
        # Shape: (output_dim, rank) × (rank, input_dim) = (output_dim, input_dim)
        self.U_cos = nn.Parameter(torch.randn(output_dim, rank) / math.sqrt(rank))
        self.V_cos = nn.Parameter(torch.randn(input_dim, rank) / math.sqrt(rank))

        # Low-rank factorization for sin modulation: P × Q^T
        self.U_sin = nn.Parameter(torch.randn(output_dim, rank) / math.sqrt(rank))
        self.V_sin = nn.Parameter(torch.randn(input_dim, rank) / math.sqrt(rank))

        # Input-dependent gating function α(x)
        # Projects input embeddings to scalar gate value
        self.gate_proj = nn.Linear(input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        W_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply spectral modulation to base weights.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            coordinates: Coordinate vector θ (batch, coordinate_dim) or (coordinate_dim,)
            W_base: Base weight matrix (output_dim, input_dim)
        Returns:
            Modulated output (batch, seq_len, output_dim)
        """
        # Compute coordinate projection: θ·c
        if coordinates.dim() == 1:
            coord_proj = torch.dot(coordinates, self.coord_direction)
        else:
            coord_proj = torch.matmul(coordinates, self.coord_direction)  # (batch,)

        # Compute trigonometric modulation values
        cos_val = torch.cos(coord_proj)
        sin_val = torch.sin(coord_proj)

        # Compute low-rank modulation matrices
        # UV^T: (output_dim, rank) @ (rank, input_dim) = (output_dim, input_dim)
        W_cos = torch.matmul(self.U_cos, self.V_cos.t())
        W_sin = torch.matmul(self.U_sin, self.V_sin.t())

        # Apply coordinate modulation to weights
        if cos_val.dim() == 0:  # Scalar case
            W_modulated = cos_val * W_cos + sin_val * W_sin
        else:  # Batch case
            W_modulated = (
                cos_val.view(-1, 1, 1) * W_cos.unsqueeze(0) +
                sin_val.view(-1, 1, 1) * W_sin.unsqueeze(0)
            )

        # Compute input-dependent gating α(x)
        # Average over sequence length for stability
        gate = torch.sigmoid(self.gate_proj(x).mean(dim=1))  # (batch, 1)

        # Apply gated modulation
        if W_modulated.dim() == 2:  # No batch dim
            return F.linear(x, W_base + gate * W_modulated)
        else:  # Batch dim present
            # For batched modulation, we need to handle each batch element
            # This is more expensive but allows per-example modulation
            output = F.linear(x, W_base)
            for b in range(x.shape[0]):
                output[b] = output[b] + gate[b] * F.linear(x[b], W_modulated[b])
            return output


class CoordinateModulation(nn.Module):
    """
    Complete coordinate modulation system with 64 wavelet pairs.

    This modulates specified weight matrices (Q, K, V, FFN_gate) in each layer
    using the coordinate vector θ.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int = 1024,
        num_wavelets: int = 64,
        rank: int = 12
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coordinate_dim = coordinate_dim
        self.num_wavelets = num_wavelets
        self.rank = rank

        # Create wavelet pairs
        self.wavelets = nn.ModuleList([
            SpectralWavelet(input_dim, output_dim, coordinate_dim, rank)
            for _ in range(num_wavelets)
        ])

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        W_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply coordinate modulation using all wavelets.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            coordinates: Coordinate vector θ
            W_base: Base weight matrix (output_dim, input_dim)
        Returns:
            Modulated output (batch, seq_len, output_dim)
        """
        # Start with base weight application
        output = F.linear(x, W_base)

        # Add contribution from each wavelet (can be parallelized)
        for wavelet in self.wavelets:
            # Each wavelet computes its own modulation
            # We accumulate the differences from base
            wavelet_out = wavelet(x, coordinates, torch.zeros_like(W_base))
            output = output + wavelet_out

        return output


class CoordinateEnergyPredictor(nn.Module):
    """
    Amortized coordinate predictor that maps inputs to optimal coordinates.

    This network learns to predict the best coordinate vector θ for a given
    input in a single forward pass, replacing expensive iterative optimization.

    Architecture: 2-layer MLP with residual connections (256K params total)
    """
    def __init__(
        self,
        input_dim: int = 448,
        coordinate_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.coordinate_dim = coordinate_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with residual connections
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection to coordinate space
        self.output_proj = nn.Linear(hidden_dim, coordinate_dim)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal coordinates for input.

        Args:
            x: Input embeddings (batch, seq_len, input_dim)
        Returns:
            Predicted coordinates θ (batch, coordinate_dim)
        """
        # Average pool over sequence dimension to get fixed-size representation
        # This allows variable-length inputs
        x_pooled = x.mean(dim=1)  # (batch, input_dim)

        # Project to hidden dimension
        h = F.gelu(self.input_proj(x_pooled))

        # Apply layers with residual connections
        for layer in self.layers:
            h_new = F.gelu(layer(self.norm(h)))
            h = h + h_new  # Residual connection

        # Project to coordinate space
        coordinates = self.output_proj(h)

        # L2 normalize to put on unit sphere (helps with training stability)
        # Then scale to reasonable range
        coordinates = F.normalize(coordinates, p=2, dim=-1) * math.sqrt(self.coordinate_dim)

        return coordinates


class CoordinateRefinement(nn.Module):
    """
    Optional gradient-based refinement of coordinates for higher quality.

    After amortized prediction, we can optionally perform 1-2 gradient steps
    to fine-tune coordinates based on the actual generation context.
    """
    def __init__(self, learning_rate: float = 0.01, num_steps: int = 2):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_steps = num_steps

    def refine(
        self,
        coordinates: torch.Tensor,
        loss_fn: callable,
        proposed_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine coordinates via gradient descent.

        Args:
            coordinates: Initial coordinates θ (requires_grad=True)
            loss_fn: Function that computes loss for given coordinates
            proposed_coords: Amortized proposal (for regularization)
        Returns:
            Refined coordinates
        """
        coords = coordinates.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([coords], lr=self.learning_rate)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Compute loss: negative log likelihood + L2 regularization
            nll = loss_fn(coords)
            reg = 0.1 * torch.sum((coords - proposed_coords) ** 2)
            loss = nll + reg

            loss.backward()
            optimizer.step()

        return coords.detach()


class CoordinateContextTracker(nn.Module):
    """
    Tracks coordinate context across generation steps using exponential moving average.

    Implements geometric attention: θ_context[t] = α × θ_context[t-1] + (1-α) × θ_t
    """
    def __init__(self, coordinate_dim: int = 1024, alpha: float = 0.95):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.alpha = alpha
        self.register_buffer('context', torch.zeros(coordinate_dim))

    def update(self, new_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Update context with new coordinates.

        Args:
            new_coordinates: New coordinate vector (coordinate_dim,) or (batch, coordinate_dim)
        Returns:
            Updated context vector
        """
        if new_coordinates.dim() == 1:
            coords = new_coordinates
        else:
            # If batched, average across batch
            coords = new_coordinates.mean(dim=0)

        # Exponential moving average
        self.context = self.alpha * self.context + (1 - self.alpha) * coords
        return self.context

    def reset(self):
        """Reset context to zero (call at start of new generation)"""
        self.context.zero_()

    def get_context(self) -> torch.Tensor:
        """Get current context vector"""
        return self.context
