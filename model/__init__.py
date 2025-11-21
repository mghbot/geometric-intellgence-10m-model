"""
Geometric Intelligence 10M Parameter Model

This module implements a novel neural architecture where weights are modulated
by a 1024-dimensional coordinate space using trigonometric functions.
"""

from .geometric_llama import GeometricLlama
from .coordinate_system import CoordinateModulation, CoordinateEnergyPredictor
from .components import RMSNorm, RotaryPositionalEmbedding, SwiGLU

__all__ = [
    'GeometricLlama',
    'CoordinateModulation',
    'CoordinateEnergyPredictor',
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'SwiGLU',
]
