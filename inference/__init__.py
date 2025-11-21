"""
Inference engine for Geometric Intelligence model.

Handles:
- Coordinate prediction and refinement
- Text generation with coordinate-guided decoding
- Multi-coordinate blending
- Coordinate context tracking across generation
"""

from .generator import GeometricGenerator, GenerationConfig

__all__ = ['GeometricGenerator', 'GenerationConfig']
