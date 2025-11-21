"""
Training system for Geometric Intelligence model.

Implements three-phase curriculum:
1. Primitive Differentiation (0-20%): Learn basic linguistic dimensions
2. Compositional Blending (20-60%): Learn to combine multiple coordinates
3. Conversational Dynamics (60-100%): Learn multi-turn dialogue
"""

from .trainer import GeometricTrainer
from .curriculum import CurriculumScheduler, TrainingPhase
from .data_loaders import (
    PrimitiveDataLoader,
    CompositionalDataLoader,
    ConversationalDataLoader
)

__all__ = [
    'GeometricTrainer',
    'CurriculumScheduler',
    'TrainingPhase',
    'PrimitiveDataLoader',
    'CompositionalDataLoader',
    'ConversationalDataLoader',
]
