"""
Three-phase curriculum scheduler for geometric intelligence training.

Phase 1 (0-20%): Primitive Differentiation
- Learn to separate basic linguistic dimensions
- Coordinates initialized as one-hot, then relaxed

Phase 2 (20-60%): Compositional Blending
- Learn to activate multiple coordinates simultaneously
- Learn interference patterns between coordinates

Phase 3 (60-100%): Conversational Dynamics
- Learn multi-turn dialogue
- Coordinate drift across conversation turns
"""

import torch
from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass


class TrainingPhase(Enum):
    """Training phase enumeration"""
    PRIMITIVE_DIFFERENTIATION = 1
    COMPOSITIONAL_BLENDING = 2
    CONVERSATIONAL_DYNAMICS = 3


@dataclass
class PhaseConfig:
    """Configuration for a training phase"""
    name: str
    phase: TrainingPhase
    tokens: int  # Total tokens for this phase
    start_step: int
    end_step: int
    learning_rate_base: float
    learning_rate_coords: float
    special_losses: Dict[str, float]  # Phase-specific loss weights


class CurriculumScheduler:
    """
    Manages the three-phase training curriculum.

    Handles phase transitions, learning rate scheduling, and phase-specific
    loss computation.
    """
    def __init__(self, config: dict):
        self.config = config

        # Extract phase configs
        phase1_cfg = config['training']['phase1']
        phase2_cfg = config['training']['phase2']
        phase3_cfg = config['training']['phase3']

        # Calculate steps per phase
        batch_size = phase1_cfg['batch_size']
        seq_len = phase1_cfg['seq_len']
        tokens_per_batch = batch_size * seq_len

        phase1_steps = phase1_cfg['tokens'] // tokens_per_batch
        phase2_steps = phase2_cfg['tokens'] // tokens_per_batch
        phase3_steps = phase3_cfg['tokens'] // tokens_per_batch

        # Create phase configs
        self.phases = [
            PhaseConfig(
                name="primitive_differentiation",
                phase=TrainingPhase.PRIMITIVE_DIFFERENTIATION,
                tokens=phase1_cfg['tokens'],
                start_step=0,
                end_step=phase1_steps,
                learning_rate_base=phase1_cfg['learning_rate_base'],
                learning_rate_coords=phase1_cfg['learning_rate_coords'],
                special_losses={
                    'contrastive_weight': 1.0,  # Maximize distance between contrasts
                    'coordinate_sparsity': 0.1,  # Encourage sparse coordinate usage
                }
            ),
            PhaseConfig(
                name="compositional_blending",
                phase=TrainingPhase.COMPOSITIONAL_BLENDING,
                tokens=phase2_cfg['tokens'],
                start_step=phase1_steps,
                end_step=phase1_steps + phase2_steps,
                learning_rate_base=phase2_cfg['learning_rate_base'],
                learning_rate_coords=phase2_cfg['learning_rate_coords'],
                special_losses={
                    'interference_weight': phase2_cfg['interference_loss_weight'],
                    'composition_smoothness': 0.05,  # Smooth coordinate blending
                }
            ),
            PhaseConfig(
                name="conversational_dynamics",
                phase=TrainingPhase.CONVERSATIONAL_DYNAMICS,
                tokens=phase3_cfg['tokens'],
                start_step=phase1_steps + phase2_steps,
                end_step=phase1_steps + phase2_steps + phase3_steps,
                learning_rate_base=phase3_cfg['learning_rate_base'],
                learning_rate_coords=phase3_cfg['learning_rate_coords'],
                special_losses={
                    'temporal_consistency': phase3_cfg['temporal_consistency_weight'],
                    'dialogue_coherence': 0.1,  # Maintain coordinate coherence across turns
                }
            ),
        ]

        self.total_steps = self.phases[-1].end_step
        self.warmup_steps = config['training']['schedule']['warmup_steps']

        self.current_phase_idx = 0

    def get_current_phase(self, step: int) -> PhaseConfig:
        """Get the current training phase based on step number"""
        for i, phase in enumerate(self.phases):
            if phase.start_step <= step < phase.end_step:
                self.current_phase_idx = i
                return phase
        # Return last phase if beyond end
        return self.phases[-1]

    def get_learning_rates(self, step: int) -> Dict[str, float]:
        """
        Get learning rates for current step with warmup and phase-specific rates.

        Returns:
            Dictionary with 'base' and 'coordinates' learning rates
        """
        phase = self.get_current_phase(step)

        # Warmup schedule
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
        else:
            warmup_factor = 1.0

        # Cosine decay within phase
        phase_progress = (step - phase.start_step) / (phase.end_step - phase.start_step)
        decay_factor = 0.5 * (1.0 + torch.cos(torch.tensor(phase_progress * 3.14159)).item())

        lr_base = phase.learning_rate_base * warmup_factor * decay_factor
        lr_coords = phase.learning_rate_coords * warmup_factor * decay_factor

        return {
            'base': lr_base,
            'coordinates': lr_coords
        }

    def should_use_one_hot_coordinates(self, step: int) -> bool:
        """
        During first 5% of phase 1, use one-hot coordinate initialization.
        Then relax to continuous values.
        """
        if self.current_phase_idx != 0:
            return False

        phase = self.phases[0]
        phase_progress = (step - phase.start_step) / (phase.end_step - phase.start_step)
        return phase_progress < 0.05

    def compute_phase_specific_losses(
        self,
        step: int,
        coordinates: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_coordinates: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phase-specific auxiliary losses.

        Args:
            step: Current training step
            coordinates: Current coordinate vectors
            predictions: Model predictions
            targets: Target tokens
            prev_coordinates: Previous step coordinates (for phase 3)
            **kwargs: Phase-specific arguments

        Returns:
            Dictionary of loss tensors
        """
        phase = self.get_current_phase(step)
        losses = {}

        if phase.phase == TrainingPhase.PRIMITIVE_DIFFERENTIATION:
            # Contrastive loss: maximize distance between different primitives
            if 'contrastive_pairs' in kwargs:
                pairs = kwargs['contrastive_pairs']
                if pairs is not None:
                    coord1 = coordinates[pairs[:, 0]]
                    coord2 = coordinates[pairs[:, 1]]
                    # Maximize distance (minimize negative distance)
                    contrastive_loss = -torch.mean(torch.norm(coord1 - coord2, dim=1))
                    losses['contrastive'] = (
                        phase.special_losses['contrastive_weight'] * contrastive_loss
                    )

            # Coordinate sparsity: encourage using few dimensions
            coord_l1 = torch.mean(torch.abs(coordinates))
            losses['sparsity'] = phase.special_losses['coordinate_sparsity'] * coord_l1

        elif phase.phase == TrainingPhase.COMPOSITIONAL_BLENDING:
            # Interference loss: learn non-linear composition
            if 'component_coordinates' in kwargs:
                components = kwargs['component_coordinates']
                if components is not None:
                    # Expected composition (linear blend)
                    expected = torch.mean(torch.stack(components), dim=0)
                    # Actual composition should differ (non-linear)
                    interference = torch.mean((coordinates - expected) ** 2)
                    losses['interference'] = (
                        phase.special_losses['interference_weight'] * interference
                    )

            # Composition smoothness: prevent erratic coordinate changes
            coord_var = torch.var(coordinates, dim=0).mean()
            losses['smoothness'] = phase.special_losses['composition_smoothness'] * coord_var

        elif phase.phase == TrainingPhase.CONVERSATIONAL_DYNAMICS:
            # Temporal consistency: coordinates should drift smoothly
            if prev_coordinates is not None:
                temporal_loss = torch.mean((coordinates - prev_coordinates) ** 2)
                losses['temporal'] = (
                    phase.special_losses['temporal_consistency'] * temporal_loss
                )

            # Dialogue coherence: similar coordinates for related utterances
            if 'dialogue_segments' in kwargs:
                segments = kwargs['dialogue_segments']
                if segments is not None:
                    # Compute within-dialogue variance (should be low)
                    coherence_loss = torch.mean(torch.var(coordinates[segments], dim=0))
                    losses['coherence'] = (
                        phase.special_losses['dialogue_coherence'] * coherence_loss
                    )

        return losses

    def get_phase_description(self, step: int) -> str:
        """Get human-readable description of current phase"""
        phase = self.get_current_phase(step)
        progress = (step - phase.start_step) / (phase.end_step - phase.start_step)
        return f"{phase.name} ({progress*100:.1f}%)"

    def is_phase_transition(self, step: int) -> bool:
        """Check if this step is a phase transition"""
        for phase in self.phases[:-1]:  # Exclude last phase
            if step == phase.end_step:
                return True
        return False
