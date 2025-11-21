"""
Main training loop for Geometric Intelligence model.

Implements three-phase curriculum training with:
- Automatic phase transitions
- Phase-specific loss computation
- Gradient accumulation
- Mixed precision training
- Checkpoint saving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, List
import os
from tqdm import tqdm
import wandb

from .curriculum import CurriculumScheduler, TrainingPhase
from .data_loaders import (
    PrimitiveDataLoader,
    CompositionalDataLoader,
    ConversationalDataLoader
)


class GeometricTrainer:
    """
    Trainer for Geometric Intelligence model with three-phase curriculum.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: dict,
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Move model to device
        self.model = self.model.to(device)

        # Initialize curriculum scheduler
        self.curriculum = CurriculumScheduler(config)

        # Initialize optimizers (separate for base and coordinates)
        self.optimizer_base = self._create_optimizer('base')
        self.optimizer_coords = self._create_optimizer('coordinates')

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config['system']['mixed_precision'] == 'fp16' else None

        # Training state
        self.step = 0
        self.epoch = 0

        # Metrics tracking
        self.metrics = {
            'loss': 0.0,
            'phase_losses': {},
            'learning_rates': {}
        }

        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=config['logging']['wandb_project'],
                config=config
            )

    def _create_optimizer(self, optimizer_type: str) -> torch.optim.Optimizer:
        """Create optimizer for base or coordinate parameters"""
        config = self.config['training']['optimizer']

        # Separate parameters
        if optimizer_type == 'base':
            params = [
                p for name, p in self.model.named_parameters()
                if 'coordinate' not in name and 'coord_mod' not in name
            ]
        else:  # coordinates
            params = [
                p for name, p in self.model.named_parameters()
                if 'coordinate' in name or 'coord_mod' in name
            ]

        return torch.optim.AdamW(
            params,
            lr=3e-4,  # Will be overridden by curriculum
            betas=(config['beta1'], config['beta2']),
            weight_decay=config['weight_decay']
        )

    def train(self, num_steps: Optional[int] = None):
        """
        Main training loop.

        Args:
            num_steps: Maximum steps to train (None = train all phases)
        """
        if num_steps is None:
            num_steps = self.curriculum.total_steps

        self.model.train()

        # Training loop
        pbar = tqdm(total=num_steps, desc="Training")
        while self.step < num_steps:
            # Get current phase
            phase = self.curriculum.get_current_phase(self.step)

            # Create data loader for current phase
            data_loader = self._create_data_loader(phase.phase)

            # Train one epoch
            for batch in data_loader:
                if self.step >= num_steps:
                    break

                # Train step
                metrics = self.train_step(batch, phase)

                # Update metrics
                self._update_metrics(metrics)

                # Logging
                if self.step % self.config['logging']['log_interval'] == 0:
                    self._log_metrics()

                # Checkpoint saving
                if self.step % self.config['logging']['save_interval'] == 0:
                    self._save_checkpoint()

                # Phase transition
                if self.curriculum.is_phase_transition(self.step):
                    print(f"\n{'='*60}")
                    print(f"Phase Transition at step {self.step}")
                    next_phase = self.curriculum.get_current_phase(self.step + 1)
                    print(f"Entering: {next_phase.name}")
                    print(f"{'='*60}\n")

                self.step += 1
                pbar.update(1)

            self.epoch += 1

        pbar.close()
        print("Training complete!")

    def train_step(self, batch: Dict, phase) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of data (format depends on phase)
            phase: Current training phase config

        Returns:
            Dictionary of metrics
        """
        # Get learning rates from curriculum
        lrs = self.curriculum.get_learning_rates(self.step)
        self._set_learning_rates(lrs)

        # Zero gradients
        self.optimizer_base.zero_grad()
        self.optimizer_coords.zero_grad()

        # Forward pass (with mixed precision if enabled)
        with autocast(enabled=self.scaler is not None):
            loss, phase_losses = self._compute_loss(batch, phase)

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer_base)
            self.scaler.unscale_(self.optimizer_coords)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['optimizer']['grad_clip']
            )
            self.scaler.step(self.optimizer_base)
            self.scaler.step(self.optimizer_coords)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['optimizer']['grad_clip']
            )
            self.optimizer_base.step()
            self.optimizer_coords.step()

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'lr_base': lrs['base'],
            'lr_coords': lrs['coordinates'],
            **{k: v.item() if isinstance(v, torch.Tensor) else v
               for k, v in phase_losses.items()}
        }

        return metrics

    def _compute_loss(
        self,
        batch: Dict,
        phase
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on current phase.

        Returns:
            (total_loss, phase_specific_losses)
        """
        phase_type = phase.phase

        if phase_type == TrainingPhase.PRIMITIVE_DIFFERENTIATION:
            return self._compute_primitive_loss(batch)
        elif phase_type == TrainingPhase.COMPOSITIONAL_BLENDING:
            return self._compute_compositional_loss(batch)
        else:  # CONVERSATIONAL_DYNAMICS
            return self._compute_conversational_loss(batch)

    def _compute_primitive_loss(self, batch: Dict) -> tuple:
        """
        Phase 1: Contrastive loss for minimal pairs.

        Forces the model to use different coordinates for contrasting examples.
        """
        # Move to device
        ids1 = batch['input_ids1'].to(self.device)
        ids2 = batch['input_ids2'].to(self.device)

        # Forward pass for both sentences
        logits1, coords1, _ = self.model(ids1, predict_coordinates=True)
        logits2, coords2, _ = self.model(ids2, predict_coordinates=True)

        # Language modeling loss (predict next token)
        lm_loss1 = F.cross_entropy(
            logits1[:, :-1].reshape(-1, logits1.size(-1)),
            ids1[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        lm_loss2 = F.cross_entropy(
            logits2[:, :-1].reshape(-1, logits2.size(-1)),
            ids2[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Contrastive loss: maximize coordinate distance
        coord_distance = torch.norm(coords1 - coords2, dim=1).mean()
        contrastive_loss = -coord_distance  # Negative because we want to maximize

        # Phase-specific losses
        phase_losses = self.curriculum.compute_phase_specific_losses(
            self.step,
            coords1,
            logits1,
            ids1,
            contrastive_pairs=torch.stack([
                torch.arange(ids1.size(0)),
                torch.arange(ids1.size(0))
            ], dim=1).to(self.device)
        )

        # Total loss
        total_loss = (lm_loss1 + lm_loss2) / 2 + 0.1 * contrastive_loss
        for loss_val in phase_losses.values():
            total_loss = total_loss + loss_val

        phase_losses['lm_loss'] = (lm_loss1 + lm_loss2) / 2
        phase_losses['contrastive'] = contrastive_loss

        return total_loss, phase_losses

    def _compute_compositional_loss(self, batch: Dict) -> tuple:
        """
        Phase 2: Compositional loss for multi-coordinate activation.
        """
        # Move to device
        input_ids = batch['input_ids'].to(self.device)

        # Forward pass
        logits, coordinates, _ = self.model(input_ids, predict_coordinates=True)

        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Phase-specific losses (interference, smoothness)
        phase_losses = self.curriculum.compute_phase_specific_losses(
            self.step,
            coordinates,
            logits,
            input_ids,
            component_coordinates=None  # Would use actual components if available
        )

        total_loss = lm_loss
        for loss_val in phase_losses.values():
            total_loss = total_loss + loss_val

        phase_losses['lm_loss'] = lm_loss

        return total_loss, phase_losses

    def _compute_conversational_loss(self, batch: Dict) -> tuple:
        """
        Phase 3: Conversational loss with temporal consistency.
        """
        # Move to device
        turn_ids = batch['turn_ids'].to(self.device)  # (batch, turns, seq_len)
        batch_size, num_turns, seq_len = turn_ids.shape

        # Process each turn and track coordinate drift
        total_lm_loss = 0
        prev_coords = None
        all_coords = []

        for turn in range(num_turns):
            turn_input = turn_ids[:, turn, :]  # (batch, seq_len)

            # Forward pass
            logits, coordinates, _ = self.model(turn_input, predict_coordinates=True)

            # LM loss
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                turn_input[:, 1:].reshape(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            total_lm_loss += lm_loss

            all_coords.append(coordinates)
            prev_coords = coordinates

        # Average LM loss across turns
        total_lm_loss = total_lm_loss / num_turns

        # Compute phase-specific losses (temporal consistency)
        phase_losses = self.curriculum.compute_phase_specific_losses(
            self.step,
            all_coords[-1],  # Last coordinates
            logits,
            turn_input,
            prev_coordinates=all_coords[0] if len(all_coords) > 1 else None,
            dialogue_segments=torch.arange(batch_size).to(self.device)
        )

        total_loss = total_lm_loss
        for loss_val in phase_losses.values():
            total_loss = total_loss + loss_val

        phase_losses['lm_loss'] = total_lm_loss

        return total_loss, phase_losses

    def _create_data_loader(self, phase: TrainingPhase):
        """Create appropriate data loader for current phase"""
        batch_size = self.config['training']['phase1']['batch_size']
        seq_len = self.config['training']['phase1']['seq_len']

        if phase == TrainingPhase.PRIMITIVE_DIFFERENTIATION:
            return PrimitiveDataLoader.create(
                self.tokenizer, batch_size, seq_len
            )
        elif phase == TrainingPhase.COMPOSITIONAL_BLENDING:
            return CompositionalDataLoader.create(
                self.tokenizer, batch_size, seq_len
            )
        else:  # CONVERSATIONAL_DYNAMICS
            return ConversationalDataLoader.create(
                self.tokenizer, batch_size, seq_len
            )

    def _set_learning_rates(self, lrs: Dict[str, float]):
        """Update optimizer learning rates"""
        for param_group in self.optimizer_base.param_groups:
            param_group['lr'] = lrs['base']
        for param_group in self.optimizer_coords.param_groups:
            param_group['lr'] = lrs['coordinates']

    def _update_metrics(self, metrics: Dict[str, float]):
        """Update running metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
            # Exponential moving average
            self.metrics[key] = 0.9 * self.metrics[key] + 0.1 * value

    def _log_metrics(self):
        """Log metrics to console and wandb"""
        phase_desc = self.curriculum.get_phase_description(self.step)

        print(f"Step {self.step} | {phase_desc} | Loss: {self.metrics['loss']:.4f} | "
              f"LR: {self.metrics.get('lr_base', 0):.2e}")

        if self.use_wandb:
            wandb.log({
                'step': self.step,
                'phase': phase_desc,
                **self.metrics
            })

    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'geometric_llama_step_{self.step}.pt'
        )

        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_base_state_dict': self.optimizer_base.state_dict(),
            'optimizer_coords_state_dict': self.optimizer_coords.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_base.load_state_dict(checkpoint['optimizer_base_state_dict'])
        self.optimizer_coords.load_state_dict(checkpoint['optimizer_coords_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.metrics = checkpoint['metrics']

        print(f"Loaded checkpoint from {checkpoint_path} (step {self.step})")
