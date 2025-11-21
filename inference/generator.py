"""
Text generation with geometric coordinate modulation.

Implements:
1. Amortized coordinate proposal
2. Optional gradient-based refinement
3. Coordinate context tracking (EMA across tokens)
4. Dynamic coordinate drift during generation
5. Adaptive decoding based on coordinate uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    use_coordinate_refinement: bool = True
    refinement_steps: int = 2
    coordinate_ema_alpha: float = 0.95
    coordinate_uncertainty_threshold: float = 0.5
    max_coordinate_norm: float = 50.0
    do_sample: bool = True


class GeometricGenerator:
    """
    Text generator with coordinate modulation.

    This implements the two-stage coordinate selection:
    1. Amortized proposal: Fast prediction from input
    2. Gradient refinement: Optional fine-tuning (adds 2-3ms)

    During generation, maintains coordinate context via EMA.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Tuple[str, List[torch.Tensor]]:
        """
        Generate text from prompt with coordinate modulation.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            (generated_text, coordinate_history)
        """
        if config is None:
            config = GenerationConfig()

        # Tokenize prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # (1, seq_len)

        # Reset coordinate context tracker
        self.model.coordinate_tracker.reset()

        # Initialize KV cache
        kv_cache = None

        # Track coordinates across generation
        coordinate_history = []

        # Generate tokens
        generated_ids = input_ids.clone()

        for step in range(config.max_length):
            # Get current input (only last token if using cache)
            if kv_cache is not None:
                current_input = generated_ids[:, -1:]
            else:
                current_input = generated_ids

            # Forward pass with coordinate prediction
            logits, coordinates, kv_cache = self.model(
                current_input,
                coordinates=None,  # Let model predict
                kv_cache=kv_cache,
                use_cache=True,
                predict_coordinates=True
            )

            # Refine coordinates if enabled
            if config.use_coordinate_refinement:
                coordinates = self._refine_coordinates(
                    coordinates,
                    logits,
                    current_input,
                    config
                )

            # Constrain coordinate norm (length control)
            coord_norm = torch.norm(coordinates, dim=-1, keepdim=True)
            if coord_norm > config.max_coordinate_norm:
                coordinates = coordinates * (config.max_coordinate_norm / coord_norm)

            # Update coordinate context with EMA
            self.model.coordinate_tracker.update(coordinates.squeeze(0))
            coordinate_history.append(coordinates.clone())

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply coordinate-based decoding strategy
            next_token = self._sample_next_token(
                next_token_logits,
                coordinates,
                config
            )

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )

        return generated_text, coordinate_history

    def _refine_coordinates(
        self,
        coordinates: torch.Tensor,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Perform gradient-based coordinate refinement.

        This fine-tunes the amortized coordinates for the specific context.
        """
        # Make coordinates trainable
        coords_refined = coordinates.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([coords_refined], lr=0.01)

        for _ in range(config.refinement_steps):
            optimizer.zero_grad()

            # Forward pass with refined coordinates
            logits_refined, _, _ = self.model(
                input_ids,
                coordinates=coords_refined,
                use_cache=False,
                predict_coordinates=False
            )

            # Compute loss: negative log likelihood + regularization
            # We want coordinates that improve prediction
            loss = F.cross_entropy(
                logits_refined[:, -1, :],
                input_ids[:, -1]
            )

            # Regularization: stay close to amortized proposal
            reg = 0.1 * torch.sum((coords_refined - coordinates) ** 2)
            total_loss = loss + reg

            total_loss.backward()
            optimizer.step()

        return coords_refined.detach()

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        coordinates: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Sample next token using adaptive decoding strategy.

        High coordinate uncertainty → nucleus sampling (exploration)
        Low coordinate uncertainty → greedy/low-temp (precision)
        """
        # Compute coordinate uncertainty (variance of recent coordinates)
        context = self.model.coordinate_tracker.get_context()
        uncertainty = torch.var(coordinates - context).item()

        # Adjust sampling based on uncertainty
        if uncertainty > config.coordinate_uncertainty_threshold:
            # High uncertainty: use nucleus sampling
            return self._nucleus_sampling(logits, config.top_p, config.temperature)
        else:
            # Low uncertainty: use more deterministic sampling
            if config.do_sample:
                return self._top_k_sampling(logits, config.top_k, config.temperature * 0.5)
            else:
                return torch.argmax(logits, dim=-1)

    def _nucleus_sampling(
        self,
        logits: torch.Tensor,
        top_p: float,
        temperature: float
    ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling.

        Sample from smallest set of tokens whose cumulative probability exceeds top_p.
        """
        # Apply temperature
        logits = logits / temperature

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Mask out removed tokens
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Sample from remaining tokens
        probs = F.softmax(sorted_logits, dim=-1)
        next_token_sorted = torch.multinomial(probs, num_samples=1)

        # Map back to original indices
        next_token = sorted_indices.gather(-1, next_token_sorted)

        return next_token.squeeze(-1)

    def _top_k_sampling(
        self,
        logits: torch.Tensor,
        top_k: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Top-k sampling.

        Sample from top k most likely tokens.
        """
        # Apply temperature
        logits = logits / temperature

        # Get top k logits
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Sample from top k
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_topk = torch.multinomial(probs, num_samples=1)

        # Map back to original indices
        next_token = top_k_indices.gather(-1, next_token_topk)

        return next_token.squeeze(-1)

    def chat(
        self,
        messages: List[str],
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Multi-turn chat with coordinate context tracking.

        Args:
            messages: List of conversation turns
            config: Generation configuration

        Returns:
            Generated response
        """
        # Format conversation as single prompt
        prompt = self._format_chat_prompt(messages)

        # Generate response
        response, _ = self.generate(prompt, config)

        return response

    def _format_chat_prompt(self, messages: List[str]) -> str:
        """
        Format multi-turn conversation as prompt.

        Simple format for now; can be customized for specific chat templates.
        """
        formatted = ""
        for i, message in enumerate(messages):
            if i % 2 == 0:
                formatted += f"User: {message}\n"
            else:
                formatted += f"Assistant: {message}\n"

        formatted += "Assistant: "
        return formatted

    def get_coordinate_statistics(
        self,
        prompt: str
    ) -> dict:
        """
        Analyze coordinate activation for a given prompt.

        Useful for understanding which linguistic dimensions are active.

        Returns:
            Dictionary with coordinate statistics
        """
        # Tokenize prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # Get coordinates
        with torch.no_grad():
            _, coordinates, _ = self.model(
                input_ids,
                predict_coordinates=True,
                use_cache=False
            )

        coords = coordinates.squeeze(0).cpu()

        # Compute statistics
        stats = {
            'norm': torch.norm(coords).item(),
            'mean': torch.mean(coords).item(),
            'std': torch.std(coords).item(),
            'max': torch.max(coords).item(),
            'min': torch.min(coords).item(),
            'top_10_dims': torch.topk(torch.abs(coords), 10).indices.tolist(),
            'top_10_values': torch.topk(torch.abs(coords), 10).values.tolist(),
        }

        return stats

    def interpolate_coordinates(
        self,
        prompt1: str,
        prompt2: str,
        steps: int = 5
    ) -> List[str]:
        """
        Interpolate between coordinate spaces of two prompts.

        Useful for exploring the coordinate manifold.

        Args:
            prompt1: First prompt
            prompt2: Second prompt
            steps: Number of interpolation steps

        Returns:
            List of generated texts at each interpolation step
        """
        # Get coordinates for both prompts
        ids1 = torch.tensor(
            self.tokenizer.encode(prompt1, add_special_tokens=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        ids2 = torch.tensor(
            self.tokenizer.encode(prompt2, add_special_tokens=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            _, coords1, _ = self.model(ids1, predict_coordinates=True)
            _, coords2, _ = self.model(ids2, predict_coordinates=True)

        # Interpolate
        results = []
        for i in range(steps):
            alpha = i / (steps - 1)
            coords_interp = (1 - alpha) * coords1 + alpha * coords2

            # Generate with interpolated coordinates
            # Use prompt1 as base
            with torch.no_grad():
                logits, _, _ = self.model(
                    ids1,
                    coordinates=coords_interp,
                    predict_coordinates=False
                )

            # Simple greedy decoding for analysis
            text = self.tokenizer.decode(
                torch.argmax(logits[0], dim=-1).tolist(),
                skip_special_tokens=True
            )

            results.append(text)

        return results
