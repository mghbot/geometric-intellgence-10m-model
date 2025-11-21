#!/usr/bin/env python3
"""
Training script for Geometric Intelligence 10M parameter model.

Usage:
    python train.py --config config/model_config.yaml
    python train.py --config config/model_config.yaml --resume checkpoints/step_10000.pt
"""

import argparse
import yaml
import torch
import sys

from model import GeometricLlama
from training import GeometricTrainer
from utils import create_tokenizer, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train Geometric Intelligence model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of training steps (overrides config)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(config['model']['base_architecture']['vocab_size'])

    # Create model
    print("Creating model...")
    model = GeometricLlama(config['model'])

    # Print model summary
    print_model_summary(model, target_params=10_000_000)

    # Check if within budget
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > 11_000_000:  # 10% over budget
        print(f"\n⚠️  WARNING: Model has {total_params:,} parameters (target: 10M)")
        print("Consider reducing hidden dimensions or number of wavelets")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = GeometricTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
        use_wandb=not args.no_wandb
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    try:
        trainer.train(num_steps=args.steps)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer._save_checkpoint()
        print("Checkpoint saved successfully")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
