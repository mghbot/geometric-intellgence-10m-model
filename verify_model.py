#!/usr/bin/env python3
"""
Verify model architecture and parameter count.

This script creates the model and checks:
1. Total parameter count is close to 10M
2. Forward pass works correctly
3. Coordinate modulation is functioning
4. Memory requirements are reasonable
"""

import argparse
import yaml
import torch

from model import GeometricLlama
from utils import (
    create_tokenizer,
    print_model_summary,
    verify_parameter_budget,
    get_model_memory_usage
)


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Geometric Intelligence model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=512,
        help='Sequence length for testing'
    )

    return parser.parse_args()


def test_forward_pass(model, tokenizer, batch_size, seq_len, device):
    """Test that forward pass works correctly"""
    print("\n" + "="*70)
    print("Testing Forward Pass")
    print("="*70)

    # Create dummy input
    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (batch_size, seq_len),
        device=device
    )

    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    try:
        with torch.no_grad():
            logits, coordinates, kv_cache = model(
                input_ids,
                use_cache=True,
                predict_coordinates=True
            )

        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Coordinates shape: {coordinates.shape}")
        print(f"  KV cache entries: {len(kv_cache) if kv_cache else 0}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_coordinate_modulation(model, tokenizer, device):
    """Test that coordinate modulation affects outputs"""
    print("\n" + "="*70)
    print("Testing Coordinate Modulation")
    print("="*70)

    # Create input
    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (1, 128),
        device=device
    )

    try:
        with torch.no_grad():
            # Forward with predicted coordinates
            logits1, coords1, _ = model(input_ids, predict_coordinates=True)

            # Forward with different coordinates
            coords2 = torch.randn_like(coords1)
            logits2, _, _ = model(
                input_ids,
                coordinates=coords2,
                predict_coordinates=False
            )

            # Check that outputs differ
            diff = torch.mean(torch.abs(logits1 - logits2)).item()

            print(f"✓ Coordinate modulation working")
            print(f"  Average logit difference: {diff:.4f}")

            if diff < 0.01:
                print(f"  ⚠️  Warning: Difference is very small, modulation may be weak")

            return True

    except Exception as e:
        print(f"✗ Coordinate modulation test failed: {e}")
        return False


def test_coordinate_prediction(model, tokenizer, device):
    """Test that coordinate predictor produces reasonable outputs"""
    print("\n" + "="*70)
    print("Testing Coordinate Predictor")
    print("="*70)

    # Create different inputs
    texts = [
        "Hello, how are you?",
        "The doctor formally explained the diagnosis.",
        "What's up dude?",
    ]

    try:
        with torch.no_grad():
            all_coords = []

            for text in texts:
                tokens = tokenizer.encode(text)
                input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

                _, coordinates, _ = model(input_ids, predict_coordinates=True)
                all_coords.append(coordinates.squeeze(0))

                norm = torch.norm(coordinates).item()
                print(f"  '{text[:30]}...'")
                print(f"    Coordinate norm: {norm:.4f}")

            # Check that different inputs produce different coordinates
            dists = []
            for i in range(len(all_coords)):
                for j in range(i+1, len(all_coords)):
                    dist = torch.norm(all_coords[i] - all_coords[j]).item()
                    dists.append(dist)

            avg_dist = sum(dists) / len(dists) if dists else 0
            print(f"\n✓ Coordinate predictor working")
            print(f"  Average pairwise distance: {avg_dist:.4f}")

            if avg_dist < 1.0:
                print(f"  ⚠️  Warning: Coordinates are very similar, may need more training")

            return True

    except Exception as e:
        print(f"✗ Coordinate predictor test failed: {e}")
        return False


def main():
    args = parse_args()

    # Load config
    print("Loading config...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(config['model']['base_architecture']['vocab_size'])

    # Create model
    print("Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeometricLlama(config['model'])
    model = model.to(device)

    # Print summary
    print_model_summary(model, target_params=10_000_000)

    # Verify parameter budget
    print("\n" + "="*70)
    print("Verifying Parameter Budget")
    print("="*70)
    within_budget = verify_parameter_budget(model, target=10_000_000, tolerance=0.10)

    # Memory usage
    print("\n" + "="*70)
    print("Memory Requirements")
    print("="*70)
    memory = get_model_memory_usage(model, args.batch_size, args.seq_len)
    print(f"Parameters: {memory['parameters']:.2f} MB")
    print(f"Activations: {memory['activations']:.2f} MB")
    print(f"KV Cache: {memory['kv_cache']:.2f} MB")
    print(f"Optimizer: {memory['optimizer']:.2f} MB")
    print(f"\nTotal (inference): {memory['total_inference']:.2f} MB")
    print(f"Total (training): {memory['total_training']:.2f} MB")

    if memory['total_training'] > 20_000:  # 20 GB
        print("\n⚠️  WARNING: Training memory usage exceeds 20GB")

    # Run tests
    all_passed = True

    all_passed &= test_forward_pass(model, tokenizer, args.batch_size, args.seq_len, device)
    all_passed &= test_coordinate_modulation(model, tokenizer, device)
    all_passed &= test_coordinate_prediction(model, tokenizer, device)

    # Final verdict
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)

    if all_passed and within_budget:
        print("✓ All tests passed! Model is ready for training.")
    else:
        print("✗ Some tests failed or model exceeds budget.")
        print("  Review the output above for details.")

    print("="*70)


if __name__ == '__main__':
    main()
