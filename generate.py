#!/usr/bin/env python3
"""
Text generation script for Geometric Intelligence model.

Usage:
    python generate.py --checkpoint checkpoints/model.pt --prompt "Hello, how are you?"
    python generate.py --checkpoint checkpoints/model.pt --interactive
"""

import argparse
import yaml
import torch

from model import GeometricLlama
from inference import GeometricGenerator, GenerationConfig
from utils import create_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text with Geometric Intelligence model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Input prompt for generation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive chat mode'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.92,
        help='Nucleus sampling top-p'
    )
    parser.add_argument(
        '--no-refinement',
        action='store_true',
        help='Disable coordinate refinement (faster)'
    )
    parser.add_argument(
        '--analyze-coordinates',
        action='store_true',
        help='Show coordinate statistics'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )

    return parser.parse_args()


def interactive_chat(generator, config):
    """Interactive chat loop"""
    print("="*70)
    print("Geometric Intelligence Chat")
    print("="*70)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'reset' to start a new conversation")
    print("Type 'coords' to see coordinate statistics for last input")
    print("="*70 + "\n")

    conversation_history = []

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'reset':
            conversation_history = []
            print("Conversation reset.\n")
            continue

        if user_input.lower() == 'coords':
            if conversation_history:
                last_input = conversation_history[-1]
                stats = generator.get_coordinate_statistics(last_input)
                print("\nCoordinate Statistics:")
                print(f"  Norm: {stats['norm']:.4f}")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std:  {stats['std']:.4f}")
                print(f"  Top 10 dimensions: {stats['top_10_dims']}")
                print(f"  Top 10 values: {[f'{v:.4f}' for v in stats['top_10_values']]}")
                print()
            else:
                print("No conversation history yet.\n")
            continue

        if not user_input:
            continue

        # Add to history
        conversation_history.append(user_input)

        # Generate response
        try:
            response = generator.chat(conversation_history, config)
            print(f"Assistant: {response}\n")

            # Add response to history
            conversation_history.append(response)

        except Exception as e:
            print(f"Error generating response: {e}\n")


def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Load config
    config = checkpoint['config']

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(config['model']['base_architecture']['vocab_size'])

    # Create model
    print("Creating model...")
    model = GeometricLlama(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully (step {checkpoint['step']})")

    # Create generator
    generator = GeometricGenerator(model, tokenizer, args.device)

    # Generation config
    gen_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        use_coordinate_refinement=not args.no_refinement
    )

    # Interactive mode
    if args.interactive:
        interactive_chat(generator, gen_config)
        return

    # Single prompt mode
    if args.prompt is None:
        print("Error: Must provide --prompt or use --interactive")
        return

    print(f"\nPrompt: {args.prompt}\n")

    # Generate
    print("Generating...")
    generated_text, coordinate_history = generator.generate(args.prompt, gen_config)

    print("="*70)
    print("Generated Text:")
    print("="*70)
    print(generated_text)
    print("="*70)

    # Analyze coordinates if requested
    if args.analyze_coordinates:
        stats = generator.get_coordinate_statistics(args.prompt)
        print("\nCoordinate Statistics:")
        print(f"  Norm: {stats['norm']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Top 10 active dimensions: {stats['top_10_dims']}")
        print(f"  Top 10 values: {[f'{v:.4f}' for v in stats['top_10_values']]}")

        print(f"\n  Total generation steps: {len(coordinate_history)}")
        print(f"  Average coordinate norm: {sum(torch.norm(c).item() for c in coordinate_history) / len(coordinate_history):.4f}")


if __name__ == '__main__':
    main()
