# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Verify the Model

Check that the architecture is correct and parameter count is ~10M:

```bash
python verify_model.py --config config/model_config.yaml
```

Expected output:
```
Total Parameters: ~10,000,000
✓ All tests passed! Model is ready for training.
```

## Training

Start training with the three-phase curriculum:

```bash
# Start from scratch
python train.py --config config/model_config.yaml

# Resume from checkpoint
python train.py --config config/model_config.yaml --resume checkpoints/model_step_10000.pt
```

Training will progress through three phases:
1. **Phase 1 (0-20%)**: Primitive Differentiation - Learning basic linguistic dimensions
2. **Phase 2 (20-60%)**: Compositional Blending - Learning multi-coordinate activation
3. **Phase 3 (60-100%)**: Conversational Dynamics - Learning dialogue coherence

## Text Generation

After training, generate text:

```bash
# Single prompt generation
python generate.py --checkpoint checkpoints/model.pt --prompt "Hello, how are you?"

# Interactive chat
python generate.py --checkpoint checkpoints/model.pt --interactive

# Analyze coordinate activation
python generate.py --checkpoint checkpoints/model.pt \
    --prompt "The doctor formally explained the diagnosis" \
    --analyze-coordinates
```

## Key Scripts

- `verify_model.py`: Verify architecture and parameter count
- `train.py`: Three-phase training with curriculum
- `generate.py`: Text generation and interactive chat

## Architecture Overview

This model uses **coordinate-modulated weights**:
```
W_effective(θ) = W_base + Σ [cos(θ·c_i) × U_i V_i^T + sin(θ·c_i) × P_i Q_i^T]
```

Different coordinate vectors θ produce different behaviors from the same base weights, enabling:
- Zero catastrophic forgetting
- Compositional generalization
- Emergent linguistic structure
- Efficient parameter usage (10M total)

See `README.md` for complete documentation.
