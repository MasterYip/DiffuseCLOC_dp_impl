# DiffuseCLoC: Co-Diffusion for Physics-Based Character Control

## Overview

DiffuseCLoC is a physics-based character control system that uses co-diffusion of states and actions within a single transformer-based diffusion model. This approach enables guided action generation conditioned on predicted future states, bridging kinematic motion diffusion and action diffusion models.

## Documentation Structure

This documentation is organized into the following sections:

### 1. [Architecture Overview](./01_architecture_overview.md)
- High-level system architecture
- Core mathematical framework
- State and action representations
- Co-diffusion mechanics

### 2. [File Structure](./02_file_structure.md)
- Package organization
- Module dependencies
- Key components and their roles

### 3. [Transformer Architecture](./03_transformer_architecture.md)
- Co-diffusion transformer design
- Attention mechanism details
- Token embedding strategies
- Model specifications

### 4. [Diffusion Models](./04_diffusion_models.md)
- Base diffusion model (SequentialDiffusionModel)
- Joint diffusion actor (JointDiffusionActor)
- DiffuseCLoC implementation
- DDPM mathematics

### 5. [Dataset and Normalization](./05_dataset_normalization.md)
- Dataset structure (G1_Dataset)
- State normalization to local frame
- Data loading and batching
- Symmetry augmentation

### 6. [Training Pipeline](./06_training_pipeline.md)
- Training loop architecture
- Loss computation
- Optimizer configuration
- EMA model handling

### 7. [Usage Guide](./07_usage_guide.md)
- Training a model
- Inference and evaluation
- Configuration management
- Example workflows

## Quick Start

### Installation

```bash
cd /home/user/CodeSpace/Diffusion/cmp_diffusion_policy/diffusion_diffuse_cloc
pip install -r requirements.txt  # You may need to create this
```

### Training

```bash
python train.py --config-name=joint_diffuse
```

### Key Configuration

The main configuration is in `diffusion_policy/config_files/joint_diffuse.yaml`:

- **Model**: Co-diffusion transformer with 2 layers, 4 heads, 256 embedding dim
- **Horizon**: 20 timesteps (past: 4, future: 16)
- **Denoising Steps**: 20
- **Batch Size**: 128
- **Learning Rate**: 1e-4

## Key Features

1. **Co-Diffusion Architecture**: Jointly diffuses states and actions in a single model
2. **Custom Attention Masks**: Causal attention for actions, non-causal for states
3. **Rolling Inference**: FIFO buffer scheme for efficient online deployment
4. **State Emphasis**: Projection matrix to emphasize global state features
5. **Symmetry Augmentation**: Left-right symmetry for data efficiency
6. **Classifier Guidance**: Supports guided generation for task-specific control

## System Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (tested on A100)
- ~2GB GPU memory for inference
- ~8GB GPU memory for training

## Project Structure

```
diffusion_diffuse_cloc/
├── diffusion_policy/          # Main package
│   ├── backbone/              # Neural network backbones
│   ├── modules/               # Diffusion models and actors
│   ├── dataset/               # Data loading and processing
│   ├── trainer/               # Training infrastructure
│   ├── agent/                 # Agent implementations
│   ├── utils/                 # Utility functions
│   └── config_files/          # Configuration files
├── doc/                       # Documentation (this folder)
└── train.py                   # Training entry point
```

## Citation

If you use this code, please cite the original DiffuseCLoC paper:

```bibtex
@article{diffusecloc2024,
  title={DiffuseCLoC: Co-Diffusion for Physics-Based Character Control},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

[Specify license here]

## Contact

For questions and issues, please [contact information or issue tracker].
