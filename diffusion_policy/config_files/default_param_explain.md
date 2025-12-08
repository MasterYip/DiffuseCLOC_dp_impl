# DiffuseCLoC Default Parameter Configuration

This document provides a detailed explanation of all configuration parameters in `default_param.yaml`. This configuration file defines the complete setup for training DiffuseCLoC models using Hydra's structured configuration system.

## Table of Contents

1. [Overview](#overview)
2. [Trainer Configuration](#trainer-configuration)
3. [Data Collection](#data-collection)
4. [Data Loading](#data-loading)
5. [EMA Model](#ema-model)
6. [Experiment Settings](#experiment-settings)
7. [Optimizer Configuration](#optimizer-configuration)
8. [Policy Architecture](#policy-architecture)
9. [Dataset Configuration](#dataset-configuration)
10. [Training Parameters](#training-parameters)
11. [Checkpoint Settings](#checkpoint-settings)

## Overview

The configuration uses Hydra's `_target_` system to instantiate Python objects from YAML. The root target is `diffusion_policy.trainer.offline_trainer.OfflineTrainer`, which orchestrates the entire training process.

## Trainer Configuration

```yaml
_target_: diffusion_policy.trainer.offline_trainer.OfflineTrainer
```

**Purpose**: Specifies the main trainer class responsible for coordinating training, validation, and checkpointing.

**Class**: `OfflineTrainer` handles offline (pre-collected) dataset training with full epoch-based loops.

## Data Collection [Not Accessible]

```yaml
data_collection:
  name: null                          # Collection run name (optional)
  record_steps: 1000000              # Maximum steps to record
  
  # Add noise to states while keeping actions clean
  noisy_state_clean_action:
    enable: true                     # Enable state noise injection
    std_scale: 1.0                   # Standard deviation scale for noise
  
  # Apply random perturbations to robot bodies
  perturbation_bodies:
    enable: false                    # Disable body perturbations
    force_range: 0.0                 # Force perturbation magnitude
    torque_range: 0.0                # Torque perturbation magnitude
  
  perfect_expert_only: true          # Use only high-quality demonstrations
```

**Purpose**: Controls data collection and augmentation strategies.

**Key Features**:
- **Noisy State Clean Action**: Adds noise to observations during training while keeping ground-truth actions clean, improving robustness
- **Body Perturbations**: Can apply random forces/torques during data collection (currently disabled)
- **Perfect Expert Only**: Filters to use only high-quality demonstration data

## Data Loading

### Training DataLoader

```yaml
dataloader:
  batch_size: 128                    # Training batch size
  num_workers: 1                     # Number of parallel data loading processes
  persistent_workers: true           # Keep workers alive between epochs
  pin_memory: true                   # Pin tensors to GPU memory for faster transfer
  shuffle: true                      # Shuffle training data
```

### Validation DataLoader

```yaml
val_dataloader:
  batch_size: 2048                   # Larger batch size for validation (no gradients)
  num_workers: 1                     # Number of workers for validation
  persistent_workers: false          # Don't keep validation workers persistent
  pin_memory: true                   # Pin memory for GPU transfer
  shuffle: false                     # Don't shuffle validation data
```

**Rationale**:
- **Training batch size (128)**: Balances GPU memory usage with gradient stability
- **Validation batch size (2048)**: Larger batches are efficient since no gradients are computed
- **Persistent workers**: Avoids worker recreation overhead during training

## EMA Model

```yaml
ema:
  _target_: diffusion_policy.backbone.ema_model.EMAModel
  inv_gamma: 1.0                     # Inverse gamma parameter
  max_value: 0.9999                  # Maximum EMA decay rate
  min_value: 0.0                     # Minimum EMA decay rate
  power: 0.75                        # Power for decay schedule
  update_after_step: 0               # Start EMA updates after this step
```

**Purpose**: Exponential Moving Average model for stable inference.

**EMA Decay Formula**:
```
decay = min_value + (max_value - min_value) * (1 + step / inv_gamma) ** (-power)
```

**Benefits**:
- **Stable inference**: Smoothed model weights reduce inference variance
- **Better generalization**: EMA models often perform better than final training weights
- **Adaptive decay**: Starts aggressive, becomes more conservative over time

## Experiment Settings

```yaml
exp_name: null                       # Experiment name (auto-generated if null)
output_dir: "./outputs/"             # Base directory for outputs
dataset_dir: "/home/takaraet/Projects/DiffuseCloC/data/"  # Dataset directory

logging:
  group: null                        # Wandb group name
  id: null                           # Wandb run ID (auto-generated if null)
  mode: online                       # Wandb logging mode (online/offline/disabled)
  name: 'velocity_loss_1'            # Wandb run name
  project: diffuse_cloc              # Wandb project name
  resume: true                       # Resume wandb run if exists
  tags:                              # Tags for experiment organization
  - velocity_1
  - default
```

**Purpose**: Manages experiment tracking and output organization.

**Wandb Integration**:
- **Project**: Groups related experiments
- **Tags**: Facilitate experiment filtering and comparison
- **Resume**: Allows continuing interrupted runs

## Optimizer Configuration

```yaml
optimizer:
  betas: [0.9, 0.95]                 # Adam beta parameters
  learning_rate: 0.0001              # Base learning rate (1e-4)
  weight_decay: 0.001                # AdamW weight decay (1e-3)
```

**Purpose**: Defines AdamW optimizer hyperparameters.

**Parameter Details**:
- **betas**: Momentum parameters for gradient and squared gradient
  - `beta1 = 0.9`: Momentum for gradients
  - `beta2 = 0.95`: Momentum for squared gradients (higher than typical 0.999)
- **learning_rate**: Conservative learning rate suitable for transformers
- **weight_decay**: L2 regularization to prevent overfitting

## Policy Architecture

### Agent Configuration

```yaml
policy:
  _target_: diffusion_policy.agent.bc_agent.BCAgent
```

**Purpose**: Behavior Cloning agent that wraps the DiffuseCLoC actor.

### Actor Configuration

```yaml
actor:
  _target_: diffusion_policy.modules.diffuse_cloc.DiffuseCLoC
```

**Purpose**: The main DiffuseCLoC model with co-diffusion capabilities.

### Backbone (Transformer) Configuration

```yaml
backbone:
  _target_: diffusion_policy.backbone.transformer_codiffuse.Transformer
  
  # Sequence dimensions
  x_horizon: 20                      # State prediction horizon
  y_horizon: 20                      # Action prediction horizon
  
  # Input/output dimensions
  x_input_dim: 384                   # State dimension (G1 robot state)
  y_input_dim: 29                    # Action dimension (G1 joint angles)
  x_output_dim: 384                  # State output dimension
  y_output_dim: 29                   # Action output dimension
  
  # Transformer architecture
  n_emb: 256                         # Embedding dimension
  n_head: 4                          # Number of attention heads
  n_layer: 2                         # Number of transformer layers
  
  # Attention configuration
  causal_attn: true                  # Enable custom attention masks
  x_to_x_attn: full                  # States attend to all states
  x_to_y_attn: no_attn              # States don't attend to actions
  y_to_x_attn: causal               # Actions attend to past/current states
  y_to_y_attn: causal               # Actions attend to past actions
```

**Architecture Details**:
- **Total sequence length**: 40 tokens (20 states + 20 actions, interleaved)
- **Model size**: ~19.95M parameters
- **Attention pattern**: Asymmetric masks for different token types

**Commented Values**: The configuration shows various tested alternatives:
- **Embedding dimensions**: 256, 384, 512
- **Attention heads**: 4, 6, 8
- **Layers**: 2, 4, 6, 10

### Diffusion Configuration

```yaml
# Diffusion parameters
denoising_steps: 20                  # Number of DDPM denoising steps (in rolling diffusion)
predict_epsilon: false               # Use x0 prediction (not noise prediction)
denoised_clip_value: 1.0            # Clip predictions to [-1, 1]
action_weight_schedule: constant-to-8 # Action loss weighting schedule
clean_past_state: true              # Keep past states clean during training
clean_past_action: false            # Allow past actions to be noisy

# Observation configuration
n_past_steps: 4                     # Number of past observation steps

# State emphasis
state_emphasis: random_emph_symm     # State emphasis strategy
state_proj: true                     # Enable state projection
```

**Key Features**:
- **X0 Prediction**: Model directly predicts clean data (not noise)
- **Clean Past State**: Past observations remain unnoised during training
- **State Emphasis**: Emphasizes important state features (root pose, velocities)
- **Action Weight Schedule**: Gradually increases action loss importance

## Dataset Configuration

```yaml
dataset:
  _target_: diffusion_policy.dataset.g1_offline_dataset.G1_Dataset
  
  # Sequence parameters
  horizon: null                      # Prediction horizon (inherits from actor)
  n_obs_steps: null                  # Observation steps (inherits from actor)
  pad_after: 1                       # Padding after sequences
  pad_before: 1                      # Padding before sequences
  
  # Data splitting
  seed: 42                           # Random seed for reproducible splits
  val_ratio: 0.02                    # 2% of data for validation
  
  # Data augmentation
  symm_aug: true                     # Enable left-right symmetry augmentation
  
  # Data file
  zarr_path: ankle_limit.zarr        # Dataset file path
```

**Purpose**: Configures G1 humanoid robot dataset with character frame normalization.

**Dataset Features**:
- **G1_Dataset**: Handles 384-dimensional states with character frame normalization
- **Symmetry Augmentation**: Doubles effective dataset size via left-right reflection
- **Validation Split**: 2% held-out data for monitoring generalization

**Dataset Path Comments**: Multiple dataset files are commented showing different experimental conditions:
- Various noise levels and joint stiffness settings
- Different episode counts and step durations
- Delay and perturbation experiments

## Training Parameters

```yaml
training:
  # Checkpointing
  checkpoint_every: 10               # Save checkpoint every N epochs
  
  # Debugging
  debug: false                       # Enable debug mode
  
  # Hardware
  device: cuda:0                     # GPU device
  
  # Optimization
  gradient_accumulate_every: 1       # Gradient accumulation steps
  lr_scheduler: cosine               # Learning rate scheduler type
  lr_warmup_steps: 10000            # Warmup steps for scheduler
  
  # Training limits
  max_train_steps: null              # Maximum training steps (null = no limit)
  max_val_steps: null                # Maximum validation steps
  num_epochs: 1000                   # Total training epochs
  
  # Resume training
  resume: false                      # Resume from checkpoint
  resume_path: '/path/to/checkpoint' # Path to resume from
  
  # Evaluation and sampling
  rollout_every: 10                  # Run rollout evaluation every N epochs
  rollout_steps: 200                 # Steps per rollout evaluation
  sample_every: 3                    # Generate samples every N epochs
  val_every: 10                      # Run validation every N epochs
  
  # Miscellaneous
  seed: 42                           # Training random seed
  tqdm_interval_sec: 1.0             # Progress bar update interval
  use_ema: true                      # Enable EMA model
```

**Training Schedule**:
- **Total epochs**: 1000 (can be adjusted based on convergence)
- **Validation frequency**: Every 10 epochs to monitor overfitting
- **Checkpoint frequency**: Every 10 epochs for recovery

**Learning Rate Schedule**:
- **Type**: Cosine annealing with warmup
- **Warmup**: 10,000 steps of linear increase
- **Decay**: Cosine decay to near-zero learning rate

## Checkpoint Settings

```yaml
checkpoint: 
  save_last_ckpt: true               # Always save the most recent checkpoint
```

**Purpose**: Controls checkpoint saving behavior.

**Benefits**:
- **Recovery**: Can resume training from any saved checkpoint
- **Model deployment**: Final checkpoint used for inference
- **Experiment comparison**: Checkpoints enable performance comparison across epochs

## Configuration Usage Examples

### Training with Default Configuration

```bash
python train.py --config-name=default_param
```

### Common Parameter Overrides

```bash
# Adjust batch size for different GPU memory
python train.py training.batch_size=64

# Change dataset
python train.py dataset.zarr_path=new_dataset.zarr

# Modify model architecture
python train.py policy.actor.backbone.n_layer=4 policy.actor.backbone.n_emb=512

# Extend training
python train.py training.num_epochs=2000

# Resume training
python train.py training.resume=true training.resume_path=/path/to/checkpoint
```

### Multi-run Experiments

```bash
# Hyperparameter sweep
python train.py -m optimizer.learning_rate=1e-4,5e-5,2e-4 training.seed=42,43,44
```

## Parameter Tuning Guidelines

### Memory Optimization
- **Reduce batch size**: Lower `dataloader.batch_size` if OOM
- **Reduce model size**: Decrease `n_emb`, `n_head`, or `n_layer`
- **Enable gradient accumulation**: Increase `gradient_accumulate_every`

### Performance Optimization
- **Increase batch size**: Higher throughput on powerful GPUs
- **Adjust workers**: Tune `num_workers` based on CPU cores
- **Reduce denoising steps**: Lower `denoising_steps` for faster training

### Quality Optimization
- **Increase model size**: Higher `n_emb`, `n_head`, `n_layer`
- **More denoising steps**: Higher `denoising_steps`
- **Longer training**: Increase `num_epochs`
- **Better scheduling**: Tune `lr_warmup_steps` and learning rate

This configuration provides a robust starting point for training DiffuseCLoC models on G1 humanoid robot data with proven hyperparameters and architectural choices.