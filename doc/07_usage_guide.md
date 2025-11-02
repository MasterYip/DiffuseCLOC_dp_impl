# Usage Guide

## Overview

This guide provides practical examples for training, evaluating, and deploying DiffuseCLoC models. It covers common workflows, configuration management, and troubleshooting tips.

## Quick Start

### Installation

```bash
# Navigate to project directory
cd /home/user/CodeSpace/Diffusion/cmp_diffusion_policy/diffusion_diffuse_cloc

# Install dependencies (create requirements.txt if needed)
pip install torch>=1.12 hydra-core>=1.2 wandb zarr numpy scipy
pip install pygame moderngl pyrr  # For visualization (optional)
```

### Project Setup

```bash
# Verify file structure
ls diffusion_policy/
# Should show: backbone/ modules/ dataset/ trainer/ agent/ utils/ config_files/

# Check main config
cat diffusion_policy/config_files/joint_diffuse.yaml
```

## 1. Training a Model

### Basic Training

```bash
# Train with default configuration
python train.py --config-name=joint_diffuse

# Train with custom config overrides
python train.py --config-name=joint_diffuse \
    training.batch_size=64 \
    training.learning_rate=5e-5 \
    training.n_epochs=2000
```

### Configuration Override Examples

#### Dataset Configuration
```bash
# Use different dataset
python train.py dataset.zarr_path=/path/to/new_data.zarr

# Change horizon length
python train.py \
    policy.actor.x_horizon=30 \
    policy.actor.y_horizon=30 \
    dataset.horizon=30

# Disable symmetry augmentation
python train.py dataset.augment_symmetry=false
```

#### Model Architecture
```bash
# Larger transformer
python train.py \
    policy.actor.backbone.n_layer=4 \
    policy.actor.backbone.n_head=8 \
    policy.actor.backbone.n_emb=512

# Different denoising steps
python train.py policy.actor.denoising_steps=50
```

#### Training Parameters
```bash
# Longer training with smaller batch
python train.py \
    training.n_epochs=5000 \
    training.batch_size=32 \
    training.learning_rate=2e-4

# Multi-GPU training (if supported)
python train.py \
    training.distributed=true \
    training.world_size=4
```

### Experiment Management

#### Using Hydra Multirun
```bash
# Hyperparameter sweep
python train.py -m \
    training.learning_rate=1e-4,5e-5,2e-4 \
    training.batch_size=64,128,256

# Random seeds for statistical significance
python train.py -m training.seed=42,43,44,45,46
```

#### Wandb Integration
```bash
# Enable wandb logging
python train.py \
    training.use_wandb=true \
    training.wandb_project=my_project \
    training.wandb_run_name=experiment_1

# Disable wandb
python train.py training.use_wandb=false
```

## 2. Model Inference and Evaluation

### Loading a Trained Model

```python
import torch
import hydra
from omegaconf import OmegaConf
from diffusion_policy.agent.bc_agent import BCAgent

# Load checkpoint
checkpoint_path = "outputs/2024-01-01/12-00-00/checkpoints/latest.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

# Reconstruct config
cfg = OmegaConf.create(checkpoint['config'])

# Initialize agent
agent = hydra.utils.instantiate(cfg.policy)
agent.load_state_dict(checkpoint['agent'])
agent.eval()

# Load normalizer
normalizer = hydra.utils.instantiate(cfg.dataset).get_normalizer()
normalizer.load_state_dict(checkpoint['normalizer'])
agent.set_normalizer(normalizer)
```

### Single Step Inference

```python
import numpy as np

# Prepare observation (past 4 timesteps)
obs_history = np.random.randn(1, 4, 384)  # (batch=1, history=4, state_dim=384)
obs_dict = {'obs': torch.from_numpy(obs_history).float().to(device)}

# Get action prediction
with torch.no_grad():
    result = agent.act(obs_dict, deterministic=True)
    
action = result['action'].cpu().numpy()  # (1, 29)
next_state_pred = result.get('state', None)  # (1, 384) if available

print(f"Predicted action: {action.shape}")
print(f"Joint angles: {action[0, :6]}")  # First 6 joints (hips)
```

### Rollout Evaluation

```python
def evaluate_rollout(agent, initial_obs, max_steps=1000):
    """Evaluate model performance over a full episode"""
    
    # Initialize trajectory storage
    obs_trajectory = [initial_obs]
    action_trajectory = []
    
    # Rolling observation buffer (past 4 timesteps)
    obs_buffer = np.zeros((1, 4, 384))
    obs_buffer[0, -1] = initial_obs  # Set current observation
    
    for step in range(max_steps):
        # Prepare observation dict
        obs_dict = {'obs': torch.from_numpy(obs_buffer).float().to(device)}
        
        # Get action
        with torch.no_grad():
            result = agent.act(obs_dict, deterministic=True)
        
        action = result['action'].cpu().numpy()[0]  # (29,)
        action_trajectory.append(action)
        
        # Simulate next state (replace with actual simulator)
        next_obs = simulate_physics_step(obs_buffer[0, -1], action)
        
        # Update observation buffer (rolling window)
        obs_buffer = np.roll(obs_buffer, -1, axis=1)
        obs_buffer[0, -1] = next_obs
        
        obs_trajectory.append(next_obs)
        
        # Check termination conditions
        if is_episode_done(next_obs):
            break
    
    return {
        'observations': np.array(obs_trajectory),
        'actions': np.array(action_trajectory),
        'episode_length': len(action_trajectory)
    }

# Run evaluation
rollout_data = evaluate_rollout(agent, initial_observation)
print(f"Episode completed in {rollout_data['episode_length']} steps")
```

### Batch Evaluation

```python
def evaluate_dataset(agent, eval_dataset, n_episodes=100):
    """Evaluate model on multiple episodes"""
    
    results = {
        'episode_lengths': [],
        'success_rates': [],
        'avg_rewards': []
    }
    
    for episode_idx in range(n_episodes):
        # Get episode data from dataset
        episode_data = eval_dataset.get_episode(episode_idx)
        initial_obs = episode_data['obs'][0]
        
        # Run rollout
        rollout = evaluate_rollout(agent, initial_obs)
        
        # Compute metrics
        results['episode_lengths'].append(rollout['episode_length'])
        results['success_rates'].append(compute_success_rate(rollout))
        results['avg_rewards'].append(compute_average_reward(rollout))
    
    # Aggregate results
    print(f"Average episode length: {np.mean(results['episode_lengths']):.1f}")
    print(f"Success rate: {np.mean(results['success_rates']):.3f}")
    print(f"Average reward: {np.mean(results['avg_rewards']):.3f}")
    
    return results
```

## 3. Configuration Management

### Creating Custom Configs

#### Custom Dataset Config
```yaml
# config/dataset/my_dataset.yaml
_target_: diffusion_policy.dataset.g1_offline_dataset.G1_Dataset
zarr_path: /path/to/my_data.zarr
horizon: 25
pad_before: 6
augment_symmetry: true
val_ratio: 0.05
```

#### Custom Model Config
```yaml
# config/model/large_transformer.yaml
_target_: diffusion_policy.modules.diffuse_cloc.DiffuseCLoC
backbone:
  _target_: diffusion_policy.backbone.transformer_codiffuse.Transformer
  n_layer: 6
  n_head: 12
  n_emb: 768
denoising_steps: 100
rolling_inference: true
```

#### Custom Training Config
```yaml
# config/training/fast_training.yaml
device: cuda:0
n_epochs: 1000
batch_size: 256
learning_rate: 2e-4
lr_scheduler: linear
ema_decay: 0.995
checkpoint_every: 50
```

### Using Custom Configs
```bash
# Use custom configurations
python train.py \
    dataset=my_dataset \
    model=large_transformer \
    training=fast_training
```

## 4. Data Preprocessing

### Creating Dataset from Raw Data

```python
import zarr
from diffusion_policy.dataset.replay_buffer import ReplayBuffer

def create_zarr_dataset(episodes_data, output_path):
    """Convert episode data to zarr format"""
    
    # Create zarr group
    root = zarr.open(output_path, mode='w')
    
    # Compute episode boundaries
    episode_ends = []
    total_steps = 0
    
    for episode in episodes_data:
        episode_length = len(episode['obs'])
        total_steps += episode_length
        episode_ends.append(total_steps)
    
    # Create arrays
    obs_array = root.create_dataset(
        'data/obs', 
        shape=(total_steps, 384), 
        dtype=np.float32,
        chunks=(1000, 384),
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    action_array = root.create_dataset(
        'data/action',
        shape=(total_steps, 29),
        dtype=np.float32,
        chunks=(1000, 29),
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )
    
    # Fill arrays
    start_idx = 0
    for episode in episodes_data:
        end_idx = start_idx + len(episode['obs'])
        obs_array[start_idx:end_idx] = episode['obs']
        action_array[start_idx:end_idx] = episode['action']
        start_idx = end_idx
    
    # Create metadata
    root.create_dataset('meta/episode_ends', data=episode_ends)
    
    print(f"Created dataset with {len(episodes_data)} episodes, {total_steps} total steps")
    print(f"Saved to: {output_path}")

# Usage
episodes = load_raw_episodes()  # Your data loading function
create_zarr_dataset(episodes, "data/processed_dataset.zarr")
```

### Data Validation

```python
def validate_dataset(zarr_path):
    """Validate dataset format and statistics"""
    
    # Load dataset
    dataset = G1_Dataset(zarr_path=zarr_path, horizon=20, pad_before=4)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Check sample format
    sample = dataset[0]
    print(f"Observation shape: {sample['obs'].shape}")  # Should be (4, 384)
    print(f"Action shape: {sample['action'].shape}")    # Should be (20, 29)
    
    # Check data ranges
    all_obs = []
    all_actions = []
    
    for i in range(min(1000, len(dataset))):  # Sample first 1000
        sample = dataset[i]
        all_obs.append(sample['obs'])
        all_actions.append(sample['action'])
    
    obs_stats = torch.cat(all_obs, dim=0)  # (N*4, 384)
    action_stats = torch.cat(all_actions, dim=0)  # (N*20, 29)
    
    print(f"Observation range: [{obs_stats.min():.3f}, {obs_stats.max():.3f}]")
    print(f"Action range: [{action_stats.min():.3f}, {action_stats.max():.3f}]")
    print(f"Observation mean: {obs_stats.mean():.3f}")
    print(f"Action mean: {action_stats.mean():.3f}")

validate_dataset("data/my_dataset.zarr")
```

## 5. Deployment and Real-Time Usage

### Real-Time Control Loop

```python
class DiffuseCLoCController:
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = torch.device(device)
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        cfg = OmegaConf.create(checkpoint['config'])
        
        self.agent = hydra.utils.instantiate(cfg.policy)
        self.agent.load_state_dict(checkpoint['agent'])
        self.agent.to(self.device)
        self.agent.eval()
        
        # Load normalizer
        normalizer = hydra.utils.instantiate(cfg.dataset).get_normalizer()
        normalizer.load_state_dict(checkpoint['normalizer'])
        self.agent.set_normalizer(normalizer)
        
        # Initialize observation buffer
        self.obs_buffer = np.zeros((1, 4, 384))
        self.initialized = False
    
    def reset(self, initial_obs):
        """Reset controller with initial observation"""
        self.obs_buffer.fill(0)
        self.obs_buffer[0, -1] = initial_obs
        self.initialized = True
    
    def get_action(self, current_obs):
        """Get action for current observation"""
        if not self.initialized:
            raise ValueError("Controller not initialized. Call reset() first.")
        
        # Update observation buffer
        self.obs_buffer = np.roll(self.obs_buffer, -1, axis=1)
        self.obs_buffer[0, -1] = current_obs
        
        # Get action prediction
        obs_dict = {'obs': torch.from_numpy(self.obs_buffer).float().to(self.device)}
        
        with torch.no_grad():
            result = self.agent.act(obs_dict, deterministic=True)
        
        action = result['action'].cpu().numpy()[0]  # (29,)
        return action

# Usage in control loop
controller = DiffuseCLoCController("checkpoints/best_model.ckpt")

# Initialize with first observation
initial_obs = get_robot_state()  # Your robot interface
controller.reset(initial_obs)

# Control loop
for step in range(1000):
    # Get current robot state
    current_obs = get_robot_state()
    
    # Predict action
    action = controller.get_action(current_obs)
    
    # Send to robot
    send_action_to_robot(action)
    
    # Wait for next timestep (e.g., 32 Hz = 31.25ms)
    time.sleep(0.03125)
```

### Performance Optimization

```python
# Use torch.jit for faster inference
@torch.jit.script
def fast_forward(model, obs):
    return model.act(obs, deterministic=True)

# Compile model
traced_model = torch.jit.trace(agent.actor, example_input)
traced_model.save("compiled_model.pt")

# Use half precision for speed
agent.half()  # Convert to FP16
obs_dict = {k: v.half() for k, v in obs_dict.items()}
```

## 6. Troubleshooting

### Common Issues and Solutions

#### Training Issues

**Loss not decreasing:**
```bash
# Check learning rate
python train.py training.learning_rate=1e-3  # Increase LR

# Check batch size
python train.py training.batch_size=64  # Reduce batch size

# Check gradient clipping
python train.py training.max_grad_norm=10.0  # Increase clip threshold
```

**Out of memory:**
```bash
# Reduce batch size
python train.py training.batch_size=32

# Reduce model size
python train.py \
    policy.actor.backbone.n_layer=1 \
    policy.actor.backbone.n_emb=128

# Enable gradient accumulation
python train.py \
    training.batch_size=32 \
    training.gradient_accumulate_every=4  # Effective batch size: 128
```

**Training too slow:**
```bash
# Reduce denoising steps
python train.py policy.actor.denoising_steps=10

# Use smaller model
python train.py \
    policy.actor.backbone.n_layer=1 \
    policy.actor.backbone.n_head=2
```

#### Data Issues

**Dataset loading errors:**
```python
# Check zarr file integrity
import zarr
root = zarr.open("data.zarr", mode='r')
print(root.tree())  # Should show data/ and meta/ groups

# Validate episode boundaries
episode_ends = root['meta/episode_ends'][:]
print(f"Episodes: {len(episode_ends)}")
print(f"Total timesteps: {episode_ends[-1]}")
```

**Normalization issues:**
```python
# Check normalizer statistics
normalizer = dataset.get_normalizer(mode='limits')
print("Observation limits:", normalizer.params['obs'])
print("Action limits:", normalizer.params['action'])

# Manual normalization check
sample = dataset[0]
normalized = normalizer.normalize(sample)
recovered = normalizer.unnormalize(normalized)
error = torch.abs(sample['obs'] - recovered['obs']).max()
print(f"Normalization error: {error}")  # Should be < 1e-6
```

#### Inference Issues

**Poor performance:**
```python
# Use EMA model for inference
with ema_model.average_parameters():
    result = agent.act(obs_dict)

# Check deterministic mode
result = agent.act(obs_dict, deterministic=True)

# Verify normalization
obs_raw = get_robot_state()
obs_normalized = normalizer.normalize({'obs': obs_raw})['obs']
print(f"Normalized obs range: [{obs_normalized.min()}, {obs_normalized.max()}]")
```

### Debugging Tools

#### Logging Model Information
```python
def log_model_info(agent):
    """Print model architecture and parameter count"""
    print("Model Architecture:")
    print(agent.actor)
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (FP32)")

log_model_info(agent)
```

#### Gradient Analysis
```python
def analyze_gradients(model):
    """Analyze gradient statistics during training"""
    grad_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
    
    # Sort by gradient magnitude
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 gradient magnitudes:")
    for name, norm in sorted_grads[:10]:
        print(f"  {name}: {norm:.6f}")
    
    return grad_norms

# Use during training
gradients = analyze_gradients(agent)
```

## 7. Performance Benchmarks

### Training Performance
```bash
# Benchmark training speed
python train.py \
    training.n_epochs=10 \
    training.checkpoint_every=5 \
    +benchmark=true  # Custom flag for timing

# Expected results on A100:
# - Batch size 128: ~30 seconds/epoch
# - Total training (3000 epochs): ~25 hours
```

### Inference Performance
```python
import time

def benchmark_inference(agent, n_samples=1000):
    """Benchmark inference speed"""
    
    # Prepare random inputs
    obs_batch = torch.randn(1, 4, 384, device=agent.device)
    obs_dict = {'obs': obs_batch}
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = agent.act(obs_dict, deterministic=True)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_samples):
        with torch.no_grad():
            _ = agent.act(obs_dict, deterministic=True)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_samples * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference rate: {fps:.1f} Hz")
    
    return avg_time

benchmark_inference(agent)
# Expected: ~30ms on A100, ~50ms on RTX 3090
```

This completes the comprehensive documentation bundle for DiffuseCLoC! The documentation now covers:

1. **README.md** - Project overview and quick start
2. **01_architecture_overview.md** - System architecture and mathematical framework
3. **02_file_structure.md** - Code organization and dependencies
4. **03_transformer_architecture.md** - Detailed transformer implementation
5. **04_diffusion_models.md** - DDPM mathematics and implementations
6. **05_dataset_normalization.md** - Data processing and character frame normalization
7. **06_training_pipeline.md** - Training loop and optimization
8. **07_usage_guide.md** - Practical examples and deployment guide

Each document provides both theoretical understanding and practical implementation details, with code examples, configuration snippets, and troubleshooting guidance.