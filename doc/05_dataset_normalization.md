# Dataset and Normalization

## Overview

This document covers the dataset structure, normalization procedures, and data processing pipeline used in DiffuseCLoC. The system uses G1 robot trajectory data with sophisticated character frame normalization and symmetry augmentation.

## Dataset Architecture

```mermaid
graph TB
    subgraph "Storage Layer"
        ZARR[Zarr Archive<br/>Compressed Storage]
        EPISODES[Episodes<br/>Variable Length]
    end
    
    subgraph "Dataset Classes"
        ZARR --> BASE[G1DatasetBase<br/>Common functionality]
        BASE --> G1[G1_Dataset<br/>Standard 384-dim states]
        BASE --> G1_EE[G1_Dataset_EE<br/>+End-effector rotations]
        BASE --> G1_LIM[G1_Dataset_limited<br/>Subset of bodies]
    end
    
    subgraph "Processing Pipeline"
        G1 --> NORM[Character Frame<br/>Normalization]
        NORM --> SYMM[Symmetry<br/>Augmentation]
        SYMM --> BATCH[Batching<br/>& Padding]
    end
    
    subgraph "Training Data"
        BATCH --> LOADER[DataLoader<br/>Sequence Sampling]
    end
```

## File Organization

### Core Files

- **`replay_buffer.py`**: Zarr-based storage backend
- **`offline_dataset.py`**: Base dataset classes and interfaces
- **`g1_offline_dataset.py`**: G1-specific implementations with normalization
- **`sampler.py`**: Sequence sampling utilities (if exists)

## 1. Zarr Storage Backend

**File**: `diffusion_policy/dataset/replay_buffer.py`

### ReplayBuffer Class

The `ReplayBuffer` provides efficient storage and retrieval of trajectory data using Zarr arrays.

#### Key Features

1. **Compressed Storage**: Uses LZ4/Zstd compression
2. **Episode Organization**: Stores variable-length episodes
3. **Memory Mapping**: Efficient access without loading entire dataset
4. **Chunked Arrays**: Optimized for sequential access patterns

#### Data Structure

```python
# Zarr group structure
replay_buffer/
├── meta/
│   ├── episode_ends        # [ep0_len, ep0_len+ep1_len, ...]
│   └── episode_starts      # [0, ep0_len, ep0_len+ep1_len, ...]
└── data/
    ├── obs                 # States: (total_timesteps, 384)
    ├── action              # Actions: (total_timesteps, 29)
    └── ...                 # Additional fields
```

#### Usage Example

```python
# Load dataset
buffer = ReplayBuffer.copy_from_path("data.zarr", keys=['obs', 'action'])

# Get episode
episode = buffer.get_episode(episode_idx=0)
# Returns: {'obs': (T, 384), 'action': (T, 29)}

# Get sequence slice
sequence = buffer.sample_sequence(
    episode_idx=0, 
    start_ts=10, 
    end_ts=30
)
```

## 2. Base Dataset Classes

**File**: `diffusion_policy/dataset/offline_dataset.py`

### BaseDataset

Abstract interface for all datasets:

```python
class BaseDataset(torch.utils.data.Dataset):
    def get_normalizer(self, mode='limits', **kwargs):
        # Create normalizer fitted to this dataset
        pass
    
    def get_validation_dataset(self):
        # Split into train/val
        pass
```

### OfflineDataset

Base for offline (pre-collected) datasets:

```python
class OfflineDataset(BaseDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, ...):
        # ...existing code...
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path)
        self.horizon = horizon           # Sequence length
        self.pad_before = pad_before     # Past context
        self.pad_after = pad_after       # Future context
```

#### Sequence Sampling

```python
def __getitem__(self, idx):
    # Sample sequence starting at timestep idx
    buffer_start_idx, buffer_end_idx = self.indices[idx]
    
    # Get raw sequence
    sample = self.replay_buffer.sample_sequence(
        start_ts=buffer_start_idx,
        end_ts=buffer_end_idx
    )
    
    # Apply normalization (implemented by subclass)
    sample = self.state_normalize(sample)
    
    return sample
```

## 3. G1 Dataset Implementation

**File**: `diffusion_policy/dataset/g1_offline_dataset.py`

### G1DatasetBase

Shared functionality for all G1 dataset variants:

#### Character Frame Definition

The character frame is a **body-centric coordinate system** that removes global translation and rotation:

- **Origin**: Root position at the **nominal timestep** (index 3 in 4-timestep history)
- **Orientation**: **Yaw-only rotation** (gravity-aligned, no pitch/roll)
- **Benefits**: Translation/rotation invariance for policy learning

#### Root State Extraction

```python
def get_root_state(self, obs):
    """Extract root pose and velocity from observation"""
    # obs shape: (B, T, 384) or (T, 384)
    
    # Root position (last 3 dims)
    root_pos = obs[..., -6:-3]      # (B, T, 3)
    
    # Root orientation (quaternion -> euler -> yaw)
    root_quat = obs[..., -10:-6]    # (B, T, 4)
    root_euler = get_euler_xyz(root_quat)  # (B, T, 3)
    root_yaw = root_euler[..., 2:3] # (B, T, 1) - only yaw
    
    # Root velocities
    root_lin_vel = obs[..., -3:]    # (B, T, 3)
    # Angular velocity computed from quaternion differences
    
    return {
        'pos': root_pos,
        'yaw': root_yaw, 
        'lin_vel': root_lin_vel,
        # ...
    }
```

### Character Frame Normalization

#### 1. Compute Character Frame

For a sequence with past observations:

```python
def get_character_frame(self, obs_sequence):
    """
    obs_sequence: (B, T_past, 384) where T_past = 4
    Returns character frame at nominal timestep (index 3)
    """
    nominal_idx = 3  # Use last observation as reference
    
    # Extract root state at nominal timestep
    root_state = self.get_root_state(obs_sequence[:, nominal_idx])
    
    # Character frame transformation
    char_pos = root_state['pos']        # (B, 3)
    char_yaw = root_state['yaw']        # (B, 1)
    char_rot_mat = yaw_to_rotation_matrix(char_yaw)  # (B, 3, 3)
    
    return {
        'pos': char_pos,
        'rot_mat': char_rot_mat,
        'yaw': char_yaw
    }
```

#### 2. Transform States to Character Frame

```python
def state_normalize(self, sample):
    """Transform all states to character frame"""
    
    obs_seq = sample['obs']  # (B, T_past, 384)
    
    # Get character frame from nominal timestep
    char_frame = self.get_character_frame(obs_seq)
    
    # Transform each timestep
    normalized_obs = []
    for t in range(obs_seq.shape[1]):
        obs_t = obs_seq[:, t]  # (B, 384)
        
        # Extract body positions and velocities
        body_pos = obs_t[:, 0:90].reshape(B, 30, 3)      # (B, 30, 3)
        body_vel = obs_t[:, 90:180].reshape(B, 30, 3)    # (B, 30, 3)
        
        # Transform to character frame
        # Position: rotate then translate
        body_pos_char = torch.bmm(
            char_frame['rot_mat'].transpose(-1, -2),  # Inverse rotation
            (body_pos - char_frame['pos'].unsqueeze(1)).transpose(-1, -2)
        ).transpose(-1, -2)
        
        # Velocity: only rotate (no translation)
        body_vel_char = torch.bmm(
            char_frame['rot_mat'].transpose(-1, -2),
            body_vel.transpose(-1, -2)
        ).transpose(-1, -2)
        
        # Transform root state relative to character frame
        root_pos_char, root_rot_char = self.transform_root_state(
            obs_t, char_frame
        )
        
        # Reassemble normalized state
        obs_t_norm = torch.cat([
            body_pos_char.reshape(B, 90),      # Body positions
            body_vel_char.reshape(B, 90),      # Body velocities  
            root_pos_char,                     # Root position (relative)
            root_rot_char,                     # Root rotation (relative)
            # ... other features
        ], dim=-1)
        
        normalized_obs.append(obs_t_norm)
    
    sample['obs'] = torch.stack(normalized_obs, dim=1)
    return sample
```

#### 3. Root State Transformation

```python
def transform_root_state(self, obs, char_frame):
    """Transform root state to character-relative coordinates"""
    
    # Extract root pose
    root_pos_global = obs[..., -6:-3]    # (B, 3)
    root_quat_global = obs[..., -10:-6]  # (B, 4)
    
    # Position: translate to character origin, then rotate
    root_pos_rel = root_pos_global - char_frame['pos']  # (B, 3)
    root_pos_char = torch.bmm(
        char_frame['rot_mat'].transpose(-1, -2),
        root_pos_rel.unsqueeze(-1)
    ).squeeze(-1)
    
    # Rotation: relative to character yaw
    root_euler_global = get_euler_xyz(root_quat_global)  # (B, 3)
    root_yaw_rel = root_euler_global[..., 2:3] - char_frame['yaw']  # (B, 1)
    
    # Keep pitch and roll unchanged (gravity-aligned assumption)
    root_euler_rel = torch.cat([
        root_euler_global[..., :2],  # pitch, roll unchanged
        root_yaw_rel                 # yaw relative to character
    ], dim=-1)
    
    # Convert back to rotation vector representation
    root_rotvec = euler_to_rotation_vector(root_euler_rel)  # (B, 3)
    
    return root_pos_char, root_rotvec
```

### Unnormalization (Inference)

For inference, predicted states must be transformed back to global coordinates:

```python
def state_unnormalize(self, normalized_sample, char_frame):
    """Transform normalized states back to global coordinates"""
    
    obs_norm = normalized_sample['obs']  # (B, T, 384)
    
    # Reverse the character frame transformation
    unnormalized_obs = []
    for t in range(obs_norm.shape[1]):
        obs_t_norm = obs_norm[:, t]  # (B, 384)
        
        # Extract normalized components
        body_pos_char = obs_t_norm[:, 0:90].reshape(B, 30, 3)
        body_vel_char = obs_t_norm[:, 90:180].reshape(B, 30, 3)
        
        # Transform back to global frame
        # Position: rotate then translate
        body_pos_global = torch.bmm(
            char_frame['rot_mat'], 
            body_pos_char.transpose(-1, -2)
        ).transpose(-1, -2) + char_frame['pos'].unsqueeze(1)
        
        # Velocity: only rotate
        body_vel_global = torch.bmm(
            char_frame['rot_mat'],
            body_vel_char.transpose(-1, -2) 
        ).transpose(-1, -2)
        
        # Transform root state back to global
        root_pos_global, root_quat_global = self.untransform_root_state(
            obs_t_norm, char_frame
        )
        
        # Reassemble global state
        obs_t_global = torch.cat([
            body_pos_global.reshape(B, 90),
            body_vel_global.reshape(B, 90),
            root_pos_global,
            root_quat_global,
            # ... other features
        ], dim=-1)
        
        unnormalized_obs.append(obs_t_global)
    
    normalized_sample['obs'] = torch.stack(unnormalized_obs, dim=1)
    return normalized_sample
```

## 4. Dataset Variants

### G1_Dataset (Standard)

**State Dimension**: 384

```python
class G1_Dataset(G1DatasetBase):
    def __init__(self, zarr_path, horizon=20, pad_before=4, pad_after=0, **kwargs):
        # Standard implementation with 384-dim states
        # Body positions (90) + velocities (90) + root state (204)
        pass
```

**State Breakdown**:
- **Body positions**: 30 bodies × 3 coords = 90 dims
- **Body velocities**: 30 bodies × 3 coords = 90 dims  
- **Root position**: 3 dims (relative to character frame)
- **Root rotation**: 3 dims (rotation vector, relative yaw)
- **Root linear velocity**: 3 dims
- **Root angular velocity**: 3 dims
- **Additional features**: ~192 dims (joint angles, etc.)

### G1_Dataset_EE (Extended)

**State Dimension**: 384 + N (where N varies)

```python
class G1_Dataset_EE(G1DatasetBase):
    def __init__(self, zarr_path, **kwargs):
        # Adds end-effector orientation information
        # Useful for manipulation tasks
        pass
```

**Additional Features**:
- **End-effector rotations**: Wrist, ankle orientations
- **Contact information**: Foot/hand contact states
- **Task-specific features**: Object poses, etc.

### G1_Dataset_limited (Subset)

**State Dimension**: Reduced (subset of bodies)

```python
class G1_Dataset_limited(G1DatasetBase):
    def __init__(self, zarr_path, body_indices=None, **kwargs):
        # Uses only subset of body parts (e.g., upper body only)
        self.body_indices = body_indices or [0, 1, 2, ...]  # Specific bodies
        pass
```

**Use Cases**:
- **Upper body control**: Arms and torso only
- **Lower body control**: Legs and pelvis only
- **Reduced complexity**: Fewer parameters for specific tasks

## 5. Symmetry Augmentation

### Left-Right Reflection

DiffuseCLoC uses left-right symmetry to double the effective dataset size:

```python
def get_reflection_ops(self):
    """Get reflection operators for left-right symmetry"""
    
    # State reflection matrix (384 x 384)
    obs_reflect_matrix = torch.eye(384)
    
    # Flip left-right body pairs
    left_indices = [0, 3, 6, ...]   # Left hip, shoulder, etc.
    right_indices = [1, 4, 7, ...]  # Right hip, shoulder, etc.
    
    # Swap left and right
    obs_reflect_matrix[left_indices] = 0
    obs_reflect_matrix[right_indices] = 0
    obs_reflect_matrix[left_indices, right_indices] = 1
    obs_reflect_matrix[right_indices, left_indices] = 1
    
    # Flip y-coordinates (left-right axis)
    y_coord_indices = [1, 4, 7, ...]  # y-components of positions/velocities
    obs_reflect_matrix[y_coord_indices, y_coord_indices] = -1
    
    # Action reflection (similar for joint angles)
    action_reflect_matrix = get_action_reflection_matrix()
    
    return obs_reflect_matrix, action_reflect_matrix
```

### Data Augmentation in Collate Function

```python
def collate_fn(self, batch):
    """Custom batching with optional symmetry augmentation"""
    
    # Standard batching
    batch_dict = dict_apply(batch, lambda x: torch.stack(x, dim=0))
    
    # Optional symmetry augmentation (50% chance)
    if self.augment_symmetry and torch.rand(1) > 0.5:
        obs_reflect, action_reflect = self.get_reflection_ops()
        
        # Apply reflection
        batch_dict['obs'] = torch.matmul(batch_dict['obs'], obs_reflect.T)
        batch_dict['action'] = torch.matmul(batch_dict['action'], action_reflect.T)
    
    return batch_dict
```

## 6. Linear Normalizer

**File**: `diffusion_policy/utils/normalizer.py`

### Multi-Field Normalization

The `LinearNormalizer` handles different data fields separately:

```python
normalizer = LinearNormalizer()

# Fit on training data
normalizer.fit({
    'obs': train_observations,      # (N, T, 384)
    'action': train_actions         # (N, T, 29)
}, mode='limits')  # or 'gaussian'

# Apply normalization
normalized_batch = normalizer.normalize(batch)
original_batch = normalizer.unnormalize(normalized_batch)
```

### Normalization Modes

#### 1. Limits Mode (Min-Max Scaling)

$$x_{\text{norm}} = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1$$

Maps data to $[-1, 1]$ range.

#### 2. Gaussian Mode (Z-Score)

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

Maps data to zero mean, unit variance.

### Implementation

```python
class LinearNormalizer:
    def fit(self, dataset_dict, mode='limits'):
        self.mode = mode
        for key, data in dataset_dict.items():
            if mode == 'limits':
                self.params[key] = {
                    'min': data.min(dim=(0,1), keepdim=True)[0],
                    'max': data.max(dim=(0,1), keepdim=True)[0]
                }
            elif mode == 'gaussian':
                self.params[key] = {
                    'mean': data.mean(dim=(0,1), keepdim=True),
                    'std': data.std(dim=(0,1), keepdim=True)
                }
    
    def normalize(self, data_dict):
        # Apply fitted normalization
        # ...existing code...
    
    def unnormalize(self, data_dict):
        # Reverse normalization
        # ...existing code...
```

## 7. Data Loading Pipeline

### Training Data Flow

```mermaid
sequenceDiagram
    participant Trainer as OfflineTrainer
    participant Dataset as G1_Dataset
    participant Buffer as ReplayBuffer
    participant Loader as DataLoader
    
    Trainer->>Dataset: Initialize with zarr_path
    Dataset->>Buffer: Load from zarr archive
    Buffer->>Dataset: Return episode data
    Dataset->>Dataset: Character frame normalization
    Dataset->>Loader: Provide normalized sequences
    Loader->>Dataset: collate_fn (batching + augmentation)
    Dataset->>Trainer: Final batch dict
```

### Configuration Example

```yaml
dataset:
  _target_: diffusion_policy.dataset.g1_offline_dataset.G1_Dataset
  zarr_path: /path/to/data.zarr
  horizon: 20                    # Prediction horizon
  pad_before: 4                  # Past context (n_obs_steps)
  pad_after: 0                   # Future context
  seed: 42                       # For train/val split
  val_ratio: 0.02               # 2% validation split
  augment_symmetry: true        # Enable left-right flipping

dataloader:
  batch_size: 128
  num_workers: 4
  persistent_workers: true
  pin_memory: true
```

## 8. Key Design Principles

### Character Frame Benefits

1. **Translation Invariance**: Policy works regardless of global position
2. **Rotation Invariance**: Policy works regardless of global orientation  
3. **Generalization**: Better transfer to new environments/starting positions
4. **Numerical Stability**: Smaller coordinate values, better conditioning

### Symmetry Augmentation Benefits

1. **Data Efficiency**: Effective 2x increase in dataset size
2. **Symmetric Policies**: Encourages left-right symmetric behavior
3. **Robustness**: Better performance on mirror tasks
4. **Reduced Bias**: Prevents overfitting to one-sided demonstrations

### Normalization Strategy

1. **Separate Fields**: Different normalization for obs vs. action
2. **Preserve Structure**: Character frame before normalization
3. **Consistent Scaling**: All features in similar numerical ranges
4. **Reversible**: Perfect reconstruction for inference