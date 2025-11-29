# DiffuseCLoC Wandb Sweep Guide

This guide explains how to run a wandb sweep to tune the `x_horizon`, `y_horizon`, and `num_epochs` parameters for DiffuseCLoC training.

## Sweep Configuration

The sweep will test the following parameter combinations:
- **x_horizon & y_horizon**: [6, 12, 24, 36, 48] (25 combinations total)
- **num_epochs**: [3, 5, 10, 30]
- **Total runs**: 25 × 4 = 100 experiments

## Prerequisites

1. **Wandb account**: Make sure you have a wandb account and are logged in:
   ```bash
   wandb login
   ```

2. **Environment**: Activate your conda environment:
   ```bash
   mamba activate pdplanner
   ```

3. **Dataset**: Ensure your dataset is available at the path specified in the config:
   ```
   /home/user/CodeSpace/Diffusion/cmp_diffusion_policy/diffusion_diffuse_cloc/data/legged_gym/elspider_dataset.zarr
   ```

## Quick Start

### Option 1: Using the Helper Script (Recommended)

1. **Make the script executable:**
   ```bash
   chmod +x start_sweep.sh
   ```

2. **Initialize the sweep:**
   ```bash
   ./start_sweep.sh init
   ```

3. **Start sweep agents:**
   ```bash
   ./start_sweep.sh agent
   ```

4. **Check sweep status:**
   ```bash
   ./start_sweep.sh status
   ```

### Option 2: Manual Setup

1. **Initialize the sweep:**
   ```bash
   wandb sweep sweep_config.yaml
   ```
   This will output a sweep ID like: `your-entity/diffuse_cloc/sweep_id`

2. **Run sweep agents:**
   ```bash
   wandb agent your-entity/diffuse_cloc/sweep_id
   ```

## Running Multiple Agents

To speed up the sweep, you can run multiple agents in parallel:

```bash
# Terminal 1
./start_sweep.sh agent

# Terminal 2  
./start_sweep.sh agent

# Terminal 3
./start_sweep.sh agent
```

Each agent will pick up the next parameter combination from the sweep queue.

## Files Overview

- **`sweep_config.yaml`**: Wandb sweep configuration defining parameters to sweep
- **`train_sweep.py`**: Training script that integrates with wandb sweeps
- **`legged_gym_diffuse_sweep.yaml`**: Base training config (modified for sweeps)
- **`start_sweep.sh`**: Helper script to manage sweep operations

## Monitoring Results

1. **Wandb Dashboard**: Visit your wandb dashboard to monitor progress:
   ```
   https://wandb.ai/your-entity/diffuse_cloc/sweeps/sweep_id
   ```

2. **Local Status**: Use the helper script:
   ```bash
   ./start_sweep.sh status
   ```

## Understanding Results

The sweep will optimize for `val_loss` (validation loss). After completion, you can:

1. **View parallel coordinates plot** in wandb to see parameter relationships
2. **Check the best performing runs** based on validation loss
3. **Compare training curves** across different parameter combinations

## Key Changes for Sweep

The sweep setup makes these important changes:

1. **CLoCAnalyzer disabled** during training for speed
2. **Resume disabled** to ensure clean runs
3. **x_horizon = y_horizon constraint** enforced in the sweep
4. **Dataset horizon** automatically updated to match x_horizon
5. **Experiment names** include parameter values for easy identification

## Troubleshooting

### Common Issues

1. **"wandb not found"**: Install wandb: `pip install wandb`
2. **"Not logged in"**: Run `wandb login`
3. **Dataset not found**: Check the `zarr_path` in `legged_gym_diffuse_sweep.yaml`
4. **CUDA out of memory**: Reduce batch size in config or use smaller horizons first

### Tips

1. **Start small**: Test with a few parameter combinations first
2. **Monitor GPU usage**: Larger horizons require more memory
3. **Check disk space**: Each run creates checkpoints and logs
4. **Use screen/tmux**: For long-running sweeps on remote machines

## Expected Results

Based on your training log observations:
- **Smaller horizons (6, 12)** may train faster but walk slower
- **Larger horizons (36, 48)** may provide better walking speed
- **Optimal horizon** likely between 12-36 for your ElSpider setup
- **Longer training (30 epochs)** should show better convergence

## Next Steps

After the sweep completes:

1. **Identify best parameters** from wandb dashboard
2. **Run final training** with optimal parameters and more epochs
3. **Evaluate best model** using `eval.py` with CLoCAnalyzer enabled
4. **Deploy to robot** using the elspider_air_cloc.py deployment script