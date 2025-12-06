#!/usr/bin/env python3
"""
Sweep-compatible training script for DiffuseCLoC.
This script integrates with wandb sweeps to automatically handle parameter updates.
"""

import os
import sys
import wandb
import hydra
import time
from omegaconf import OmegaConf
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from diffusion_policy.trainer.base_trainer import BaseTrainer

# Register eval resolver like in train.py
OmegaConf.register_new_resolver("eval", eval, replace=True)


def update_config_with_sweep_params(cfg, sweep_params):
    """Update configuration with wandb sweep parameters."""
    
    # Update horizon parameters (single parameter sets both x and y)
    if 'horizon' in sweep_params:
        horizon = sweep_params['horizon']
        cfg.policy.actor.backbone.x_horizon = horizon
        cfg.policy.actor.backbone.y_horizon = horizon
        # Also update dataset horizon to match
        cfg.dataset.horizon = horizon
        
    # Update training parameters
    if 'num_epochs' in sweep_params:
        cfg.training.num_epochs = sweep_params['num_epochs']
        
    # Update CLoCAnalyzer horizon if enabled
    if hasattr(cfg, 'cloc_analyzer') and cfg.cloc_analyzer.get('enabled', False):
        if 'horizon' in sweep_params:
            cfg.cloc_analyzer.data.horizon = sweep_params['horizon']
    
    return cfg


def main():
    """
    Main training function for wandb sweep.
    """
    
    # Initialize wandb (will be configured by sweep agent)
    wandb.init()
    
    # Get sweep parameters from wandb
    sweep_params = dict(wandb.config)
    
    print(f"Starting training with sweep parameters: {sweep_params}")
    
    # Load base configuration
    config_path = Path(__file__).parent / 'diffusion_policy' / 'config_files' / 'legged_gym_diffuse_sweep.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    cfg = OmegaConf.load(config_path)
    
    # Update config with sweep parameters
    cfg = update_config_with_sweep_params(cfg, sweep_params)
    
    # Resolve any interpolations (like in train.py)
    OmegaConf.resolve(cfg)
    
    # Set experiment name based on sweep parameters
    horizon = cfg.policy.actor.backbone.x_horizon
    exp_name = f"sweep_h{horizon}_e{cfg.training.num_epochs}"
    cfg.exp_name = exp_name
    
    # Set output directory with timestamp (like in train.py)
    cfg.output_dir = os.path.join(
        cfg.output_dir, 
        time.strftime("%B-%d-%H-%M-%S", time.localtime()) + "-" + exp_name
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Update wandb run name
    wandb.run.name = exp_name
    
    print(f"Experiment name: {exp_name}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"horizon (x=y): {horizon}")
    print(f"num_epochs: {cfg.training.num_epochs}")
    
    # Initialize trainer (like in train.py)
    cls = hydra.utils.get_class(cfg._target_)
    trainer: BaseTrainer = cls(cfg)
    
    # Create a simple args object for the trainer (like in train.py)
    class Args:
        def __init__(self):
            self.cfg = 'legged_gym_diffuse_sweep.yaml'
            self.exp_name = exp_name
    
    args = Args()
    
    # Start training (like in train.py)
    trainer.train(args)


if __name__ == '__main__':
    main()