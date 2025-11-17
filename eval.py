"""
Evaluation script for DiffuseCLoC policies in Legged Gym environments.

Usage:
    python eval.py --checkpoint outputs/latest.ckpt -o eval_output --task g1_flat --num_envs 16
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

try:
    import isaacgym
except ImportError:
    print("Warning: Isaac Gym not installed. Evaluation may fail.")

import os
import pathlib
import click
import torch
import json
from omegaconf import OmegaConf

from diffusion_policy.env_runner.legged_gym_runner import LeggedGymRunner


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint file')
@click.option('-o', '--output_dir', required=True, help='Output directory for results')
@click.option('-d', '--device', default='cuda:0', help='Device for inference')
@click.option('-t', '--task', default='g1_flat', help='Legged gym task name')
@click.option('--num_envs', default=16, help='Number of parallel environments')
@click.option('--max_steps', default=1000, help='Maximum steps per evaluation')
@click.option('--n_obs_steps', default=4, help='Observation history length')
@click.option('--headless', is_flag=True, default=False, help='Run headless (no visualization)')
def main(checkpoint, output_dir, device, task, num_envs, max_steps, n_obs_steps, headless):
    """Evaluate a trained DiffuseCLoC policy in Legged Gym."""
    
    # Create output directory
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location='cpu')
    
    # Extract config and policy
    if 'cfg' in payload:
        cfg = payload['cfg']
        print("\nCheckpoint configuration:")
        print(OmegaConf.to_yaml(cfg))
    else:
        print("Warning: No config found in checkpoint")
        cfg = None

    # Load policy from checkpoint
    # FIXME: this the temporary fix for loading policy weights
    print("Available keys in checkpoint:", payload['state_dicts'].keys() if 'state_dicts' in payload else payload.keys())
    
    if 'state_dicts' in payload and 'agent' in payload['state_dicts']:
        # Standard checkpoint format
        full_state_dict = payload['state_dicts']['agent']
        
        # Extract only the actor (policy) weights, removing the 'actor.' prefix
        policy_state_dict = {}
        normalizer_state_dict = {}
        
        for key, value in full_state_dict.items():
            if key.startswith('actor.'):
                # Remove 'actor.' prefix to match the policy structure
                new_key = key[6:]  # Remove 'actor.' (6 characters)
                if not new_key.startswith('normalizer.'):
                    policy_state_dict[new_key] = value
                else:
                    # Store normalizer separately
                    normalizer_state_dict[new_key] = value
            elif key.startswith('normalizer.'):
                normalizer_state_dict[key] = value
            elif not key.startswith('_dummy_variable'):
                # Direct policy weights without prefix
                policy_state_dict[key] = value
                
        print(f"Extracted {len(policy_state_dict)} policy parameters")
        print(f"Extracted {len(normalizer_state_dict)} normalizer parameters")
        
    elif 'model' in payload:
        # Direct model state dict
        policy_state_dict = payload['model']
    else:
        raise ValueError("Cannot find model state dict in checkpoint")

    # Create policy from config
    if cfg is not None:
        from hydra.utils import instantiate
        policy = instantiate(cfg.policy.actor)
    else:
        # Fallback: create default DiffuseCLoC
        from diffusion_policy.modules.diffuse_cloc import DiffuseCLoC
        from diffusion_policy.backbone.transformer_codiffuse import Transformer
        
        print("Creating default DiffuseCLoC policy...")
        backbone = Transformer(
            x_horizon=20,
            y_horizon=20,
            x_input_dim=384,
            y_input_dim=29,
            x_output_dim=384,
            y_output_dim=29,
            n_emb=256,
            n_head=4,
            n_layer=2,
            causal_attn=True,
        )
        policy = DiffuseCLoC(
            backbone=backbone,
            denoising_steps=20,
            n_past_steps=4,
            state_emphasis='random_emph_symm',
        )

    # Load policy weights
    policy.load_state_dict(policy_state_dict)
    policy.to(device)
    policy.eval()

    print(f"\nPolicy loaded on {device}")
    print(f"Policy type: {type(policy).__name__}")

    # Create environment runner
    env_runner = LeggedGymRunner(
        output_dir=output_dir,
        task_name=task,
        n_envs=num_envs,
        max_steps=max_steps,
        n_obs_steps=n_obs_steps,
        headless=headless,
        device=device,
    )

    # Run evaluation
    print(f"\nStarting evaluation...")
    print(f"  Task: {task}")
    print(f"  Num envs: {num_envs}")
    print(f"  Max steps: {max_steps}")
    print(f"  Headless: {headless}")
    
    results = env_runner.run(policy)

    # Save results to JSON
    output_file = os.path.join(output_dir, 'eval_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nResults saved to: {output_file}")

    # Save summary
    summary_file = os.path.join(output_dir, 'eval_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Num Envs: {num_envs}\n")
        f.write(f"Max Steps: {max_steps}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Episodes: {results['num_episodes']}\n")
        f.write(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}\n")
        f.write(f"  Mean Length: {results['mean_episode_length']:.1f}\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
