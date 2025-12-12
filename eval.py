"""
Evaluation script for DiffuseCLoC policies in Legged Gym or Isaac Lab environments.

Usage:
    # Legged Gym
    python eval.py --checkpoint outputs/latest.ckpt -o eval_output --env_type legged_gym --task g1_flat --num_envs 16
    python eval.py --checkpoint outputs/latest.ckpt --config joint_diffuse.yaml -o eval_output --env_type legged_gym --task g1_flat
    
    # Isaac Lab
    python eval.py --checkpoint outputs/latest.ckpt -o eval_output --env_type isaac_lab --task Isaac-TextOp-Diffusion-G1-v0 --num_envs 16
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
import hydra

from diffusion_policy import DIFFUSION_POLICY_ROOT
from diffusion_policy.trainer.base_trainer import BaseTrainer


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint file')
@click.option('--config', default="legged_gym_diffuse.yaml", help='Config file to load instead of using checkpoint config (e.g., joint_diffuse.yaml)')
@click.option('-o', '--output_dir', required=True, help='Output directory for results')
@click.option('-d', '--device', default='cuda:0', help='Device for inference')
@click.option('--env_type', type=click.Choice(['legged_gym', 'isaac_lab']), default='legged_gym', help='Environment type (legged_gym or isaac_lab)')
@click.option('-t', '--task', default='g1_flat', help='Task name (e.g., g1_flat for legged_gym, Isaac-TextOp-Diffusion-G1-v0 for isaac_lab)')
@click.option('--num_envs', default=16, help='Number of parallel environments')
@click.option('--max_steps', default=1000, help='Maximum steps per evaluation')
@click.option('--n_obs_steps', default=4, help='Observation history length')
@click.option('--headless', is_flag=True, default=False, help='Run headless (no visualization)')
def main(checkpoint, config, output_dir, device, env_type, task, num_envs, max_steps, n_obs_steps, headless):
    """Evaluate a trained DiffuseCLoC policy in Legged Gym or Isaac Lab."""
    
    # Create output directory
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)

    # Load configuration: from file if specified, otherwise from payload
    if config is not None:
        print(f"Loading configuration from file: {config}")
        config_path = os.path.join(DIFFUSION_POLICY_ROOT, './config_files', config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        print(f"Using config from file: {config}")
    else:
        print("Using configuration from checkpoint")
        cfg = payload['cfg']
    
    print("\nLoaded configuration.")
    # print(OmegaConf.to_yaml(cfg))

    # Initialize trainer (same as train.py)
    cls = hydra.utils.get_class(cfg._target_)
    trainer: BaseTrainer = cls(cfg)
    
    # Load checkpoint into trainer (this will also reconstruct normalizer if it exists in checkpoint)
    trainer.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # # Load normalizer from dataset if not already loaded from checkpoint
    # print("\nEnsuring normalizer is loaded...")
    # trainer.load_for_eval()
    
    # Get bc_agent from trainer
    bc_agent = trainer.agent
    bc_agent.to(device)
    bc_agent.eval()
    # print("normalizer params:")
    # d = bc_agent.normalizer.get_input_stats()
    
    # # print d in beautiful way with tensor contents
    # for key, param_dict in d.items():
    #     print(f"{key}:")
    #     for subkey, tensor in param_dict.items():
    #         print(f"  {subkey}: {tensor.tolist()}")
    
    print(f"BCAgent loaded and moved to {device}")
    print(f"Policy type: {type(bc_agent.actor).__name__}")

    # Create environment runner based on env_type
    if env_type == 'isaac_lab':
        from diffusion_policy.env_runner.isaac_lab_runner import IsaacLabRunner
        env_runner = IsaacLabRunner(
            output_dir=output_dir,
            task_name=task,
            n_envs=num_envs,
            max_steps=max_steps,
            n_obs_steps=n_obs_steps,
            headless=headless,
            device=device
        )
    else:  # legged_gym
        from diffusion_policy.env_runner.legged_gym_runner import LeggedGymRunner
        env_runner = LeggedGymRunner(
            output_dir=output_dir,
            task_name=task,
            n_envs=num_envs,
            max_steps=max_steps,
            n_obs_steps=n_obs_steps,
            headless=headless,
            device=device,
            realtime_mode=True
        )

    # Run evaluation with BCAgent
    print(f"\nStarting evaluation...")
    print(f"  Environment Type: {env_type}")
    print(f"  Task: {task}")
    print(f"  Num envs: {num_envs}")
    print(f"  Max steps: {max_steps}")
    print(f"  Headless: {headless}")
    print(f"  Normalization: {'Enabled' if hasattr(bc_agent, 'normalizer') else 'Disabled'}")
    if cfg and cfg.get('cloc_analyzer', {}).get('enabled', False):
        print(f"  CLoCAnalyzer: ENABLED")
    else:
        print(f"  CLoCAnalyzer: DISABLED")

    results = env_runner.run(bc_agent, cfg)

    # Save results to JSON
    output_file = os.path.join(output_dir, 'eval_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nResults saved to: {output_file}")

    # Save configuration used for evaluation
    config_output_file = os.path.join(output_dir, 'eval_config.yaml')
    with open(config_output_file, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Config saved to: {config_output_file}")

    # Save summary
    summary_file = os.path.join(output_dir, 'eval_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Config: {config if config else 'from checkpoint'}\n")
        f.write(f"Environment Type: {env_type}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Num Envs: {num_envs}\n")
        f.write(f"Max Steps: {max_steps}\n")
        f.write(f"CLoCAnalyzer: {'ENABLED' if cfg and cfg.get('cloc_analyzer', {}).get('enabled', False) else 'DISABLED'}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Episodes: {results['num_episodes']}\n")
        f.write(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}\n")
        f.write(f"  Mean Length: {results['mean_episode_length']:.1f}\n")
        
        # Add analyzer summary if available
        if results.get('analyzer_enabled', False) and results.get('analyzer_summary'):
            f.write(f"\nTrajectory Analysis:\n")
            analyzer_summary = results['analyzer_summary']
            if 'inference_time' in analyzer_summary:
                inf_time = analyzer_summary['inference_time']
                f.write(f"  Inference Time: {inf_time['mean']*1000:.1f}ms ± {inf_time['std']*1000:.1f}ms\n")
            if 'action_magnitude' in analyzer_summary:
                act_mag = analyzer_summary['action_magnitude']
                f.write(f"  Action Magnitude: {act_mag['mean']:.3f} ± {act_mag['std']:.3f}\n")
            if 'velocity_tracking_error' in analyzer_summary:
                vel_err = analyzer_summary['velocity_tracking_error']
                f.write(f"  Velocity Tracking Error: {vel_err['mean']:.3f}m/s ± {vel_err['std']:.3f}m/s\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
