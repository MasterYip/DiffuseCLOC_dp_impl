"""
Runner for Legged Gym environments with DiffuseCLoC policies.
Handles observation history management and policy execution.
"""

import os
import numpy as np
import torch
import tqdm
from typing import Dict, Optional

from diffusion_policy.env.legged_gym_env import LeggedGymEnv
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class LeggedGymRunner(BaseLowdimRunner):
    """
    Evaluation runner for legged locomotion with diffusion policies.
    
    Features:
    - Parallel environment execution
    - Observation history management
    - Episode-level metric tracking
    """

    def __init__(
        self,
        output_dir: str,
        task_name: str = "g1_flat",
        n_envs: int = 16,
        max_steps: int = 1000,
        n_obs_steps: int = 4,
        headless: bool = True,
        device: Optional[str] = None,
        fps: int = 50,
        tqdm_interval_sec: float = 5.0,
        **kwargs
    ):
        """
        Initialize legged gym runner.
        
        Args:
            output_dir: Output directory for results
            task_name: Legged gym task name (e.g., 'g1_flat', 'anymal_c_rough')
            n_envs: Number of parallel environments
            max_steps: Maximum steps per evaluation
            n_obs_steps: History length for observations (e.g., 4 for DiffuseCLoC)
            headless: Disable visualization if True
            device: Device for policy inference (uses policy device if None)
            fps: Frames per second for visualization
            tqdm_interval_sec: Progress bar update interval
        """
        super().__init__(output_dir)

        self.task_name = task_name
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.headless = headless
        self.device = device
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec

        # Environment created lazily in run() to avoid issues during training
        self.env = None

    def run(self, policy) -> Dict:
        """
        Run policy evaluation in legged gym environment with trajectory analysis.
        
        Args:
            policy: Policy with act(obs_dict) method that returns (actions, states)
                   obs_dict: {'obs': (B, n_obs_steps, obs_dim)}
                   returns: actions (B, horizon, action_dim), states (B, horizon, obs_dim)
            
        Returns:
            results: {
                'episode_rewards': List of total rewards per episode,
                'episode_lengths': List of episode lengths,
                'mean_episode_reward': Average reward,
                'mean_episode_length': Average length,
                'num_episodes': Total episodes completed,
                'analyzer_summary': Diagnostic summary from trajectory analyzer
            }
        """
        # Import analyzer
        from diffusion_policy.utils.cloc_analyzer import CLoCAnalyzer, diagnose_training_issues, print_training_recommendations
        
        # Create environment if needed
        if self.env is None:
            print(f"Creating environment: {self.task_name} with {self.n_envs} envs")
            self.env = LeggedGymEnv(
                task_name=self.task_name,
                num_envs=self.n_envs,
                headless=self.headless
            )

        # Device handling
        if self.device is None:
            device = policy.device
        else:
            device = torch.device(self.device)

        # Initialize trajectory analyzer
        analyzer = CLoCAnalyzer(
            n_obs_steps=self.n_obs_steps,
            horizon=20,  # Assuming DiffuseCLoC horizon
            history_length=500,
            update_interval=5,  # Update visualization every 5 steps
            save_plots=True,
            output_dir=os.path.join(self.output_dir, "trajectory_analysis")
        )

        print("Starting evaluation with real-time trajectory analysis...")
        print("Visualization will show: obs history, action/state predictions, velocity tracking")
        
        # Reset environment
        obs, info = self.env.reset()

        # Initialize observation history: (n_envs, n_obs_steps, obs_dim)
        obs_history = obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)

        # Metrics tracking
        episode_rewards = []
        episode_lengths = []
        current_rewards = torch.zeros(self.n_envs, device=device)
        current_lengths = torch.zeros(self.n_envs, device=device, dtype=torch.long)

        # Progress bar
        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Eval {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec
        )

        # Evaluation loop
        for step_idx in range(self.max_steps):
            # Prepare observation dict for policy
            obs_dict = {"obs": obs_history.to(device)}

            # Time policy inference
            inference_start = torch.cuda.Event(enable_timing=True)
            inference_end = torch.cuda.Event(enable_timing=True)
            
            inference_start.record()
            
            # Get action from policy
            with torch.no_grad():
                # DiffuseCLoC returns (action_traj, state_traj)
                # action_traj: (B, horizon, action_dim)
                # We use the first action
                action_traj, state_traj = policy.act(obs_dict["obs"])
                actions = action_traj[:, 0, :]  # (B, action_dim)

            inference_end.record()
            torch.cuda.synchronize()
            inference_time = inference_start.elapsed_time(inference_end) / 1000.0  # Convert to seconds

            # Step environment
            next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

            # Extract environment info for first environment
            env_info = {}
            if isinstance(infos, dict):
                env_info = infos
            elif hasattr(infos, '__len__') and len(infos) > 0:
                env_info = infos[0] if isinstance(infos[0], dict) else {}
                
            # Get command from environment (velocity commands for locomotion)
            if hasattr(self.env, 'get_commands'):
                try:
                    commands = self.env.get_commands()
                    if commands is not None:
                        env_info['commands'] = commands[0].cpu().numpy()  # First env
                except:
                    env_info['commands'] = np.zeros(3)
            else:
                env_info['commands'] = np.zeros(3)

            # Analyze trajectory with analyzer
            analyzer.analyze_step(
                step_idx=step_idx,
                obs_history=obs_history,
                action_traj=action_traj,
                state_traj=state_traj,
                executed_action=actions,
                reward=rewards[0].item(),  # First environment reward
                env_info=env_info,
                inference_time=inference_time
            )

            # Update observation history (FIFO)
            obs_history = torch.cat([obs_history[:, 1:, :], next_obs.unsqueeze(1)], dim=1)

            # Update metrics
            current_rewards += rewards.to(device)
            current_lengths += 1

            # Handle episode terminations
            if dones.any():
                done_indices = torch.where(dones)[0]

                for idx in done_indices:
                    # Record episode metrics
                    episode_rewards.append(current_rewards[idx].item())
                    episode_lengths.append(current_lengths[idx].item())

                    # Reset counters
                    current_rewards[idx] = 0
                    current_lengths[idx] = 0

                    # Notify analyzer of episode reset (for first env)
                    if idx == 0:
                        analyzer.episode_reset()

                    # Print episode summary
                    if len(episode_rewards) % 10 == 0:
                        print(f"\nEpisodes: {len(episode_rewards)}, "
                              f"Last reward: {episode_rewards[-1]:.2f}, "
                              f"Mean reward: {np.mean(episode_rewards[-10:]):.2f}")

            pbar.update(1)
            
            # Check for training issues periodically
            if step_idx > 0 and step_idx % 200 == 0:
                issues = diagnose_training_issues(analyzer)
                if issues:
                    print(f"\n⚠️  DETECTED TRAINING ISSUES AT STEP {step_idx}:")
                    for issue_type, description in issues.items():
                        print(f"  - {issue_type}: {description}")
                    print("  See full recommendations below.\n")

        pbar.close()

        # Add unfinished episodes
        for i in range(self.n_envs):
            if current_lengths[i] > 0:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())

        # Get final diagnostic summary
        analyzer_summary = analyzer.get_diagnostic_summary()
        final_issues = diagnose_training_issues(analyzer)

        # Aggregate results
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "analyzer_summary": analyzer_summary
        }

        print("\n" + "="*70)
        print(f"EVALUATION RESULTS for {self.task_name}:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
        print(f"  Mean Length: {results['mean_episode_length']:.1f}")
        print("\nTRAJECTORY ANALYSIS SUMMARY:")
        
        if 'inference_time' in analyzer_summary:
            inf_time = analyzer_summary['inference_time']
            print(f"  Inference Time: {inf_time['mean']*1000:.1f}ms (±{inf_time['std']*1000:.1f}ms)")
            
        if 'action_magnitude' in analyzer_summary:
            act_mag = analyzer_summary['action_magnitude']
            print(f"  Action Magnitude: {act_mag['mean']:.3f} (±{act_mag['std']:.3f})")
            
        if 'velocity_tracking_error' in analyzer_summary:
            vel_err = analyzer_summary['velocity_tracking_error']
            print(f"  Velocity Tracking Error: {vel_err['mean']:.3f}m/s (±{vel_err['std']:.3f})")

        # Print any detected issues
        if final_issues:
            print(f"\n🔍 TRAINING ISSUE DETECTION:")
            for issue_type, description in final_issues.items():
                print(f"  ❌ {issue_type}: {description}")
            
            print(f"\n📋 RECOMMENDATIONS:")
            print_training_recommendations()
        else:
            print(f"\n✅ No major training issues detected!")

        print("="*70 + "\n")

        # Clean up analyzer
        analyzer.close()

        return results
