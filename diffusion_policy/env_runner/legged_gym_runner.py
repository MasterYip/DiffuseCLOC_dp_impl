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
        Run policy evaluation in legged gym environment.
        
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
                'num_episodes': Total episodes completed
            }
        """
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

        # Reset environment
        obs, _ = self.env.reset()

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

            # Get action from policy
            with torch.no_grad():
                # DiffuseCLoC returns (action_traj, state_traj)
                # action_traj: (B, horizon, action_dim)
                # We use the first action
                action_traj, state_traj = policy.act(obs_dict)
                actions = action_traj[:, 0, :]  # (B, action_dim)

            # Step environment
            next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

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

                    # Print episode summary
                    if len(episode_rewards) % 10 == 0:
                        print(f"Episodes: {len(episode_rewards)}, "
                              f"Last reward: {episode_rewards[-1]:.2f}, "
                              f"Mean reward: {np.mean(episode_rewards[-10:]):.2f}")

            pbar.update(1)

        pbar.close()

        # Add unfinished episodes
        for i in range(self.n_envs):
            if current_lengths[i] > 0:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())

        # Aggregate results
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
        }

        print("\n" + "="*50)
        print(f"Evaluation Results for {self.task_name}:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
        print(f"  Mean Length: {results['mean_episode_length']:.1f}")
        print("="*50 + "\n")

        return results
