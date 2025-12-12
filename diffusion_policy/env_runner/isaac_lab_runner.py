"""
Isaac Lab runner for DiffuseCLOC policies.
Handles observation history management, state normalization, and policy execution in Isaac Lab.
"""

import os
import sys
import time
import numpy as np
import torch
import tqdm
from typing import Dict, Optional
from pathlib import Path

# Add TextOpTracker to path
TEXTOP_TRACKER_PATH = str(Path(__file__).parent.parent.parent.parent.parent / "TextOpTracker")
if TEXTOP_TRACKER_PATH not in sys.path:
    sys.path.append(TEXTOP_TRACKER_PATH)

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class IsaacLabRunner(BaseLowdimRunner):
    """
    Evaluation runner for DiffuseCLOC policies in Isaac Lab environments.
    
    Features:
    - Isaac Lab environment management
    - Observation history with proper normalization
    - Episode-level metric tracking
    - Matches data collection format from g1_offline_dataset
    """

    def __init__(
        self,
        output_dir: str,
        task_name: str = "Isaac-TextOp-Diffusion-G1-v0",
        n_envs: int = 16,
        max_steps: int = 1000,
        n_obs_steps: int = 4,
        headless: bool = True,
        device: Optional[str] = None,
        fps: int = 50,
        tqdm_interval_sec: float = 5.0,
        realtime_mode: bool = False,
        **kwargs
    ):
        """
        Initialize Isaac Lab runner.
        
        Args:
            output_dir: Output directory for results
            task_name: Isaac Lab task name (e.g., 'Isaac-TextOp-Diffusion-G1-v0')
            n_envs: Number of parallel environments
            max_steps: Maximum steps per evaluation
            n_obs_steps: History length for observations (e.g., 4 for DiffuseCLOC)
            headless: Disable visualization if True
            device: Device for policy inference (uses policy device if None)
            fps: Frames per second for visualization
            tqdm_interval_sec: Progress bar update interval
            realtime_mode: If True, run in realtime with timing constraints
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
        self.realtime_mode = realtime_mode

        # Environment created lazily in run() 
        self.env = None
        self.simulation_app = None

    def _create_environment(self):
        """Create Isaac Lab environment and simulation app."""
        if self.env is not None:
            return
        
        # Import Isaac Lab after setting up paths
        from isaaclab.app import AppLauncher
        import argparse
        
        # Create app launcher
        app_launcher_args = argparse.Namespace(
            headless=self.headless,
            livestream=False,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            enable_cameras=False,
        )
        app_launcher = AppLauncher(app_launcher_args)
        self.simulation_app = app_launcher.app
        
        # Import gym after Isaac Sim is initialized
        import gymnasium as gym
        
        # Import tasks to register environments
        import textop_tracker.tasks.diffusion  # noqa: F401
        
        print(f"Creating Isaac Lab environment: {self.task_name} with {self.n_envs} envs")
        
        # Create environment
        self.env = gym.make(self.task_name, num_envs=self.n_envs, render_mode=None)
        self.env_unwrapped = self.env.unwrapped

    def _cleanup(self):
        """Clean up environment and simulation."""
        if self.env is not None:
            self.env.close()
            self.env = None
        
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None

    def run(self, bc_agent, cfg=None) -> Dict:
        """
        Run policy evaluation in Isaac Lab environment.
        
        Args:
            bc_agent: BCAgent with proper normalization and act() method
            cfg: Hydra config object (optional, for analyzer settings)
                   
        Returns:
            results: Evaluation results dictionary
        """
        try:
            return self._run_internal(bc_agent, cfg)
        finally:
            self._cleanup()

    def _run_internal(self, bc_agent, cfg) -> Dict:
        """Internal run method with proper cleanup."""
        
        # Create environment
        self._create_environment()

        # Device handling - use bc_agent's device
        if self.device is None:
            device = bc_agent.device
        else:
            device = torch.device(self.device)

        print(f"Running evaluation on device: {device}")
        
        # Reset environment
        obs_dict, info = self.env.reset()
        
        # Extract single observation (already normalized by diffusion_state_observation)
        # Shape: [n_envs, 192] for G1_Dataset format
        obs = obs_dict["policy"]  # [n_envs, 192]
        
        # Initialize observation history: (n_envs, n_obs_steps, obs_dim)
        # Repeat current observation for history
        obs_history = obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)  # [n_envs, n_obs_steps, 192]

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

        # Realtime management variables
        env_dt = 0.02  # 50 Hz control
        if self.realtime_mode:
            print(f"Running in realtime mode (target dt={env_dt:.4f}s)")

        # Evaluation loop
        for step_idx in range(self.max_steps):
            step_start_time = time.time()
            
            # Prepare observation dict for BCAgent
            # obs_history is already on GPU from environment
            obs_dict_policy = {"obs": obs_history.to(device)}

            # Get action from BCAgent (includes normalization if needed)
            with torch.no_grad():
                # BCAgent.act() returns actions for the prediction horizon
                # We only use the first action
                action_dict = bc_agent.act(obs_dict_policy)
                
                # Extract first action from horizon
                # Shape: [n_envs, horizon, action_dim] -> [n_envs, action_dim]
                if "action" in action_dict:
                    actions = action_dict["action"][:, 0, :]  # First timestep
                else:
                    raise ValueError("BCAgent did not return 'action' in action_dict")

            # Step environment
            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = terminated | truncated
            
            # Extract new observation (already normalized)
            obs = obs_dict["policy"]  # [n_envs, 192]
            
            # Update observation history (shift and append)
            obs_history = torch.cat([
                obs_history[:, 1:, :],  # Remove oldest
                obs.unsqueeze(1)  # Add newest
            ], dim=1)  # [n_envs, n_obs_steps, 192]
            
            # Update metrics
            current_rewards += rewards
            current_lengths += 1
            
            # Handle episode terminations
            if dones.any():
                done_indices = torch.where(dones)[0]
                
                for env_idx in done_indices:
                    # Record episode
                    episode_rewards.append(current_rewards[env_idx].item())
                    episode_lengths.append(current_lengths[env_idx].item())
                    
                    # Reset metrics
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            
            # Update progress
            pbar.update(1)
            
            # Realtime control
            if self.realtime_mode:
                step_elapsed = time.time() - step_start_time
                sleep_time = max(0, env_dt - step_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Check if simulation is still running
            if not self.simulation_app.is_running():
                print("\nSimulation stopped by user")
                break

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
            "analyzer_enabled": False,  # Not implemented yet for Isaac Lab
        }

        print("\n" + "="*70)
        print(f"EVALUATION RESULTS for {self.task_name}:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
        print(f"  Mean Length: {results['mean_episode_length']:.1f}")
        print("="*70 + "\n")

        return results
