"""
Legged Gym Environment Wrapper for DiffuseCLoC
Provides standardized interface for Isaac Gym legged locomotion tasks.
"""

import sys
import isaacgym
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_default_args, task_registry

from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch


class LeggedGymEnv:
    """
    Wrapper for legged gym environments with diffusion policy interface.
    
    Key features:
    - Standardized reset/step interface
    - Support for parallel environments
    - Device-agnostic observations
    """

    def __init__(
        self,
        task_name: str = "anymal_c_flat",
        num_envs: Optional[int] = None,
        headless: bool = True,
        sim_device: str = "cuda:0",
        rl_device: str = "cuda:0",
        physics_engine: int = gymapi.SIM_PHYSX,
        seed: Optional[int] = None,
        args=None,
        **kwargs
    ):
        """
        Initialize legged gym environment.
        
        Args:
            task_name: Registered task name (e.g., 'g1_flat', 'anymal_c_rough')
            num_envs: Number of parallel environments
            headless: Disable visualization if True
            sim_device: Simulation device ('cuda:0', 'cpu')
            rl_device: RL computation device
            physics_engine: gymapi.SIM_PHYSX or gymapi.SIM_FLEX
            seed: Random seed for reproducibility
        """
        # Create args from defaults if not provided
        if args is None:
            args = get_default_args()
            args.task = task_name
            args.headless = headless
            args.sim_device_type = sim_device.split(':')[0]
            if ':' in sim_device:
                args.compute_device_id = int(sim_device.split(':')[1])
            args.rl_device = rl_device
            args.physics_engine = physics_engine

            if num_envs is not None:
                args.num_envs = num_envs
            if seed is not None:
                args.seed = seed

        # Create environment from task registry
        self.env, self.env_cfg = task_registry.make_env(name=args.task, args=args)

        # Cache attributes
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.device = self.env.device
        self.task_name = task_name

    def reset(self, env_ids=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset environment(s).
        
        Args:
            env_ids: (Optional) Specific environment IDs to reset, None for all
            
        Returns:
            obs: (num_envs, obs_dim) - Initial observations
            info: Additional reset information
        """
        if env_ids is None:
            return self.env.reset()
        else:
            return self.env.reset(env_ids=env_ids)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step environment with actions.
        
        Args:
            actions: (num_envs, action_dim) - Actions to execute
            
        Returns:
            obs: (num_envs, obs_dim) - New observations
            rewards: (num_envs,) - Rewards
            dones: (num_envs,) - Done flags
            infos: Additional info dict
        """
        # Isaac Gym returns (obs, privileged_obs, rewards, dones, infos)
        # We only need (obs, rewards, dones, infos)
        obs, _, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos

    def get_observations(self) -> torch.Tensor:
        """
        Get current observations without stepping.
        
        Returns:
            obs: (num_envs, obs_dim) - Current observations
        """
        return self.env.get_observations()

    def close(self):
        """Clean up resources (Isaac Gym typically doesn't require explicit cleanup)."""
        pass


# Quick test
if __name__ == "__main__":
    import time
    
    env = LeggedGymEnv(
        task_name="g1_flat",
        num_envs=4,
        headless=False,
        seed=42
    )

    print(f"Environment: {env.task_name}")
    print(f"Num envs: {env.num_envs}")
    print(f"Obs dim: {env.num_obs}")
    print(f"Action dim: {env.num_actions}")

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")

    for i in range(1000):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2 - 1
        obs, rewards, dones, infos = env.step(actions)
        
        if i % 100 == 0:
            print(f"Step {i}, Mean reward: {rewards.mean().item():.3f}")
        time.sleep(0.01)

    env.close()
