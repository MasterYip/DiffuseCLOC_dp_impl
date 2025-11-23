"""
Runner for Legged Gym environments with DiffuseCLoC policies.
Handles observation history management and policy execution.
"""

import os
import numpy as np
import torch
import tqdm
import threading
import queue
from typing import Dict, Optional

from diffusion_policy.env.legged_gym_env import LeggedGymEnv
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class AnalysisWorker(threading.Thread):
    """Worker thread for trajectory analysis to avoid blocking simulation."""
    
    def __init__(self, analyzer, analyzer_cfg):
        super().__init__(daemon=True)
        self.analyzer = analyzer
        self.analyzer_cfg = analyzer_cfg
        self.task_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory buildup
        self.running = True
        
    def run(self):
        """Process analysis tasks from queue."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                    
                task_type = task['type']
                
                if task_type == 'analyze_step':
                    self.analyzer.analyze_step(**task['kwargs'])
                elif task_type == 'episode_reset':
                    self.analyzer.episode_reset()
                elif task_type == 'diagnose_issues':
                    # Diagnostic checks can be time-consuming, run in thread
                    issues = self._diagnose_training_issues()
                    if issues:
                        print(f"\n⚠️  DETECTED TRAINING ISSUES AT STEP {task['step_idx']}:")
                        for issue_type, description in issues.items():
                            print(f"  - {issue_type}: {description}")
                        
                        issue_cfg = self.analyzer_cfg.get('issue_detection', {})
                        if issue_cfg.get('print_recommendations', True):
                            print("  See full recommendations below.\n")
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Warning: Analysis worker error: {e}")
                continue
    
    def _diagnose_training_issues(self):
        """Run training issue diagnosis."""
        try:
            from diffusion_policy.utils.cloc_analyzer import diagnose_training_issues
            return diagnose_training_issues(self.analyzer)
        except Exception as e:
            print(f"Warning: Issue diagnosis failed: {e}")
            return {}
    
    def add_analysis_task(self, task):
        """Add analysis task to queue (non-blocking)."""
        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            # Drop task if queue is full to prevent blocking simulation
            pass
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        try:
            self.task_queue.put_nowait(None)  # Shutdown signal
        except queue.Full:
            pass


class LeggedGymRunner(BaseLowdimRunner):
    """
    Evaluation runner for legged locomotion with diffusion policies.
    
    Features:
    - Parallel environment execution
    - Observation history management
    - Episode-level metric tracking
    - Non-blocking trajectory analysis
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

    def run(self, bc_agent, cfg=None) -> Dict:
        """
        Run policy evaluation in legged gym environment with configurable trajectory analysis.
        
        Args:
            bc_agent: BCAgent with proper normalization and act() method
            cfg: Hydra config object containing cloc_analyzer settings
                   
        Returns:
            results: Evaluation results with analyzer summary
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

        # Device handling - use bc_agent's device
        if self.device is None:
            device = bc_agent.device
        else:
            device = torch.device(self.device)

        # Initialize trajectory analyzer with config parameters
        analyzer = None
        analysis_worker = None
        analyzer_cfg = cfg.get('cloc_analyzer', {}) if cfg else {}
        
        if analyzer_cfg.get('enabled', False):
            # Extract configuration parameters
            viz_cfg = analyzer_cfg.get('visualization', {})
            data_cfg = analyzer_cfg.get('data', {})
            output_cfg = analyzer_cfg.get('output', {})
            selected_dims = analyzer_cfg.get('selected_dims', {})
            
            analyzer = CLoCAnalyzer(
                n_obs_steps=data_cfg.get('n_obs_steps', self.n_obs_steps),
                horizon=data_cfg.get('horizon', 20),
                history_length=data_cfg.get('history_length', 500),
                update_interval=viz_cfg.get('update_interval', 10),
                save_plots=output_cfg.get('save_frequency', 'never') != 'never',
                output_dir=os.path.join(self.output_dir, output_cfg.get('output_dir', 'trajectory_analysis')),
                selected_dims=selected_dims if selected_dims else None
            )
            
            # Start analysis worker thread
            analysis_worker = AnalysisWorker(analyzer, analyzer_cfg)
            analysis_worker.start()

            print("Starting evaluation with configurable trajectory analysis (threaded)...")
            print(f"Analysis config: enabled={analyzer_cfg.get('enabled')}, "
                  f"env_idx={analyzer_cfg.get('analyze_env_idx', 0)}, "
                  f"update_interval={viz_cfg.get('update_interval', 10)}")
        else:
            print("Starting evaluation without trajectory analysis (disabled in config)")
        
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

        # Get analysis environment index
        analyze_env_idx = analyzer_cfg.get('analyze_env_idx', 0) if analyzer_cfg else 0
        analyze_all_envs = analyzer_cfg.get('analyze_all_envs', False) if analyzer_cfg else False

        # Evaluation loop
        for step_idx in range(self.max_steps):
            # Prepare observation dict for BCAgent (with proper normalization)
            obs_dict = {"obs": obs_history.to(device)}

            # Time policy inference if analyzer is enabled
            if analyzer:
                inference_start = torch.cuda.Event(enable_timing=True)
                inference_end = torch.cuda.Event(enable_timing=True)
                inference_start.record()
            
            # Get action from BCAgent (includes normalization/unnormalization)
            with torch.no_grad():
                # BCAgent.act() handles normalization internally and returns unnormalized actions
                result = bc_agent.act(obs_dict)
                
                # Handle different return types from BCAgent
                if isinstance(result, tuple) and len(result) >= 2:
                    # Joint diffusion case: (action_traj, state_traj, ...)
                    action_traj, state_traj = result[0], result[1]
                    actions = action_traj[:, 0, :] if action_traj.dim() == 3 else action_traj
                else:
                    # Standard diffusion case: just actions
                    actions = result[:, 0, :] if result.dim() == 3 else result
                    action_traj = result
                    state_traj = None

            if analyzer:
                inference_end.record()
                torch.cuda.synchronize()
                inference_time = inference_start.elapsed_time(inference_end) / 1000.0
            else:
                inference_time = 0.0

            # Step environment
            next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

            # Trajectory analysis (if enabled) - submit to worker thread
            if analysis_worker and state_traj is not None:
                # Determine which environments to analyze
                env_indices = list(range(self.n_envs)) if analyze_all_envs else [analyze_env_idx]
                
                for env_idx in env_indices:
                    if env_idx >= self.n_envs:
                        continue
                        
                    # Extract environment info
                    env_info = {}
                    if isinstance(infos, dict):
                        env_info = infos
                    elif hasattr(infos, '__len__') and len(infos) > env_idx:
                        env_info = infos[env_idx] if isinstance(infos[env_idx], dict) else {}
                        
                    # Extract robot-specific information from config
                    robot_cfg = analyzer_cfg.get('robot_config', {})
                    cmd_vel_dims = robot_cfg.get('cmd_vel_dims', [0, 1, 2])
                    
                    # Get command from environment or extract from observation
                    if hasattr(self.env, 'get_commands'):
                        try:
                            commands = self.env.get_commands()
                            if commands is not None:
                                env_info['commands'] = commands[env_idx].cpu().numpy()
                        except:
                            # Fallback: extract from observation if possible
                            current_obs = next_obs[env_idx].cpu().numpy()
                            if len(current_obs) >= max(cmd_vel_dims) + 1:
                                env_info['commands'] = current_obs[cmd_vel_dims]
                            else:
                                env_info['commands'] = np.zeros(3)
                    else:
                        env_info['commands'] = np.zeros(3)

                    # Submit analysis task to worker thread (non-blocking)
                    analysis_task = {
                        'type': 'analyze_step',
                        'kwargs': {
                            'step_idx': step_idx,
                            'obs_history': obs_history[env_idx:env_idx+1].clone().cpu(),  # Clone and move to CPU
                            'action_traj': action_traj[env_idx:env_idx+1].clone().cpu(),
                            'state_traj': state_traj[env_idx:env_idx+1].clone().cpu() if state_traj is not None else None,
                            'executed_action': actions[env_idx:env_idx+1].clone().cpu(),
                            'reward': rewards[env_idx].item(),
                            'env_info': env_info.copy(),  # Copy to avoid reference issues
                            'inference_time': inference_time
                        }
                    }
                    analysis_worker.add_analysis_task(analysis_task)
                    
                    # Only analyze one environment unless analyze_all_envs is True
                    if not analyze_all_envs:
                        break

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

                    # Notify analyzer of episode reset (for analyzed environment)
                    if analysis_worker and (analyze_all_envs or idx == analyze_env_idx):
                        reset_task = {'type': 'episode_reset'}
                        analysis_worker.add_analysis_task(reset_task)

                    # Print episode summary
                    if len(episode_rewards) % 10 == 0:
                        print(f"\nEpisodes: {len(episode_rewards)}, "
                              f"Last reward: {episode_rewards[-1]:.2f}, "
                              f"Mean reward: {np.mean(episode_rewards[-10:]):.2f}")

            pbar.update(1)
            
            # Check for training issues periodically (if analyzer enabled) - submit to worker thread
            if analysis_worker:
                issue_cfg = analyzer_cfg.get('issue_detection', {})
                if (issue_cfg.get('enabled', True) and 
                    step_idx > 0 and 
                    step_idx % issue_cfg.get('check_interval', 200) == 0):
                    
                    diagnosis_task = {
                        'type': 'diagnose_issues',
                        'step_idx': step_idx
                    }
                    analysis_worker.add_analysis_task(diagnosis_task)

        pbar.close()

        # Wait for analysis worker to complete pending tasks
        if analysis_worker:
            print("Waiting for trajectory analysis to complete...")
            analysis_worker.task_queue.join()  # Wait for all tasks to be processed
            analysis_worker.stop()
            analysis_worker.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish

        # Add unfinished episodes
        for i in range(self.n_envs):
            if current_lengths[i] > 0:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())

        # Get final diagnostic summary and issues (if analyzer enabled)
        analyzer_summary = {}
        final_issues = {}
        
        if analyzer:
            analyzer_summary = analyzer.get_diagnostic_summary()
            try:
                final_issues = diagnose_training_issues(analyzer)
            except Exception as e:
                print(f"Warning: Final diagnosis failed: {e}")
                final_issues = {}

        # Aggregate results
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "analyzer_summary": analyzer_summary,
            "analyzer_enabled": analyzer is not None
        }

        print("\n" + "="*70)
        print(f"EVALUATION RESULTS for {self.task_name}:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
        print(f"  Mean Length: {results['mean_episode_length']:.1f}")
        
        if analyzer:
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
                
                issue_cfg = analyzer_cfg.get('issue_detection', {})
                if issue_cfg.get('print_recommendations', True):
                    print(f"\n📋 RECOMMENDATIONS:")
                    print_training_recommendations()
            else:
                print(f"\n✅ No major training issues detected!")
        else:
            print("\n(Trajectory analysis was disabled)")

        print("="*70 + "\n")

        # Clean up analyzer
        if analyzer:
            analyzer.close()

        return results
