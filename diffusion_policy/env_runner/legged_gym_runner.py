"""
Runner for Legged Gym environments with DiffuseCLoC policies.
Handles observation history management and policy execution.
"""

import os
import numpy as np
import torch
import tqdm
import multiprocessing as mp
import queue
import atexit
import time
from typing import Dict, Optional

from diffusion_policy.env.legged_gym_env import LeggedGymEnv
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


def analyzer_worker_process(data_queue, config_dict, output_dir, n_obs_steps):
    """
    Worker process for CLoCAnalyzer to run in parallel with simulation.
    
    Args:
        data_queue: Queue for receiving analysis data
        config_dict: Configuration dictionary for analyzer
        output_dir: Output directory for analysis results
        n_obs_steps: Number of observation steps
    """
    try:
        # Import analyzer inside worker process
        from diffusion_policy.utils.cloc_analyzer import CLoCAnalyzer, diagnose_training_issues, print_training_recommendations
        
        # Extract configuration parameters
        viz_cfg = config_dict.get('visualization', {})
        data_cfg = config_dict.get('data', {})
        output_cfg = config_dict.get('output', {})
        selected_dims = config_dict.get('selected_dims', {})
        
        # Initialize analyzer
        analyzer = CLoCAnalyzer(
            n_obs_steps=data_cfg.get('n_obs_steps', n_obs_steps),
            horizon=data_cfg.get('horizon', 20),
            history_length=data_cfg.get('history_length', 500),
            update_interval=viz_cfg.get('update_interval', 10),
            save_plots=output_cfg.get('save_frequency', 'never') != 'never',
            output_dir=os.path.join(output_dir, output_cfg.get('output_dir', 'trajectory_analysis')),
            selected_dims=selected_dims if selected_dims else None
        )
        
        print("CLoCAnalyzer worker process started")
        
        # Process data from queue
        while True:
            try:
                # Get data with timeout to allow graceful shutdown
                data = data_queue.get(timeout=1.0)
                
                if data is None:  # Shutdown signal
                    break
                
                if data['type'] == 'analyze_step':
                    analyzer.analyze_step(**data['args'])
                elif data['type'] == 'episode_reset':
                    analyzer.episode_reset()
                elif data['type'] == 'diagnose_issues':
                    issues = diagnose_training_issues(analyzer)
                    if issues:
                        print(f"\n⚠️  DETECTED TRAINING ISSUES AT STEP {data['step_idx']}:")
                        for issue_type, description in issues.items():
                            print(f"  - {issue_type}: {description}")
                        
                        if data.get('print_recommendations', True):
                            print("  See full recommendations below.\n")
                elif data['type'] == 'get_summary':
                    # Send summary back through a result queue if provided
                    summary = analyzer.get_diagnostic_summary()
                    final_issues = diagnose_training_issues(analyzer)
                    result_data = {
                        'analyzer_summary': summary,
                        'final_issues': final_issues
                    }
                    if 'result_queue' in data:
                        data['result_queue'].put(result_data)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analyzer worker: {e}")
                continue
        
        # Cleanup
        analyzer.close()
        print("CLoCAnalyzer worker process ended")
        
    except Exception as e:
        print(f"Fatal error in analyzer worker process: {e}")


class LeggedGymRunner(BaseLowdimRunner):
    """
    Evaluation runner for legged locomotion with diffusion policies.
    
    Features:
    - Parallel environment execution
    - Observation history management
    - Episode-level metric tracking
    - Parallel trajectory analysis using multiprocessing
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
        realtime_mode: bool = False,  # Add realtime mode option
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

        # Environment created lazily in run() to avoid issues during training
        self.env = None
        
        # Multiprocessing components
        self.analyzer_process = None
        self.analyzer_queue = None
        self.result_queue = None

    def _cleanup_analyzer_process(self):
        """Clean up analyzer process and queues."""
        if self.analyzer_process and self.analyzer_process.is_alive():
            # Send shutdown signal
            if self.analyzer_queue:
                try:
                    self.analyzer_queue.put(None, timeout=1.0)
                except:
                    pass
            
            # Wait for process to finish
            self.analyzer_process.join(timeout=3.0)
            
            # Force terminate if still alive
            if self.analyzer_process.is_alive():
                self.analyzer_process.terminate()
                self.analyzer_process.join()
        
        self.analyzer_process = None
        self.analyzer_queue = None
        self.result_queue = None

    def run(self, bc_agent, cfg=None) -> Dict:
        """
        Run policy evaluation in legged gym environment with parallel trajectory analysis.
        
        Args:
            bc_agent: BCAgent with proper normalization and act() method
            cfg: Hydra config object containing cloc_analyzer settings
                   
        Returns:
            results: Evaluation results with analyzer summary
        """
        # Set multiprocessing start method to 'spawn'
        mp.set_start_method('spawn', force=True)
        
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

        # Initialize parallel trajectory analyzer
        analyzer_enabled = False
        analyzer_cfg = cfg.get('cloc_analyzer', {}) if cfg else {}
        
        if analyzer_cfg.get('enabled', False):
            try:
                # Create queues for inter-process communication
                self.analyzer_queue = mp.Queue(maxsize=100)  # Limit queue size to prevent memory issues
                self.result_queue = mp.Queue()
                
                # Start analyzer worker process
                self.analyzer_process = mp.Process(
                    target=analyzer_worker_process,
                    args=(self.analyzer_queue, analyzer_cfg, self.output_dir, self.n_obs_steps)
                )
                self.analyzer_process.start()
                analyzer_enabled = True
                
                # Register cleanup function
                atexit.register(self._cleanup_analyzer_process)
                
                print("Starting evaluation with parallel trajectory analysis...")
                print(f"Analysis config: enabled={analyzer_cfg.get('enabled')}, "
                      f"env_idx={analyzer_cfg.get('analyze_env_idx', 0)}, "
                      f"update_interval={analyzer_cfg.get('visualization', {}).get('update_interval', 10)}")
            except Exception as e:
                print(f"Failed to start analyzer process: {e}")
                analyzer_enabled = False
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

        # Realtime management variables
        realtime_factor_window = []
        realtime_factor_window_size = 50
        last_print_time = time.time()
        print_interval = 2.0  # Print realtime factor every 2 seconds
        env_dt = getattr(self.env, 'dt', 0.02)  # Default to 20ms if dt not available
        
        if self.realtime_mode:
            print(f"Running in realtime mode (target dt={env_dt:.4f}s)")
        else:
            print("Running at maximum speed (no realtime constraints)")

        # Evaluation loop
        for step_idx in range(self.max_steps):
            step_start_time = time.time()
            
            # Prepare observation dict for BCAgent (with proper normalization)
            obs_dict = {"obs": obs_history.to(device)}

            # Time policy inference if analyzer is enabled
            if analyzer_enabled:
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
                    actions = action_traj[:, self.n_obs_steps, :] if action_traj.dim() == 3 else action_traj
                else:
                    # Standard diffusion case: just actions
                    actions = result[:, self.n_obs_steps, :] if result.dim() == 3 else result
                    action_traj = result
                    state_traj = None

            if analyzer_enabled:
                inference_end.record()
                torch.cuda.synchronize()
                inference_time = inference_start.elapsed_time(inference_end) / 1000.0
            else:
                inference_time = 0.0

            # Step environment
            next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

            # Send trajectory analysis data to worker process (non-blocking)
            if analyzer_enabled and state_traj is not None:
                try:
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

                        # Prepare data for analyzer (convert tensors to CPU/numpy for serialization)
                        analyze_data = {
                            'type': 'analyze_step',
                            'args': {
                                'step_idx': step_idx,
                                'obs_history': obs_history[env_idx:env_idx+1].cpu(),  # Single env slice
                                'action_traj': action_traj[env_idx:env_idx+1].cpu(),
                                'state_traj': state_traj[env_idx:env_idx+1].cpu() if state_traj is not None else None,
                                'executed_action': actions[env_idx:env_idx+1].cpu(),
                                'reward': rewards[env_idx].item(),
                                'env_info': env_info,
                                'inference_time': inference_time
                            }
                        }
                        
                        # Send to analyzer process (non-blocking)
                        self.analyzer_queue.put_nowait(analyze_data)
                        
                        # Only analyze one environment unless analyze_all_envs is True
                        if not analyze_all_envs:
                            break
                            
                except queue.Full:
                    # Skip analysis if queue is full to prevent blocking
                    pass
                except Exception as e:
                    print(f"Warning: Failed to send data to analyzer: {e}")

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
                    if analyzer_enabled and (analyze_all_envs or idx == analyze_env_idx):
                        try:
                            self.analyzer_queue.put_nowait({
                                'type': 'episode_reset'
                            })
                        except queue.Full:
                            pass  # Skip if queue is full

                    # Print episode summary
                    if len(episode_rewards) % 10 == 0:
                        print(f"\nEpisodes: {len(episode_rewards)}, "
                              f"Last reward: {episode_rewards[-1]:.2f}, "
                              f"Mean reward: {np.mean(episode_rewards[-10:]):.2f}")

            pbar.update(1)
            
            # Realtime management
            if self.realtime_mode:
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                
                # Calculate realtime factor
                realtime_factor = env_dt / step_duration if step_duration > 0 else float('inf')
                realtime_factor_window.append(realtime_factor)
                
                # Maintain window size
                if len(realtime_factor_window) > realtime_factor_window_size:
                    realtime_factor_window.pop(0)
                
                # Print realtime factor periodically
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    avg_realtime_factor = np.mean(realtime_factor_window)
                    min_realtime_factor = np.min(realtime_factor_window)
                    max_realtime_factor = np.max(realtime_factor_window)
                    print(f"Step {step_idx}: Realtime factor: {avg_realtime_factor:.2f}x "
                          f"(min: {min_realtime_factor:.2f}x, max: {max_realtime_factor:.2f}x)")
                    last_print_time = current_time
                
                # Sleep to maintain realtime if computation was faster than env_dt
                sleep_time = env_dt - step_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
            # If not realtime mode, run as fast as possible (no sleep)
            
            # Check for training issues periodically (send to analyzer process)
            if analyzer_enabled:
                issue_cfg = analyzer_cfg.get('issue_detection', {})
                if (issue_cfg.get('enabled', True) and 
                    step_idx > 0 and 
                    step_idx % issue_cfg.get('check_interval', 200) == 0):
                    
                    try:
                        self.analyzer_queue.put_nowait({
                            'type': 'diagnose_issues',
                            'step_idx': step_idx,
                            'print_recommendations': issue_cfg.get('print_recommendations', True)
                        })
                    except queue.Full:
                        pass  # Skip if queue is full

        pbar.close()

        # Add unfinished episodes
        for i in range(self.n_envs):
            if current_lengths[i] > 0:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())

        # Get final diagnostic summary and issues from analyzer process
        analyzer_summary = {}
        final_issues = {}
        
        if analyzer_enabled:
            try:
                # Request final summary from analyzer
                self.analyzer_queue.put_nowait({
                    'type': 'get_summary',
                    'result_queue': self.result_queue
                })
                
                # Wait for result with timeout
                result_data = self.result_queue.get(timeout=5.0)
                analyzer_summary = result_data['analyzer_summary']
                final_issues = result_data['final_issues']
                
            except (queue.Empty, queue.Full, Exception) as e:
                print(f"Warning: Failed to get final analyzer summary: {e}")

        # Aggregate results
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "analyzer_summary": analyzer_summary,
            "analyzer_enabled": analyzer_enabled
        }

        print("\n" + "="*70)
        print(f"EVALUATION RESULTS for {self.task_name}:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Mean Reward: {results['mean_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
        print(f"  Mean Length: {results['mean_episode_length']:.1f}")
        
        if analyzer_enabled:
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
                    # Import here since we're in main process
                    from diffusion_policy.utils.cloc_analyzer import print_training_recommendations
                    print_training_recommendations()
            else:
                print(f"\n✅ No major training issues detected!")
        else:
            print("\n(Trajectory analysis was disabled)")

        print("="*70 + "\n")

        # Clean up analyzer process
        self._cleanup_analyzer_process()

        return results
