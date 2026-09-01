"""
Parallel IRL training utilities.
Supports CPU multiprocessing and sequential MPS processing.
"""

import os
import torch as th #type: ignore[import]
import numpy as np # type: ignore[import]
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3.ppo import MlpPolicy # type: ignore[import]
import multiprocessing as mp
from functools import partial
import traceback
import random

from training import train_and_evaluate_irl_agent

@dataclass
class IRLTaskConfig:
    """Configuration for a single IRL training task."""
    cluster_id: int
    trajectories: List
    trajectories_with_rew: List
    true_labels: List
    env_name: str
    env_id: str
    tr_name: str
    seed: int
    stage: str
    rl_timesteps: int
    base_learner_policy: Any = None  # Will be set at runtime


def _run_single_irl_task(task_config: Dict) -> Tuple[int, Optional[Dict]]:
    """
    Worker function to run a single IRL training task.
    This runs in a separate process for CPU parallelization.
    
    Args:
        task_config: Dictionary containing all task parameters
        
    Returns:
        Tuple of (cluster_id, results_dict or None)
    """
    try:
        # Import inside worker to avoid pickling issues
        
        cluster_id = task_config['cluster_id']
        
        # Set random seeds for reproducibility in this worker
        seed = task_config['seed'] 
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"[Worker] Starting IRL training for cluster {cluster_id}...")
        
        results = train_and_evaluate_irl_agent(
            cluster_id=cluster_id,
            trajectories=task_config['trajectories'],
            trajectories_with_rew=task_config['trajectories_with_rew'],
            true_labels=task_config['true_labels'],
            env_name=task_config['env_name'],
            env_id=task_config['env_id'],
            tr_name=task_config['tr_name'],
            seed=seed,
            base_learner_policy=MlpPolicy,
            stage=task_config['stage'],
            rl_timesteps=task_config['rl_timesteps'],
        )
        
        print(f"[Worker] Completed IRL training for cluster {cluster_id}")
        return cluster_id, results
        
    except Exception as e:
        print(f"[Worker] Error training cluster {task_config.get('cluster_id', '?')}: {e}")
        traceback.print_exc()
        return task_config.get('cluster_id', -1), None


def run_parallel_irl_cpu(
    task_configs: List[Dict],
    max_workers: Optional[int] = None,
    verbose: bool = False,
) -> Dict[int, Dict]:
    """
    Run IRL training in parallel using CPU multiprocessing.
    
    Args:
        task_configs: List of task configuration dictionaries
        max_workers: Maximum number of parallel workers (default: CPU count - 1)
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping cluster_id to results
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    # Limit workers to number of tasks
    max_workers = min(max_workers, len(task_configs))
    
    if verbose:
        print(f"\n[Parallel IRL] Starting {len(task_configs)} tasks with {max_workers} workers (CPU)")
    
    all_results = {}
    
    # Use spawn context for better cross-platform compatibility
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all tasks
        future_to_cid = {
            executor.submit(_run_single_irl_task, config): config['cluster_id']
            for config in task_configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_cid):
            cluster_id = future_to_cid[future]
            try:
                cid, result = future.result()
                if result is not None:
                    all_results[cid] = result
                    if verbose:
                        print(f"[Parallel IRL] Cluster {cid} completed successfully")
                else:
                    if verbose:
                        print(f"[Parallel IRL] Cluster {cid} returned no results")
            except Exception as e:
                print(f"[Parallel IRL] Cluster {cluster_id} failed with exception: {e}")
    
    if verbose:
        print(f"[Parallel IRL] Completed {len(all_results)}/{len(task_configs)} tasks successfully")
    
    return all_results


def run_sequential_irl(
    task_configs: List[Dict],
    verbose: bool = True,
) -> Dict[int, Dict]:
    """
    Run IRL training sequentially (for MPS or when parallelization is not desired).
    
    Args:
        task_configs: List of task configuration dictionaries
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping cluster_id to results
    """
    if verbose:
        print(f"\n[Sequential IRL] Starting {len(task_configs)} tasks")
    
    all_results = {}
    
    for i, config in enumerate(task_configs):
        cluster_id = config['cluster_id']
        
        if verbose:
            print(f"\n[Sequential IRL] Task {i+1}/{len(task_configs)}: Cluster {cluster_id}")
        
        try:
            results = train_and_evaluate_irl_agent(
                cluster_id=cluster_id,
                trajectories=config['trajectories'],
                trajectories_with_rew=config['trajectories_with_rew'],
                true_labels=config['true_labels'],
                env_name=config['env_name'],
                env_id=config['env_id'],
                tr_name=config['tr_name'],
                seed=config['seed'],
                base_learner_policy=MlpPolicy,
                stage=config['stage'],
                rl_timesteps=config['rl_timesteps'],
            )
            
            if results is not None:
                all_results[cluster_id] = results
                if verbose:
                    print(f"[Sequential IRL] Cluster {cluster_id} completed successfully")
                    
        except Exception as e:
            print(f"[Sequential IRL] Cluster {cluster_id} failed: {e}")
            traceback.print_exc()
    
    if verbose:
        print(f"\n[Sequential IRL] Completed {len(all_results)}/{len(task_configs)} tasks successfully")
    
    return all_results


def run_irl_training(
    task_configs: List[Dict],
    device: th.device,
    max_workers: Optional[int] = None,
    force_sequential: bool = False,
    verbose: bool = False,
) -> Dict[int, Dict]:
    """
    Unified interface for running IRL training with automatic parallelization strategy.
    
    - CPU: Uses multiprocessing for parallel training
    - MPS: Uses sequential training (MPS doesn't support multiprocessing well)
    - CUDA: Uses sequential training (GPU memory conflicts with multiprocessing)
    
    Args:
        task_configs: List of task configuration dictionaries
        device: PyTorch device (cpu, mps, or cuda)
        max_workers: Maximum parallel workers (only used for CPU)
        force_sequential: Force sequential execution regardless of device
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping cluster_id to results
    """
    device_type = str(device).split(':')[0]  # Get 'cpu', 'mps', or 'cuda'
    
    if force_sequential:
        if verbose:
            print(f"[IRL Training] Forced sequential mode")
        return run_sequential_irl(task_configs, verbose=verbose)
    
    if device_type == 'cpu':
        # CPU: Use multiprocessing
        if verbose:
            print(f"[IRL Training] Using CPU parallelization")
        return run_parallel_irl_cpu(task_configs, max_workers=max_workers, verbose=verbose)
    
    elif device_type == 'mps':
        # MPS: Sequential (MPS doesn't work well with multiprocessing)
        if verbose:
            print(f"[IRL Training] Using sequential mode (MPS device)")
        return run_sequential_irl(task_configs, verbose=verbose)
    
    elif device_type == 'cuda':
        # CUDA: Sequential (avoid GPU memory conflicts)
        if verbose:
            print(f"[IRL Training] Using sequential mode (CUDA device)")
        return run_sequential_irl(task_configs, verbose=verbose)
    
    else:
        # Unknown device: default to sequential
        if verbose:
            print(f"[IRL Training] Unknown device '{device_type}', using sequential mode")
        return run_sequential_irl(task_configs, verbose=verbose)


def prepare_irl_tasks(
    cluster_ids: List[int],
    labels: np.ndarray,
    indices: np.ndarray,
    trajectory_manager: Dict,
    env_name: str,
    env_id: str,
    tr_name: str,
    seed: int,
    stage: str,
    rl_timesteps: int,
) -> List[Dict]:
    """
    Prepare IRL task configurations for a list of cluster IDs.
    
    Args:
        cluster_ids: List of cluster IDs to process
        labels: Cluster labels for all trajectories
        indices: Indices into trajectory_manager
        trajectory_manager: Dictionary containing trajectory data
        env_name: Environment name
        env_id: Environment ID
        tr_name: Training algorithm name
        seed: Random seed
        stage: Training stage name
        rl_timesteps: Number of RL timesteps
        
    Returns:
        List of task configuration dictionaries
    """
    task_configs = []
    
    for cid in cluster_ids:
        # Extract trajectories for this cluster
        cluster_mask = (labels == cid)
        manager_indices = indices[cluster_mask]
        if tr_name == "gail":
            manager_indices = np.sort(manager_indices) #let's see if ordering them like khgail works
        
        
        trajectories = [trajectory_manager[i]['original_trajectory'] for i in manager_indices]
        trajectories_with_rew = [trajectory_manager[i]['original_trajectory_with_rew'] for i in manager_indices]
        true_labels = [trajectory_manager[i]['real_cluster_label'] for i in manager_indices]
        # from collections import Counter
        # expert_mode = Counter(true_labels).most_common(1)[0][0] - 10
        # print(f"Sorted manager indices for cluster {cid} with expert mode {expert_mode}: {manager_indices}")
        if not trajectories:
            print(f"[prepare_irl_tasks] Cluster {cid} has no trajectories. Skipping.")
            continue
        
        task_config = {
            'cluster_id': cid,
            'trajectories': trajectories,
            'trajectories_with_rew': trajectories_with_rew,
            'true_labels': true_labels,
            'env_name': env_name,
            'env_id': env_id,
            'tr_name': tr_name,
            'seed': seed,
            'stage': stage,
            'rl_timesteps': rl_timesteps,
        }
        
        task_configs.append(task_config)
    
    return task_configs