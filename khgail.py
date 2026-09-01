import os
import argparse
import random
import pickle
from collections import Counter
import math
import csv  # added

import numpy as np #type: ignore[import]
import torch as th #type: ignore[import]
from scipy.sparse.csgraph import connected_components #type: ignore[import]
from typing import Tuple #type: ignore[import]
from sklearn.preprocessing import StandardScaler, QuantileTransformer #type: ignore[import]
from sklearn.cluster import KMeans #type: ignore[import]
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score #type: ignore[import]

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # ignore warnings

try:
    from sklearn.cluster import HDBSCAN #type: ignore[import]
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

from imitation.data.types import Trajectory, TrajectoryWithRew  # type: ignore
from stable_baselines3 import PPO #type: ignore[import]
from stable_baselines3.ppo import MlpPolicy #type: ignore[import]
from stable_baselines3.common.env_util import make_vec_env #type: ignore[import]
from imitation.algorithms.adversarial.gail import GAIL  # type: ignore
from imitation.algorithms.adversarial.airl import AIRL  # type: ignore
from imitation.rewards.reward_nets import BasicShapedRewardNet  # type: ignore
from imitation.util.networks import RunningNorm  # type: ignore

# Local envs
from essinfogail.envs.wrappers import FixedLengthEnvWrapper
from graph_clustering import *

def make_env_by_name(env_name: str, num_modes: int, render: bool = False):
    render_kwargs = {"render_mode": "human"} if render else {}
    if env_name == "Reacher-v4":
        import essinfogail.envs.reacher as reacher_mod
        try:
            return reacher_mod.MultimodalReacher(num_modes, **render_kwargs)
        except TypeError:
            return reacher_mod.MultimodalReacher(num_modes=num_modes)
    if env_name == "Pusher-v4":
        import essinfogail.envs.pusher as pusher_mod
        try:
            return pusher_mod.MultimodalPusher(num_modes, **render_kwargs)
        except TypeError:
            return pusher_mod.MultimodalPusher(num_modes=num_modes)
    if env_name == "Humanoid-v4":
        import essinfogail.envs.humanoid as humanoid_mod
        try:
            env = humanoid_mod.MultimodalHumanoid(num_modes, **render_kwargs)
        except TypeError:
            env = humanoid_mod.MultimodalHumanoid(num_modes=num_modes)
        return FixedLengthEnvWrapper(env)
    if env_name == "Walker2d-v4":
        import essinfogail.envs.walker as walker_mod
        try:
            env = walker_mod.MultimodalWalker(num_modes, **render_kwargs)
        except TypeError:
            env = walker_mod.MultimodalWalker(num_modes=num_modes)
        return FixedLengthEnvWrapper(env)
    if env_name == "Traj2d":
        import my_envs.traj2d_gymnasium as traj2d_mod #type: ignore[import]
        return traj2d_mod.Traj()
    if env_name == "Hopper-v4":
        import essinfogail.envs.hopper_v4 as hopper_mod
        try:
            return hopper_mod.MultimodalHopper(num_modes, **render_kwargs)
        except TypeError:
            return hopper_mod.MultimodalHopper(num_modes=num_modes)
    if env_name == "HalfCheetah-v5":
        import essinfogail.envs.halfcheetah_v5 as halfcheetah_mod
        try:
            return halfcheetah_mod.MultimodalHalfCheetah(num_modes, **render_kwargs)
        except TypeError:
            return halfcheetah_mod.MultimodalHalfCheetah(num_modes=num_modes)
    raise ValueError(f"Unsupported env: {env_name}")

def evaluate_policy_reward(policy, env, num_episodes=20, mode_idx=None, env_name=""):
    returns = []
    for _ in range(num_episodes):
        try:
            reset_out = env.reset(mode_idx=mode_idx)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        except TypeError:
            obs = env.reset()
        terminated = False
        truncated = False
        ep_ret = 0.0
        max_steps = getattr(env, "_max_episode_steps", 1000)
        for _t in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            out = env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
            else:
                obs, reward, done, info = out
                terminated = bool(done); truncated = False
            reward = info.get("reward_eval", reward)
            ep_ret += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_ret)
    if len(returns) == 0:
        return 0.0, 0.0
    return float(np.mean(returns)), float(np.std(returns))

def calculate_original_expert_reward_stats(trajectories_with_rew):
    totals = [float(np.sum(t.rews)) for t in trajectories_with_rew if getattr(t, "rews", None) is not None]
    if not totals:
        return 0.0, 0.0
    return float(np.mean(totals)), float(np.std(totals))

def load_expert_set(env_name: str, num_trajs: int, ratio: int, seed: int):
    random.seed(seed); np.random.seed(seed); th.manual_seed(seed)
    name_env = "2D-Trajectory" if env_name == "Traj2d" else env_name
    if env_name in ["Reacher-v4", "Pusher-v4", "Traj2d"]:
        modes = 6
    elif env_name in ["Humanoid-v4", "Walker2d-v4", "Hopper-v4", "HalfCheetah-v5"]:
        modes = 3
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    demos = []
    demos_withrew = []
    labels = []
    for m in range(modes):
        if env_name in ["Reacher-v4", "Pusher-v4", "Walker2d-v4"]:
            fp = f"essinfogail/expert_imitation_trajectories/expert_imitation_trajectories_{name_env}_mode_{m}.pkl"
        elif env_name in ["Hopper-v4", "HalfCheetah-v5"]:
            fp = f"expert_trajectories_new/{name_env}_task_{m}.pkl"
        fpr = fp.replace(".pkl", "_withrew.pkl")
        if not os.path.exists(fp) or not os.path.exists(fpr):
            raise FileNotFoundError(f"Missing expert files for mode {m}: {fp} / {fpr}")
        with open(fp, "rb") as f:
            d = pickle.load(f)
        with open(fpr, "rb") as f:
            dr = pickle.load(f)
        demos.append(d)
        demos_withrew.append(dr)
        labels.append(np.array([10 + m] * len(d), dtype=int))

    weights = [ratio**(modes - 1 - i) for i in range(modes)]
    total = sum(weights)
    counts = [int(round(num_trajs * w / total)) for w in weights]
    counts[-1] += num_trajs - sum(counts)

    sel_d, sel_dr, sel_lbl = [], [], []
    for m in range(modes):
        sel_d.extend(demos[m][:counts[m]])
        sel_dr.extend(demos_withrew[m][:counts[m]])
        sel_lbl.append(labels[m][:counts[m]])
    true_labels = np.concatenate(sel_lbl, axis=0)
    trajectories = sel_d
    trajectories_with_rew = sel_dr
    return trajectories, trajectories_with_rew, true_labels, modes

def infer_dims(traj: Trajectory):
    s = np.asarray(traj.obs)
    a = np.asarray(traj.acts) if getattr(traj, "acts", None) is not None else None
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    s_dim = s.shape[-1]
    if a is None or np.asarray(a).size == 0:
        raise ValueError("Trajectory has no actions; cannot interleave.")
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    a_dim = a.shape[-1]
    return s_dim, a_dim

def interleave_flatten(traj: Trajectory, max_steps: int, pad_value: float = 0.0):
    s = np.asarray(traj.obs)
    a = np.asarray(traj.acts)
    if s.ndim == 1: s = s.reshape(-1, 1)
    if a.ndim == 1: a = a.reshape(-1, 1)
    # typical Trajectory has len(obs) = len(acts)+1; pair s[0..T-1] with a[0..T-1]
    T = min(len(a), len(s)-1)
    T_use = min(T, max_steps)
    s_dim, a_dim = s.shape[-1], a.shape[-1]
    step_dim = s_dim + a_dim
    out = np.full((max_steps, step_dim), pad_value, dtype=np.float32)
    if T_use > 0:
        sa = np.concatenate([s[:T_use], a[:T_use]], axis=1)
        out[:T_use] = sa
    return out.reshape(-1)

def build_interleaved_matrix(trajs, max_steps: int = None, pad_value: float = 0.0):
    # choose max_steps if not given: 95th percentile of action lengths (cap >= 8)
    lengths = [len(np.asarray(t.acts)) for t in trajs]
    if max_steps is None:
        max_steps = max(8, int(np.ceil(np.percentile(lengths, 95))))
    # get dims from first trajectory
    s_dim, a_dim = infer_dims(trajs[0])
    vec_dim = max_steps * (s_dim + a_dim)
    X = np.zeros((len(trajs), vec_dim), dtype=np.float32)
    for i, t in enumerate(trajs):
        X[i] = interleave_flatten(t, max_steps=max_steps, pad_value=pad_value)
    return X, {"max_steps": int(max_steps), "s_dim": int(s_dim), "a_dim": int(a_dim), "vec_dim": int(vec_dim)}

def cluster_features(X: np.ndarray, algo: str, k: int, seed: int, min_cluster_size: int = 10):
    algo = algo.lower()
    if algo == "kmeans":
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X).labels_
        info = {"noise_count": 0}
        return labels.astype(int), info
    elif algo == "hdbscan":
        if not _HAS_HDBSCAN:
            raise RuntimeError("hdbscan not installed. pip install hdbscan")
        min_samples = max(1, int(math.sqrt(min_cluster_size)))
        model = HDBSCAN(min_cluster_size=max(5, min_cluster_size), min_samples=min_samples, metric="euclidean")
        labels = model.fit_predict(X)  # -1 for noise
        info = {"noise_count": int(np.sum(labels == -1))}
        return labels.astype(int), info
    elif algo == "leiden":
        # Graph-based Leiden clustering on raw features with proper k-resolution sweep
        return _cluster_leiden_with_sweep(X, seed, min_cluster_size)
    
    else:
        raise ValueError("algo must be 'kmeans' or 'hdbscan' or 'leiden'")
    
def _cluster_leiden_with_sweep(
    X: np.ndarray, 
    seed: int, 
    min_cluster_size: int,
) -> Tuple[np.ndarray, dict]:
    """
    Leiden clustering with joint k-resolution sweep (matches CoMI-IRL's graph clustering).
    
    Selection strategy: STABILITY-BASED (most frequent cluster count wins)
    Then among stable configurations, pick by best silhouette as tiebreaker.
    
    This is a simplified version without:
    - Behavioral/Jacobian features (those are for learned embeddings only)
    - Target-aware selection (baselines don't know the true K)
    
    Args:
        X: Feature matrix [N, D]
        seed: Random seed
        min_cluster_size: Minimum cluster size
        
    Returns:
        labels: Cluster assignments
        info: Dictionary with clustering metadata
    """
    N = len(X)
    
    # Normalize for cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # Define search grids
    k_min = max(5, N // 50)
    k_max = min(100, N // 3)
    n_k_points = min(15, (k_max - k_min) // 5 + 1)
    k_values = list(np.linspace(k_min, k_max, n_k_points).astype(int))
    k_values = sorted(set(k_values))
    
    resolution_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    # Step 1: Check for isolated components first
    for test_k in [15, 30, 50]:
        if test_k >= N:
            continue
        try:
            graph = TrajectoryGraph(embeddings=X_norm, k=test_k, metric='cosine', symmetric=True)
            n_components, comp_labels = connected_components(graph.adjacency, directed=False, return_labels=True)
            
            # Count significant components
            unique, counts = np.unique(comp_labels, return_counts=True)
            significant_components = [(u, c) for u, c in zip(unique, counts) if c >= min_cluster_size]
            
            if len(significant_components) >= 2:
                # Use connected components directly
                labels = comp_labels.copy()
                # Filter small components to noise
                for u, c in zip(unique, counts):
                    if c < min_cluster_size:
                        labels[comp_labels == u] = -1
                
                # Relabel to consecutive integers
                valid_labels = sorted(set(labels) - {-1})
                label_map = {old: new for new, old in enumerate(valid_labels)}
                label_map[-1] = -1
                final_labels = np.array([label_map[l] for l in labels])
                
                # Compute silhouette for the result
                mask_core = final_labels != -1
                sil = -1.0
                if mask_core.sum() >= 10 and len(valid_labels) >= 2:
                    try:
                        sil = silhouette_score(X_norm[mask_core], final_labels[mask_core], metric='cosine')
                    except Exception:
                        pass
                
                info = {
                    "noise_count": int(np.sum(final_labels == -1)),
                    "method": "leiden_components",
                    "k_used": test_k,
                    "n_clusters": len(valid_labels),
                    "silhouette": sil,
                    "selection_reason": "isolated_components",
                }
                print(f"[G-GAIL Leiden] Found {len(valid_labels)} isolated components at k={test_k}")
                return final_labels.astype(int), info
        except Exception:
            continue
    
    # Step 2: Joint k-resolution sweep
    all_results = []
    
    for k in k_values:
        if k >= N:
            continue
        try:
            graph = TrajectoryGraph(embeddings=X_norm, k=k, metric='cosine', symmetric=True)
        except Exception:
            continue
        
        for res in resolution_values:
            try:
                clusterer = GraphClusterer(resolution=res, seed=seed)
                labels = clusterer.fit(graph)
                
                # Filter small clusters
                unique, counts = np.unique(labels, return_counts=True)
                filtered_labels = labels.copy()
                for lab, cnt in zip(unique, counts):
                    if cnt < min_cluster_size:
                        filtered_labels[labels == lab] = -1
                
                # Count valid clusters
                valid_labels_set = set(filtered_labels) - {-1}
                n_valid = len(valid_labels_set)
                
                if n_valid < 2:
                    continue
                
                # Compute silhouette score
                mask_core = filtered_labels != -1
                if mask_core.sum() >= 10:
                    try:
                        sil = silhouette_score(X_norm[mask_core], filtered_labels[mask_core], metric='cosine')
                    except Exception:
                        sil = -1.0
                else:
                    sil = -1.0
                
                all_results.append({
                    'k': k,
                    'resolution': res,
                    'n_clusters': n_valid,
                    'silhouette': sil,
                    'modularity': clusterer.modularity,
                    'labels': filtered_labels.copy(),
                    'noise_count': int((~mask_core).sum()),
                })
            except Exception:
                continue
    
    if not all_results:
        # Fallback: single cluster
        print("[G-GAIL Leiden] No valid clustering found, falling back to single cluster")
        labels = np.zeros(N, dtype=int)
        return labels, {"noise_count": 0, "method": "leiden_fallback", "n_clusters": 1, "selection_reason": "fallback"}
    
    # Step 3: STABILITY-BASED SELECTION (matching CoMI-IRL)
    # Group by n_clusters, find most stable (frequent) cluster count
    cluster_counts = Counter(r['n_clusters'] for r in all_results)
    
    # Sort by frequency (stability), then by cluster count (prefer more granular)
    sorted_counts = sorted(cluster_counts.items(), key=lambda x: (-x[1], -x[0]))
    
    print(f"[G-GAIL Leiden] Stability analysis:")
    for n_clust, freq in sorted_counts[:5]:
        print(f"  {n_clust} clusters: {freq} configurations ({100*freq/len(all_results):.1f}%)")
    
    # Most stable cluster count
    most_stable_n = sorted_counts[0][0]
    stability_count = sorted_counts[0][1]
    
    # Get all configurations with the most stable cluster count
    stable_candidates = [r for r in all_results if r['n_clusters'] == most_stable_n]
    
    # Among stable candidates, pick by best silhouette (tiebreaker)
    best = max(stable_candidates, key=lambda x: x['silhouette'])
    
    print(f"[G-GAIL Leiden] Selected: {most_stable_n} clusters (stability={stability_count}/{len(all_results)})")
    print(f"[G-GAIL Leiden] Best config: k={best['k']}, resolution={best['resolution']:.3f}, silhouette={best['silhouette']:.4f}")
    
    # Relabel to consecutive integers
    labels = best['labels']
    valid_labels = sorted(set(labels) - {-1})
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    final_labels = np.array([label_map.get(l, -1) for l in labels])
    
    info = {
        "noise_count": best['noise_count'],
        "method": "leiden_sweep",
        "k_used": best['k'],
        "resolution_used": best['resolution'],
        "n_clusters": best['n_clusters'],
        "silhouette": best['silhouette'],
        "modularity": best['modularity'],
        "stability_count": stability_count,
        "total_configs_tested": len(all_results),
        "selection_reason": "stability_then_silhouette",
    }
    
    return final_labels.astype(int), info

def train_irl_for_cluster(cluster_id: int,
                          trajectories: list,
                          env_name: str,
                          num_modes: int,
                          expert_mode_idx: int,
                          algo: str,
                          seed: int,
                          rl_steps: int,
                          save_prefix: str = None,
                          skip_if_exists: bool = True):  # NEW: skip training if model exists
    """
    Train an IRL agent for a cluster. If skip_if_exists=True and saved model exists,
    load and evaluate instead of retraining.
    """
    def count_transitions(trajs):
        return sum(len(np.asarray(t.acts)) for t in trajs if getattr(t, "acts", None) is not None)

    # Example:
    total_demo_transitions = count_transitions(trajectories)
    print(f"[Cluster {cluster_id}] Total demo transitions:", total_demo_transitions)

    batch_size = 256 if env_name not in ["Hopper-v4", "HalfCheetah-v5"] else 1024

    if total_demo_transitions < batch_size:
        while total_demo_transitions < batch_size:
            batch_size = batch_size // 2
        print(f"[Cluster {cluster_id}] Adjusted batch size to {batch_size} due to limited demo transitions.")
    else:
        print(f"[Cluster {cluster_id}] Using batch size {batch_size}.")

    # Check if model already exists
    if skip_if_exists and save_prefix is not None:
        model_path = save_prefix + ".zip"
        if os.path.exists(model_path):
            print(f"[Cluster {cluster_id}] Found existing model at {model_path}. Loading instead of training...")
            
            # Load the saved PPO model
            env = make_env_by_name(env_name, num_modes)
            try:
                env.reset(mode_idx=expert_mode_idx)
            except Exception:
                _ = env.reset()
            
            learner = PPO.load(model_path)
            
            # Evaluate the loaded model
            mean_ret, std_ret = evaluate_policy_reward(
                learner, env, 
                num_episodes=max(5, min(20, len(trajectories))),
                mode_idx=expert_mode_idx, 
                env_name=env_name
            )
            
            try:
                env.close()
            except Exception:
                pass
            
            print(f"[Cluster {cluster_id}] Loaded model evaluation: {mean_ret:.2f} ± {std_ret:.2f}")
            return {"learner_mean": mean_ret, "learner_std": std_ret}
    
    # --- Original training code ---
    # Env + VecEnv
    env = make_env_by_name(env_name, num_modes)
    try:
        env.reset(mode_idx=expert_mode_idx)
    except Exception:
        _ = env.reset()
    venv = make_vec_env(lambda: make_env_by_name(env_name, num_modes), n_envs=1, seed=seed)
    if env_name == "HalfCheetah-v5":
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
    elif env_name == "Hopper-v4":
        policy_kwargs = dict(activation_fn=th.nn.Tanh,
                    net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else:
        policy_kwargs = dict()

    
    learner = PPO(env=venv, policy=MlpPolicy, batch_size=64, ent_coef=0.01,
                  learning_rate=0.0003, gamma=0.99, clip_range=0.2, n_epochs=10, seed=seed, policy_kwargs=policy_kwargs,
                  verbose=0
                  )
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        reward_hid_sizes=(32,),
        potential_hid_sizes=(32, 32),
    )

    algo = algo.lower()
    if algo == "gail":
        trainer = GAIL(
            demonstrations=trajectories,
            demo_batch_size=batch_size,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=32 if env_name not in ["Hopper-v4", "HalfCheetah-v5"] else 8,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
        )
    elif algo == "airl":
        trainer = AIRL(
            demonstrations=trajectories,
            demo_batch_size=batch_size,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=32 if env_name not in ["Hopper-v4", "HalfCheetah-v5"] else 8,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
        )
    else:
        raise ValueError("IRL algo must be 'gail' or 'airl'")

    trainer.train(total_timesteps=rl_steps)

    # Save the trained agent and components if requested (match main.py naming)
    if save_prefix is not None:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        # PPO/learner zip
        learner.save(save_prefix + ".zip")
        # policy state dict
        th.save(learner.policy.state_dict(), save_prefix + "_policy.zip")
        # reward net full module and its state dict
        th.save(reward_net.state_dict(), save_prefix + "_reward_net_state_dict.zip")
        th.save(reward_net, save_prefix + "_reward_net.zip")

    mean_ret, std_ret = evaluate_policy_reward(learner, env, num_episodes=max(5, min(20, len(trajectories))),
                                               mode_idx=expert_mode_idx, env_name=env_name)
    try:
        env.close()
    except Exception:
        pass
    return {"learner_mean": mean_ret, "learner_std": std_ret}

def _approach_name(clusterer: str, irl: str) -> str:
    if clusterer.lower() == "kmeans":
        c = "K"
    elif clusterer.lower() == "hdbscan":
        c = "H"
    elif clusterer.lower() == "leiden":
        c = "G"  # G for Graph-based
    else:
        c = clusterer[0].upper()
    a = irl.upper()
    return f"{c}-{a}"

def compute_clustering_metrics(X: np.ndarray, labels_pred: np.ndarray, true_labels: np.ndarray) -> dict:
    """
    Compute NMI, ARI, and Silhouette score for clustering results.
    
    Args:
        X: Feature matrix [N, D]
        labels_pred: Predicted cluster labels
        true_labels: Ground truth labels
        
    Returns:
        Dictionary with nmi, ari, silhouette scores
    """
    # Filter out noise points (label == -1)
    mask_core = labels_pred != -1
    
    metrics = {
        'nmi': np.nan,
        'ari': np.nan,
        'silhouette': np.nan,
        'n_core_samples': int(mask_core.sum()),
        'n_noise_samples': int((~mask_core).sum()),
    }
    
    # Need at least 2 samples and 2 unique labels for metrics
    if mask_core.sum() < 2:
        return metrics
    
    labels_core = labels_pred[mask_core]
    true_labels_core = true_labels[mask_core]
    X_core = X[mask_core]
    
    n_pred_clusters = len(np.unique(labels_core))
    n_true_clusters = len(np.unique(true_labels_core))
    
    # NMI and ARI require at least 2 clusters in both pred and true
    if n_pred_clusters > 1 and n_true_clusters > 1:
        metrics['nmi'] = float(normalized_mutual_info_score(true_labels_core, labels_core))
        metrics['ari'] = float(adjusted_rand_score(true_labels_core, labels_core))
    
    # Silhouette requires at least 2 clusters in predictions
    if n_pred_clusters >= 2 and len(X_core) > n_pred_clusters:
        try:
            metrics['silhouette'] = float(silhouette_score(X_core, labels_core, metric='euclidean'))
        except Exception as e:
            print(f"[Warning] Silhouette computation failed: {e}")
            metrics['silhouette'] = np.nan
    
    return metrics

def _run_single_irl_task_khgail(task_config: dict) -> dict:
    """
    Worker function to run a single IRL training task for khgail.
    This runs in a separate process for CPU parallelization.
    
    Args:
        task_config: Dictionary containing all task parameters
        
    Returns:
        Dictionary with cluster results or error info
    """
    try:
        cluster_id = task_config['cluster_id']
        
        # Set random seeds for reproducibility in this worker
        seed = task_config['seed']
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"[Worker] Starting IRL training for cluster {cluster_id}...")
        
        irl_res = train_irl_for_cluster(
            cluster_id=cluster_id,
            trajectories=task_config['trajectories'],
            env_name=task_config['env_name'],
            num_modes=task_config['num_modes'],
            expert_mode_idx=task_config['dominant_mode'],
            algo=task_config['irl_algo'],
            seed=seed,
            rl_steps=task_config['rl_steps'],
            save_prefix=task_config['save_prefix'],
        )
        
        # Build result record
        rec = {
            "env": task_config['env_name'],
            "seed": seed,
            "approach": task_config['approach'],
            "clusterer": task_config['clusterer'],
            "irl": task_config['irl_algo'],
            "requested_k": task_config['requested_k'],
            "actual_num_clusters": task_config['actual_num_clusters'],
            "noise_count": task_config['noise_count'],
            "nmi": task_config['nmi'],
            "ari": task_config['ari'],
            "silhouette": task_config['silhouette'],
            "cluster_id": cluster_id,
            "cluster_size": task_config['cluster_size'],
            "dominant_mode": task_config['dominant_mode'],
            "learner_reward_mean": float(irl_res["learner_mean"]),
            "learner_reward_std": float(irl_res["learner_std"]),
            "rl_steps": task_config['rl_steps'],
            "vec_dim": task_config['vec_dim'],
            "interleave_steps": task_config['interleave_steps'],
            "agent_path": task_config['save_prefix'] + ".zip",
            "policy_state_dict_path": task_config['save_prefix'] + "_policy.zip",
            "reward_net_path": task_config['save_prefix'] + "_reward_net.zip",
            "reward_net_state_dict_path": task_config['save_prefix'] + "_reward_net_state_dict.zip",
            "success": True,
            "error": None,
        }
        
        print(f"[Worker] Completed IRL training for cluster {cluster_id}")
        return rec
        
    except Exception as e:
        print(f"[Worker] Error training cluster {task_config.get('cluster_id', '?')}: {e}")
        traceback.print_exc()
        return {
            "cluster_id": task_config.get('cluster_id', -1),
            "success": False,
            "error": str(e),
        }

def run_parallel_irl_khgail(
    task_configs: list,
    max_workers: int = None,
    verbose: bool = True,
) -> list:
    """
    Run IRL training in parallel using CPU multiprocessing.
    
    Args:
        task_configs: List of task configuration dictionaries
        max_workers: Maximum number of parallel workers (default: CPU count - 1)
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    # Limit workers to number of tasks
    max_workers = min(max_workers, len(task_configs))
    
    if verbose:
        print(f"\n[Parallel IRL] Starting {len(task_configs)} tasks with {max_workers} workers (CPU)")
    
    all_results = []
    
    # Use spawn context for better cross-platform compatibility
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all tasks
        future_to_cid = {
            executor.submit(_run_single_irl_task_khgail, config): config['cluster_id']
            for config in task_configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_cid):
            cluster_id = future_to_cid[future]
            try:
                result = future.result()
                all_results.append(result)
                if verbose:
                    if result.get('success', False):
                        print(f"[Parallel IRL] Cluster {cluster_id} completed successfully")
                    else:
                        print(f"[Parallel IRL] Cluster {cluster_id} failed: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"[Parallel IRL] Cluster {cluster_id} raised exception: {e}")
                all_results.append({
                    "cluster_id": cluster_id,
                    "success": False,
                    "error": str(e),
                })
    
    # Sort by cluster_id for consistent ordering
    all_results.sort(key=lambda x: x.get('cluster_id', -1))
    
    successful = sum(1 for r in all_results if r.get('success', False))
    if verbose:
        print(f"[Parallel IRL] Completed {successful}/{len(task_configs)} tasks successfully")
    
    return all_results

def run_sequential_irl_khgail(
    task_configs: list,
    verbose: bool = True,
) -> list:
    """
    Run IRL training sequentially.
    
    Args:
        task_configs: List of task configuration dictionaries
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries
    """
    if verbose:
        print(f"\n[Sequential IRL] Starting {len(task_configs)} tasks")
    
    all_results = []
    
    for i, config in enumerate(task_configs):
        cluster_id = config['cluster_id']
        
        if verbose:
            print(f"\n[Sequential IRL] Task {i+1}/{len(task_configs)}: Cluster {cluster_id}")
        
        result = _run_single_irl_task_khgail(config)
        all_results.append(result)
    
    successful = sum(1 for r in all_results if r.get('success', False))
    if verbose:
        print(f"\n[Sequential IRL] Completed {successful}/{len(task_configs)} tasks successfully")
    
    return all_results

def prepare_khgail_tasks(
    unique_clusters: list,
    labels_pred: np.ndarray,
    trajectories: list,
    trajectories_with_rew: list,
    true_labels: np.ndarray,
    env_name: str,
    irl_algo: str,
    seed: int,
    rl_steps: int,
    base_dir: str,
    approach: str,
    clusterer: str,
    requested_k: int,
    noise_count: int,
    metrics: dict,  # Changed from nmi to metrics dict
    meta: dict,
    num_modes: int,
) -> list:
    """
    Prepare IRL task configurations for khgail.
    
    Args:
        unique_clusters: List of unique cluster IDs (excluding -1)
        labels_pred: Predicted cluster labels for all trajectories
        trajectories: List of all trajectories
        trajectories_with_rew: List of all trajectories with rewards
        true_labels: True cluster labels
        env_name: Environment name
        irl_algo: IRL algorithm ('gail' or 'airl')
        seed: Random seed
        rl_steps: Number of RL training steps
        base_dir: Base directory for saving models
        approach: Approach name string
        clusterer: Clusterer name ('kmeans' or 'hdbscan')
        requested_k: Requested K for kmeans
        noise_count: Number of noise points
        metrics: Dictionary containing nmi, ari, silhouette scores
        meta: Metadata dictionary from build_interleaved_matrix
        num_modes: Number of environment modes
        
    Returns:
        List of task configuration dictionaries
    """
    task_configs = []
    
    # Extract metrics, handling NaN values
    nmi = metrics.get('nmi', np.nan)
    ari = metrics.get('ari', np.nan)
    silhouette = metrics.get('silhouette', np.nan)
    
    for cid in unique_clusters:
        c_mask = (labels_pred == cid)
        idxs = np.where(c_mask)[0]
        c_trajs = [trajectories[i] for i in idxs]
        c_trajs_withrew = [trajectories_with_rew[i] for i in idxs]
        c_true = [int(x) for x in true_labels[idxs]]
        
        if len(c_trajs) == 0:
            continue
        
        # Determine dominant mode
        per_modes = [m - 10 for m in c_true]
        dom_mode, _ = Counter(per_modes).most_common(1)[0]
        # print(f"Trajectory indices for cluster {cid} with dominant mode {dom_mode}: {idxs.tolist()}")

        
        # Save path prefix
        agent_dir = os.path.join(base_dir, f"C{cid}")
        os.makedirs(agent_dir, exist_ok=True)
        base_name = f"{irl_algo}_cluster_{cid}_mode_{dom_mode}"
        save_prefix = os.path.join(agent_dir, base_name)
        
        task_config = {
            'cluster_id': cid,
            'trajectories': c_trajs,
            'trajectories_with_rew': c_trajs_withrew,
            'true_labels': c_true,
            'env_name': env_name,
            'num_modes': num_modes,
            'dominant_mode': dom_mode,
            'irl_algo': irl_algo,
            'seed': seed,
            'rl_steps': rl_steps,
            'save_prefix': save_prefix,
            'approach': approach,
            'clusterer': clusterer,
            'requested_k': requested_k if clusterer == "kmeans" else "",
            'actual_num_clusters': len(unique_clusters),
            'noise_count': noise_count,
            'nmi': nmi if not np.isnan(nmi) else "",
            'ari': ari if not np.isnan(ari) else "",
            'silhouette': silhouette if not np.isnan(silhouette) else "",
            'cluster_size': len(c_trajs),
            'vec_dim': meta["vec_dim"],
            'interleave_steps': meta["max_steps"],
        }
        
        task_configs.append(task_config)
    
    return task_configs

def main():
    ap = argparse.ArgumentParser("Baseline: interleave (state,action), flatten, cluster (KMeans/HDBSCAN), then GAIL/AIRL per cluster; report NMI.")
    ap.add_argument("--env", choices=["Reacher-v4", "Pusher-v4", "Walker2d-v4","Hopper-v4","HalfCheetah-v5"], required=True)
    ap.add_argument("--ratio", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clusterer", choices=["kmeans", "hdbscan", "leiden"], default="kmeans")
    ap.add_argument("--k", type=int, default=6, help="K for KMeans (ignored for HDBSCAN)")
    ap.add_argument("--irl", choices=["gail", "airl"], default="gail")
    ap.add_argument("--rl-steps", type=int, default=250_000)
    ap.add_argument("--max-steps", type=int, default=0, help="T_max for interleaving (0: auto 95th pct)")
    ap.add_argument("--pad", type=float, default=0.0, help="Pad value for shorter trajectories")
    # New parallelization arguments
    ap.add_argument("--parallel", action="store_true", help="Enable parallel IRL training")
    ap.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: CPU count - 1)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); th.manual_seed(args.seed)
    num_trajs = 300 if args.env == "Walker2d-v4" or args.env == "Hopper-v4" or args.env == "HalfCheetah-v5" else 600

    RL_STEPS = 250_000 if args.env in ["Reacher-v4", "Pusher-v4"] else 750_000 if args.env == "Hopper-v4" else 500_000 if args.env == "HalfCheetah-v5" else 1_500_000

    # 1) Load expert trajectories
    trajectories, trajectories_with_rew, true_labels, modes = load_expert_set(args.env, num_trajs, args.ratio, args.seed)
    print(f"Loaded {len(trajectories)} trajectories for {args.env} (modes={modes}).")

    # 2) Build interleaved flattened features
    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    X_raw, meta = build_interleaved_matrix(trajectories, max_steps=max_steps, pad_value=args.pad)
    scaler = QuantileTransformer(
        n_quantiles=min(1000, X_raw.shape[0]),
        output_distribution='normal',
        random_state=args.seed,
    ).fit(X_raw)
    X = scaler.transform(X_raw)
    print(f"Interleaved shape per traj = {meta['vec_dim']} (steps={meta['max_steps']}, s_dim={meta['s_dim']}, a_dim={meta['a_dim']}).")

    # 3) Cluster
    K = args.k if args.clusterer == "kmeans" else None
    if args.clusterer == "kmeans" and K is None:
        K = (6 if args.env in ["Reacher-v4", "Pusher-v4", "Traj2d"] else 3)
    granularity = 0.075 if args.env == "Reacher-v4" else 0.01 if args.env == "Pusher-v4" else 0.015 if args.env == "Walker2d-v4" or args.env == "Hopper-v4" or args.env == "HalfCheetah-v5" else 0.1
    nX = len(X)
    min_cluster_size = max(5, int(granularity * nX))
    labels_pred, cinfo = cluster_features(X, args.clusterer, K or 0, args.seed, min_cluster_size)
    noise = int(cinfo.get("noise_count", 0))
    uniq = sorted(np.unique(labels_pred).tolist())
    print(f"Clusterer={args.clusterer} -> unique labels: {uniq} (noise={noise})")

    # 4) Compute clustering metrics (NMI, ARI, Silhouette)
    metrics = compute_clustering_metrics(X, labels_pred, true_labels)
    nmi = metrics['nmi']
    ari = metrics['ari']
    silhouette = metrics['silhouette']
    
    print(f"\n=== Clustering Metrics ===")
    print(f"  NMI:        {nmi:.4f}" if not np.isnan(nmi) else "  NMI:        N/A (degenerate labels)")
    print(f"  ARI:        {ari:.4f}" if not np.isnan(ari) else "  ARI:        N/A (degenerate labels)")
    print(f"  Silhouette: {silhouette:.4f}" if not np.isnan(silhouette) else "  Silhouette: N/A (insufficient clusters)")
    print(f"  Core samples: {metrics['n_core_samples']} | Noise samples: {metrics['n_noise_samples']}")

    # Prepare output dirs and CSV summary
    approach = _approach_name(args.clusterer, args.irl)
    k_path_segment = ""
    if args.clusterer == 'kmeans':
        k_val = args.k if args.k is not None else (6 if args.env in ["Reacher-v4", "Pusher-v4", "Traj2d"] else 3)
        k_path_segment = f"K_{k_val}"

    base_dir = os.path.join(f"learners_{approach}", args.env, f"seed_{args.seed}", k_path_segment)
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "summary.csv")

    # 5) Train IRL per cluster
    unique_clusters = sorted([c for c in np.unique(labels_pred).tolist() if c != -1])
    num_modes = 6 if args.env in ["Reacher-v4", "Pusher-v4", "Traj2d"] else 3

    # Prepare all task configurations
    task_configs = prepare_khgail_tasks(
        unique_clusters=unique_clusters,
        labels_pred=labels_pred,
        trajectories=trajectories,
        trajectories_with_rew=trajectories_with_rew,
        true_labels=true_labels,
        env_name=args.env,
        irl_algo=args.irl,
        seed=args.seed,
        rl_steps=RL_STEPS,
        base_dir=base_dir,
        approach=approach,
        clusterer=args.clusterer,
        requested_k=K,
        noise_count=noise,
        metrics=metrics,  # Pass full metrics dict
        meta=meta,
        num_modes=num_modes,
    )

    # Run IRL training (parallel or sequential)
    if args.parallel:
        print(f"\n[Mode: PARALLEL] Training {len(task_configs)} clusters with up to {args.max_workers or (mp.cpu_count() - 1)} workers")
        per_cluster_results = run_parallel_irl_khgail(
            task_configs=task_configs,
            max_workers=args.max_workers,
            verbose=True,
        )
    else:
        print(f"\n[Mode: SEQUENTIAL] Training {len(task_configs)} clusters one by one")
        per_cluster_results = run_sequential_irl_khgail(
            task_configs=task_configs,
            verbose=True,
        )

    # Filter successful results for reporting
    successful_results = [r for r in per_cluster_results if r.get('success', False)]
    failed_results = [r for r in per_cluster_results if not r.get('success', False)]

    # Print results for successful clusters
    for rec in successful_results:
        cid = rec['cluster_id']
        print(f"[C{cid}] n={rec['cluster_size']} dom_mode={rec['dominant_mode']} | saved to {base_dir}/C{cid}")
        print(f"  - agent: {rec['agent_path']}")
        print(f"  - policy_state_dict: {rec['policy_state_dict_path']}")
        print(f"  - reward_net: {rec['reward_net_path']}")
        print(f"  - reward_net_state_dict: {rec['reward_net_state_dict_path']}")

    # Report failures
    if failed_results:
        print(f"\n[WARNING] {len(failed_results)} cluster(s) failed:")
        for rec in failed_results:
            print(f"  - Cluster {rec.get('cluster_id', '?')}: {rec.get('error', 'unknown error')}")

    print("\n==== Summary ====")
    print(f"Env={args.env} | Clusterer={args.clusterer} | IRL={args.irl}")
    print(f"Clustering Metrics: NMI={nmi:.4f}" if not np.isnan(nmi) else 'N/A', " | "
          f"ARI={ari:.4f}" if not np.isnan(ari) else 'N/A', " | "
          f"Silhouette={silhouette:.4f}" if not np.isnan(silhouette) else 'N/A')
    print(f"Successful: {len(successful_results)}/{len(task_configs)} clusters")
    for r in successful_results:
        print(f"  C{r['cluster_id']}: n={r['cluster_size']} dom={r['dominant_mode']} | "
              f"learner_reward={r.get('learner_reward_mean', 'N/A'):.2f} ± {r.get('learner_reward_std', 'N/A'):.2f}")

    # Write CSV summary (append if exists) - only successful results
    if successful_results:
        # Remove internal keys before writing
        csv_results = []
        for r in successful_results:
            csv_row = {k: v for k, v in r.items() if k not in ['success', 'error']}
            csv_results.append(csv_row)
        
        header = list(csv_results[0].keys())
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            for row in csv_results:
                writer.writerow(row)
        print(f"Saved CSV -> {csv_path}")


if __name__ == "__main__":
    main()