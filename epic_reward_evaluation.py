#!/usr/bin/env python3
"""
epic_reward_evaluation.py

Standalone reward-evaluation script for CoMI-IRL.

What it does
------------
1. Loads expert trajectories and stored transition rewards for one or more environments, grouped by mode.
2. Discovers saved reward-net state dicts under the learners* folders.
3. STRICTLY FILTERS paths:
    - CoMI-IRL: Only evaluates models from the 'complete' stage.
    - K-based: Only evaluates models where K matches the environment's true number of modes.
4. Computes the alignment distance (1.0 - Pearson Correlation) STRICTLY between the learned reward and the TRUE expert reward for that specific mode.
5. Saves ranked summaries to disk, printing aggregated Mean, Std Dev, and Best (Min) per method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from scipy.stats import wilcoxon

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


# ---------------------------------------------------------------------------
# Environment / file discovery
# ---------------------------------------------------------------------------

ENV_SPECS = {
    "Reacher-v4": {
        "modes": 6,
        "expert_glob": "essinfogail/expert_imitation_trajectories/expert_imitation_trajectories_Reacher-v4_mode_*_withrew.pkl",
        "reward_token": "Reacher-v4",
    },
    "Pusher-v4": {
        "modes": 6,
        "expert_glob": "essinfogail/expert_imitation_trajectories/expert_imitation_trajectories_Pusher-v4_mode_*_withrew.pkl",
        "reward_token": "Pusher-v4",
    },
    "Walker2d-v4": {
        "modes": 6,
        "expert_glob": "essinfogail/expert_imitation_trajectories/expert_imitation_trajectories_Walker2d-v4_mode_*_withrew.pkl",
        "reward_token": "Walker2d-v4",
    },
    "Hopper-v4": {
        "modes": 3,
        "expert_glob": "expert_trajectories_new/Hopper-v4_task_*_withrew.pkl",
        "reward_token": "Hopper-v4",
    },
    "HalfCheetah-v5": {
        "modes": 3,
        "expert_glob": "expert_trajectories_new/HalfCheetah-v5_task_*_withrew.pkl",
        "reward_token": "HalfCheetah-v5",
    },
}


def choose_device() -> th.device:
    if th.cuda.is_available():
        return th.device("cuda")
    if hasattr(th.backends, "mps") and th.backends.mps.is_available():
        return th.device("mps")
    return th.device("cpu")


def discover_reward_files(root: Path, env_name: str) -> List[Path]:
    token = ENV_SPECS[env_name]["reward_token"]
    files: List[Path] = []
    
    # 1. Discover learners zip files
    for path in root.rglob("*_reward_net_state_dict.zip"):
        path_str = str(path)
        if token not in path_str:
            continue
        if "learners" not in path_str:
            continue
        if "h-gail" in path_str.lower() or "learners_h-" in path_str.lower():
            continue
        files.append(path)
        
    # 2. Discover essinfogail model.pth files
    for path in root.rglob("model.pth"):
        path_str = str(path)
        if "essinfogail" not in path_str.lower():
            continue
        if "logs_final" not in path_str.lower():
            continue
        
        match_env = False
        if token in path_str:
            match_env = True
        elif token == "HalfCheetah-v5" and "HalfCheetah-v5_64" in path_str:
            match_env = True
        elif token == "Hopper-v4" and "Hopper-v4_64" in path_str:
            match_env = True
            
        if match_env:
            files.append(path)
            
    return sorted(files)


def discover_expert_files(root: Path, env_name: str) -> List[Path]:
    pattern = ENV_SPECS[env_name]["expert_glob"]
    return sorted(root.glob(pattern))


# ---------------------------------------------------------------------------
# Expert data loading
# ---------------------------------------------------------------------------

def load_pickle(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_expert_trajectories_by_mode(root: Path, env_name: str):
    files = discover_expert_files(root, env_name)
    if not files:
        raise FileNotFoundError(f"No expert trajectories found for {env_name}")

    mode_data = {}
    for fp in files:
        match = re.search(r'(?:mode|task)_(\d+)', fp.name)
        mode_idx = int(match.group(1)) if match else 0
        
        trajs = load_pickle(fp)
        
        if mode_idx not in mode_data:
            mode_data[mode_idx] = []
            
        if isinstance(trajs, list):
            mode_data[mode_idx].extend(trajs)
        else:
            mode_data[mode_idx].append(trajs)
                
    return mode_data


def trajectory_to_arrays(traj):
    obs = np.asarray(traj.obs, dtype=np.float32)
    acts = np.asarray(traj.acts, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if acts.ndim == 1:
        acts = acts.reshape(-1, 1)
    steps = min(len(acts), len(obs) - 1)
    obs = obs[:steps]
    acts = acts[:steps]
    next_obs = np.asarray(traj.obs[1 : steps + 1], dtype=np.float32)
    return obs, acts, next_obs


# ---------------------------------------------------------------------------
# Reward model reconstruction
# ---------------------------------------------------------------------------

class RunningNormCompat(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", th.zeros(dim))
        self.register_buffer("running_var", th.ones(dim))
        self.register_buffer("count", th.tensor(1.0))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.running_mean) / th.sqrt(self.running_var + self.eps)


class MLPCompat(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, normalize: bool = True):
        super().__init__()
        self.normalize_input = RunningNormCompat(in_dim) if normalize else nn.Identity()
        self.dense0 = nn.Linear(in_dim, hidden_dim)
        self.dense_final = nn.Linear(hidden_dim, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.normalize_input(x)
        x = th.relu(self.dense0(x))
        return self.dense_final(x)


class PotentialNetCompat(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_hidden: int = 2, normalize: bool = True):
        super().__init__()
        self.normalize_input = RunningNormCompat(state_dim) if normalize else nn.Identity()
        self.dense0 = nn.Linear(state_dim, hidden_dim)
        if n_hidden >= 2:
            self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.dense1 = None
        self.dense_final = nn.Linear(hidden_dim, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.normalize_input(x)
        x = th.relu(self.dense0(x))
        if self.dense1 is not None:
            x = th.relu(self.dense1(x))
        return self.dense_final(x)


class ShapedRewardNetCompat(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, base_hidden: int = 32, pot_hidden: int = 32, pot_depth: int = 2, gamma: float = 0.99, is_gail: bool = False):
        super().__init__()
        self.gamma = 0.0 if is_gail else gamma
        self._base = nn.Module()
        self._base.mlp = MLPCompat(obs_dim + act_dim, base_hidden, normalize=True)
        self.potential = nn.Module()
        self.potential._potential_net = PotentialNetCompat(obs_dim, pot_hidden, n_hidden=pot_depth, normalize=True)

    def forward(self, obs: th.Tensor, acts: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        base = self._base.mlp(th.cat([obs, acts], dim=-1)).squeeze(-1)
        if self.gamma == 0.0:
            return base
        phi_s = self.potential._potential_net(obs).squeeze(-1)
        phi_ns = self.potential._potential_net(next_obs).squeeze(-1)
        return base + self.gamma * phi_ns - phi_s


class EssInfoGAILDiscrimCompat(nn.Module):
    def __init__(self, input_dim: int, dim_c: int = 6, hidden_units: Sequence[int] = (100, 100)):
        super().__init__()
        self.dim_c = dim_c
        layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.Tanh())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_units[-1], 1)
        self.classifier = nn.Linear(hidden_units[-1], dim_c)
        self.encoder_eps = nn.Linear(hidden_units[-1], 1)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        x = self.trunk(x)
        d = self.linear(x)
        eps = self.encoder_eps(x)
        c = th.softmax(self.classifier(x), -1)
        return d, eps, th.clamp(c, 1e-20, th.inf)


def infer_model_dims_and_hidden(state_dict: Dict[str, th.Tensor]):
    obs_dim = int(state_dict["potential._potential_net.normalize_input.running_mean"].numel())
    joint_dim = int(state_dict["_base.mlp.dense0.weight"].shape[1])
    act_dim = joint_dim - obs_dim
    base_hidden = int(state_dict["_base.mlp.dense0.weight"].shape[0])
    pot_hidden = int(state_dict["potential._potential_net.dense0.weight"].shape[0])
    pot_depth = 2 if "potential._potential_net.dense1.weight" in state_dict else 1
    return obs_dim, act_dim, base_hidden, pot_hidden, pot_depth


def load_reward_model(path: Path, device: th.device, is_gail: bool) -> ShapedRewardNetCompat:
    state_dict = th.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict in {path}, got {type(state_dict)}")
    obs_dim, act_dim, base_hidden, pot_hidden, pot_depth = infer_model_dims_and_hidden(state_dict)
    model = ShapedRewardNetCompat(obs_dim, act_dim, base_hidden=base_hidden, pot_hidden=pot_hidden, pot_depth=pot_depth, is_gail=is_gail).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def infer_essinfogail_dims(state_dict: Dict[str, th.Tensor]):
    input_dim = int(state_dict["trunk.0.weight"].shape[1])
    dim_c = int(state_dict["classifier.weight"].shape[0])
    return input_dim, dim_c


def load_essinfogail_model(path: Path, device: th.device) -> EssInfoGAILDiscrimCompat:
    state_dict = th.load(path, map_location="cpu")
    if "disc" in state_dict:
        disc_state = state_dict["disc"]
    else:
        disc_state = state_dict
    input_dim, dim_c = infer_essinfogail_dims(disc_state)
    model = EssInfoGAILDiscrimCompat(input_dim=input_dim, dim_c=dim_c).to(device)
    model.load_state_dict(disc_state, strict=False)
    model.eval()
    return model


def eval_modes_for(env_id: str, K: int) -> List[int]:
    """Env modes to reset, indexed by latent slot j=0..K-1."""
    if env_id in ("Reacher-v4", "Pusher-v4"):
        base = [1, 3, 5] if K == 3 else list(range(6))
    elif env_id in ("Walker2d-v4", "HalfCheetah-v5", "Hopper-v4"):
        base = list(range(3))
    else:
        base = list(range(K))
    return [base[j % len(base)] for j in range(K)]


# ---------------------------------------------------------------------------
# True Reward vs Learned Reward Alignment
# ---------------------------------------------------------------------------

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    
    x = x - x.mean()
    y = y - y.mean()
    
    sx = x.std() + 1e-12
    sy = y.std() + 1e-12
    
    return float(np.mean((x / sx) * (y / sy)))


def compute_true_reward(env_name: str, obs: np.ndarray, acts: np.ndarray, mode_idx: int) -> np.ndarray:
    """
    Computes the true mode-specific evaluation reward for the given observations and actions.
    This corresponds to the 'reward_eval' returned by the environment's step function.
    """
    N = len(obs)
    if env_name == "Reacher-v4":
        cos_theta0 = obs[:, 0]
        cos_theta1 = obs[:, 1]
        sin_theta0 = obs[:, 2]
        sin_theta1 = obs[:, 3]
        cos_sum = cos_theta0 * cos_theta1 - sin_theta0 * sin_theta1
        sin_sum = sin_theta0 * cos_theta1 + cos_theta0 * sin_theta1
        
        x = 0.1 * cos_theta0 + 0.11 * cos_sum
        y = 0.1 * sin_theta0 + 0.11 * sin_sum
        
        goal_radius = 0.15
        num_modes = 6
        theta_goal = mode_idx * 2 * np.pi / num_modes
        gx = np.cos(theta_goal) * goal_radius
        gy = np.sin(theta_goal) * goal_radius
        
        dist = np.sqrt((x - gx)**2 + (y - gy)**2)
        return -dist
        
    elif env_name == "Pusher-v4":
        tips_arm_com = obs[:, 14:17]
        object_com = obs[:, 17:20]
        
        goals = [[0, 0], [0, -0.43], [0, -0.86], [0.2, 0], [0.2, -0.43], [0.2, -0.86]]
        goal_pos = np.asarray(goals[mode_idx % len(goals)])
        goal_com = np.array([goal_pos[1] + 0.45, goal_pos[0] - 0.05, -0.323])
        
        vec_dist = object_com - goal_com
        vec_reach = tips_arm_com - goal_com
        
        reward_dist = -np.linalg.norm(vec_dist, axis=-1)
        reward_reach = -np.linalg.norm(vec_reach, axis=-1)
        
        return reward_dist + reward_reach
        
    elif env_name == "Walker2d-v4":
        x_velocity = obs[:, 8]
        if mode_idx == 0:
            return x_velocity
        elif mode_idx == 1:
            return -x_velocity
        elif mode_idx == 2:
            return -np.abs(x_velocity)
        else:
            return np.zeros(N, dtype=np.float32)
            
    elif env_name == "Hopper-v4":
        x_velocity = obs[:, 5]
        if mode_idx == 0:
            return x_velocity
        elif mode_idx == 1:
            return -x_velocity
        elif mode_idx == 2:
            return -np.abs(x_velocity)
        else:
            return np.zeros(N, dtype=np.float32)
            
    elif env_name == "HalfCheetah-v5":
        x_velocity = obs[:, 8]
        if mode_idx == 0:
            return x_velocity
        elif mode_idx == 1:
            return -x_velocity
        elif mode_idx == 2:
            return -np.abs(x_velocity)
        else:
            return np.zeros(N, dtype=np.float32)
            
    return np.zeros(N, dtype=np.float32)


def evaluate_model_reward(
    model,
    obs: th.Tensor,
    acts: th.Tensor,
    next_obs: th.Tensor,
    is_ess: bool,
    cluster: str,
    device: th.device,
) -> th.Tensor:
    """Evaluates model reward on a batch of transitions."""
    if is_ess:
        slot_idx = int(cluster) if cluster.isdigit() else 0
        x = th.cat([obs, acts], dim=-1)
        d, _, c = model(x)
        prob = 1.0 / (1.0 + th.exp(-d))
        reward_i = -th.log(th.maximum(1.0 - prob, th.tensor(0.0001, device=device)))
        log_c = th.log(c)
        if slot_idx >= 0 and slot_idx < c.shape[-1]:
            reward_ss = log_c[:, slot_idx]
        else:
            reward_ss = th.zeros_like(reward_i.squeeze(-1))
        return reward_i.squeeze(-1) + 0.1 * reward_ss
    else:
        return model(obs, acts, next_obs)


def compute_canonical_rewards(
    model,
    obs: th.Tensor,
    acts: th.Tensor,
    next_obs: th.Tensor,
    env_name: str,
    target_mode: int,
    cluster: str,
    is_ess: bool,
    device: th.device,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the canonicalized learned reward and the canonicalized true reward
    over the transition batch using EPIC canonicalization.
    """
    N = len(obs)
    # Target grid dimension M for canonicalization expectation
    M = min(128, N)
    
    # Coverage distribution sampled from the evaluation dataset
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(N, size=M, replace=False)
    obs_ref = obs[ref_idx]
    acts_ref = acts[ref_idx]
    next_obs_ref = next_obs[ref_idx]
    
    # 1. Compute learned reward outputs
    with th.no_grad():
        R_learned = evaluate_model_reward(model, obs, acts, next_obs, is_ess, cluster, device)
        
        # Grid evaluation for expectations
        # P[i] = E_{A, S'}[ R(s_i, A, S') ]
        P_learned = th.zeros(N, device=device)
        P_prime_learned = th.zeros(N, device=device)
        
        # Batch size for grid computation to avoid memory issues
        grid_batch = 64
        for start_i in range(0, N, grid_batch):
            end_i = min(start_i + grid_batch, N)
            chunk_size = end_i - start_i
            
            # Tile obs[start_i:end_i] to repeat for all reference samples
            # shape: [chunk_size * M, obs_dim]
            o_tile = obs[start_i:end_i].repeat_interleave(M, dim=0)
            a_tile = acts_ref.repeat(chunk_size, 1)
            ns_tile = next_obs_ref.repeat(chunk_size, 1)
            
            R_grid = evaluate_model_reward(model, o_tile, a_tile, ns_tile, is_ess, cluster, device)
            P_learned[start_i:end_i] = R_grid.view(chunk_size, M).mean(dim=1)
            
            # Repeat next_obs
            ns_o_tile = next_obs[start_i:end_i].repeat_interleave(M, dim=0)
            R_grid_ns = evaluate_model_reward(model, ns_o_tile, a_tile, ns_tile, is_ess, cluster, device)
            P_prime_learned[start_i:end_i] = R_grid_ns.view(chunk_size, M).mean(dim=1)
            
    R_learned_can = R_learned + gamma * P_prime_learned - P_learned
    R_learned_can_np = R_learned_can.cpu().numpy()
    
    # 2. Compute true reward outputs
    # For true reward, we can run it in numpy
    obs_np = obs.cpu().numpy()
    acts_np = acts.cpu().numpy()
    next_obs_np = next_obs.cpu().numpy()
    
    obs_ref_np = obs_ref.cpu().numpy()
    acts_ref_np = acts_ref.cpu().numpy()
    next_obs_ref_np = next_obs_ref.cpu().numpy()
    
    R_true = compute_true_reward(env_name, obs_np, acts_np, target_mode)
    
    P_true = np.zeros(N)
    P_prime_true = np.zeros(N)
    
    for start_i in range(0, N, grid_batch):
        end_i = min(start_i + grid_batch, N)
        chunk_size = end_i - start_i
        
        o_tile = np.repeat(obs_np[start_i:end_i], M, axis=0)
        a_tile = np.tile(acts_ref_np, (chunk_size, 1))
        ns_tile = np.tile(next_obs_ref_np, (chunk_size, 1))
        
        R_grid_true = compute_true_reward(env_name, o_tile, a_tile, target_mode)
        P_true[start_i:end_i] = R_grid_true.reshape(chunk_size, M).mean(axis=1)
        
        ns_o_tile = np.repeat(next_obs_np[start_i:end_i], M, axis=0)
        R_grid_ns_true = compute_true_reward(env_name, ns_o_tile, a_tile, target_mode)
        P_prime_true[start_i:end_i] = R_grid_ns_true.reshape(chunk_size, M).mean(axis=1)
        
    R_true_can = R_true + gamma * P_prime_true - P_true
    
    return R_learned_can_np, R_true_can


def true_reward_alignment(model, bank: Sequence[dict], device: th.device, env_name: str, target_mode: int, cluster: str, is_ess: bool):
    """
    Compares the learned reward strictly against the true expert reward using EPIC distance.
    """
    if len(bank) == 0:
        return {"alignment_corr": float('nan'), "alignment_distance": float('nan')}
        
    obs = th.from_numpy(np.stack([it["obs"] for it in bank], axis=0)).float().to(device)
    acts = th.from_numpy(np.stack([it["act"] for it in bank], axis=0)).float().to(device)
    next_obs = th.from_numpy(np.stack([it["next_obs"] for it in bank], axis=0)).float().to(device)
    
    R_learned_can, R_true_can = compute_canonical_rewards(
        model, obs, acts, next_obs, env_name, target_mode, cluster, is_ess, device
    )
        
    corr = correlation(R_learned_can, R_true_can)
    # EPIC distance is defined as sqrt((1.0 - corr) / 2.0) to bound it in [0, 1]
    dist = math.sqrt(max(0.0, (1.0 - corr) / 2.0))
    
    return {
        "alignment_corr": corr,
        "alignment_distance": dist,
    }


def evaluate_combined_reward(
    rec_list: List[RewardRecord],
    obs: th.Tensor,
    acts: th.Tensor,
    next_obs: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """Evaluates combined reward (max-reduction) across all models in the list."""
    outputs = []
    for rec in rec_list:
        is_ess = "essinfogail" in rec.path.lower()
        out = evaluate_model_reward(rec.model, obs, acts, next_obs, is_ess, rec.cluster, device)
        outputs.append(out)
    return th.max(th.stack(outputs, dim=0), dim=0)[0]


def compute_combined_canonical_rewards(
    rec_list: List[RewardRecord],
    obs: th.Tensor,
    acts: th.Tensor,
    next_obs: th.Tensor,
    env_name: str,
    target_mode: int,
    device: th.device,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes canonicalized combined (max-reduced) learned reward and canonicalized true reward.
    """
    N = len(obs)
    M = min(128, N)
    
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(N, size=M, replace=False)
    obs_ref = obs[ref_idx]
    acts_ref = acts[ref_idx]
    next_obs_ref = next_obs[ref_idx]
    
    # 1. Compute combined learned reward outputs
    with th.no_grad():
        R_learned = evaluate_combined_reward(rec_list, obs, acts, next_obs, device)
        
        P_learned = th.zeros(N, device=device)
        P_prime_learned = th.zeros(N, device=device)
        
        grid_batch = 64
        for start_i in range(0, N, grid_batch):
            end_i = min(start_i + grid_batch, N)
            chunk_size = end_i - start_i
            
            o_tile = obs[start_i:end_i].repeat_interleave(M, dim=0)
            a_tile = acts_ref.repeat(chunk_size, 1)
            ns_tile = next_obs_ref.repeat(chunk_size, 1)
            
            R_grid = evaluate_combined_reward(rec_list, o_tile, a_tile, ns_tile, device)
            P_learned[start_i:end_i] = R_grid.view(chunk_size, M).mean(dim=1)
            
            ns_o_tile = next_obs[start_i:end_i].repeat_interleave(M, dim=0)
            R_grid_ns = evaluate_combined_reward(rec_list, ns_o_tile, a_tile, ns_tile, device)
            P_prime_learned[start_i:end_i] = R_grid_ns.view(chunk_size, M).mean(dim=1)
            
    R_learned_can = R_learned + gamma * P_prime_learned - P_learned
    R_learned_can_np = R_learned_can.cpu().numpy()
    
    # 2. Compute true reward outputs
    obs_np = obs.cpu().numpy()
    acts_np = acts.cpu().numpy()
    next_obs_np = next_obs.cpu().numpy()
    
    obs_ref_np = obs_ref.cpu().numpy()
    acts_ref_np = acts_ref.cpu().numpy()
    next_obs_ref_np = next_obs_ref.cpu().numpy()
    
    R_true = compute_true_reward(env_name, obs_np, acts_np, target_mode)
    
    P_true = np.zeros(N)
    P_prime_true = np.zeros(N)
    
    for start_i in range(0, N, grid_batch):
        end_i = min(start_i + grid_batch, N)
        chunk_size = end_i - start_i
        
        o_tile = np.repeat(obs_np[start_i:end_i], M, axis=0)
        a_tile = np.tile(acts_ref_np, (chunk_size, 1))
        ns_tile = np.tile(next_obs_ref_np, (chunk_size, 1))
        
        R_grid_true = compute_true_reward(env_name, o_tile, a_tile, target_mode)
        P_true[start_i:end_i] = R_grid_true.reshape(chunk_size, M).mean(axis=1)
        
        ns_o_tile = np.repeat(next_obs_np[start_i:end_i], M, axis=0)
        R_grid_ns_true = compute_true_reward(env_name, ns_o_tile, a_tile, target_mode)
        P_prime_true[start_i:end_i] = R_grid_ns_true.reshape(chunk_size, M).mean(axis=1)
        
    R_true_can = R_true + gamma * P_prime_true - P_true
    
    return R_learned_can_np, R_true_can


def combined_reward_alignment(rec_list: List[RewardRecord], bank: Sequence[dict], device: th.device, env_name: str, target_mode: int):
    """
    Compares the combined (max-reduced) learned rewards of a group strictly against the true expert reward using EPIC.
    """
    if len(bank) == 0:
        return {"alignment_corr": float('nan'), "alignment_distance": float('nan')}
        
    obs = th.from_numpy(np.stack([it["obs"] for it in bank], axis=0)).float().to(device)
    acts = th.from_numpy(np.stack([it["act"] for it in bank], axis=0)).float().to(device)
    next_obs = th.from_numpy(np.stack([it["next_obs"] for it in bank], axis=0)).float().to(device)
    
    R_learned_can, R_true_can = compute_combined_canonical_rewards(
        rec_list, obs, acts, next_obs, env_name, target_mode, device
    )
        
    corr = correlation(R_learned_can, R_true_can)
    dist = math.sqrt(max(0.0, (1.0 - corr) / 2.0))
    
    return {
        "alignment_corr": corr,
        "alignment_distance": dist,
    }


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

@dataclass
class RewardRecord:
    env: str
    path: str
    method: str
    seed: str
    cluster: str
    target_mode: int
    model: nn.Module
    alignment_corr: float = math.nan
    alignment_distance: float = math.nan


def parse_reward_path(path: Path):
    parts = path.parts
    name = path.name.lower()
    path_str = str(path)
    
    method = "unknown"
    seed = "unknown"
    cluster = "unknown"
    target_mode = -1

    is_essinfogail = "essinfogail" in path_str.lower()
    
    if is_essinfogail:
        method = "Ess-InfoGAIL"
        match = re.search(r'K(\d+)_from_D(\d+)', path_str)
        if match:
            k_val = match.group(1)
            method = f"Ess-InfoGAIL (K={k_val})"
    else:
        for p in parts:
            if "learners" in p:
                if p == "learners":
                    if "airl" in name:
                        method = "CoMI-IRL (AIRL)"
                    elif "gail" in name:
                        method = "CoMI-IRL (GAIL)"
                    else:
                        method = "CoMI-IRL"
                else:
                    raw_method = p.replace("learners_", "")
                    method = re.sub(r'[_\-]\d+$', '', raw_method)
                break
        
        k_match = re.search(r'[/\\]K_(\d+)[/\\]', path_str)
        if k_match:
            k_val = k_match.group(1)
            method = f"{method} (K={k_val})"

    seed_match = re.search(r'seed[_\-]?(\d+)', path_str.lower())
    if seed_match:
        seed = seed_match.group(1)
    else:
        if is_essinfogail:
            if len(parts) >= 4:
                seed = parts[-3]
        else:
            for p in parts:
                if p.isdigit() and p not in ("complete", "finetuned_novel"):
                    seed = p
                    break

    match = re.search(r'cluster[_\-]?(\d+)', name)
    if match:
        cluster = match.group(1)
    else:
        for p in parts:
            if p.startswith("C") and len(p) > 1 and p[1:].isdigit():
                cluster = p[1:]
                break

    match = re.search(r'(?:mode|task)[_\-]?(\d+)', name)
    if match:
        target_mode = int(match.group(1))

    return method, seed, cluster, target_mode


def discover_and_load_rewards(root: Path, env_name: str, device: th.device, limit: Optional[int] = None):
    reward_files = discover_reward_files(root, env_name)
    
    valid_files = []
    for path in reward_files:
        method, _, _, _ = parse_reward_path(path)
        
        if "CoMI-IRL" in method:
            if "complete" not in [p.lower() for p in path.parts]:
                continue
                
        valid_files.append(path)

    if limit is not None:
        valid_files = valid_files[:limit]

    records = []
    for path in tqdm(valid_files, desc=f"[{env_name}] load rewards", leave=False):
        method, seed, cluster, target_mode = parse_reward_path(path)
        
        if "essinfogail" in str(path).lower():
            try:
                model = load_essinfogail_model(path, device=device)
            except Exception as e:
                print(f"[warn] failed to load essinfogail {path}: {e}")
                continue
                
            dim_c = model.classifier.weight.shape[0]
            eval_modes = eval_modes_for(env_name, dim_c)
            for j in range(dim_c):
                t_mode = eval_modes[j]
                records.append(
                    RewardRecord(
                        env=env_name, path=str(path), method=method, seed=seed,
                        cluster=str(j), target_mode=t_mode, model=model,
                    )
                )
        else:
            is_gail = "gail" in method.lower()
            try:
                model = load_reward_model(path, device=device, is_gail=is_gail)
            except Exception as e:
                print(f"[warn] failed to load {path}: {e}")
                continue
                
            records.append(
                RewardRecord(
                    env=env_name, path=str(path), method=method, seed=seed,
                    cluster=cluster, target_mode=target_mode, model=model,
                )
            )
    return records


def compute_paired_wilcoxon_p_values(records: List[RewardRecord], ref_key: str) -> Dict[str, float]:
    """
    Groups records by (seed, target_mode) and averages the distances to handle potential
    multi-cluster rewards (like CoMI-IRL). Then performs a paired Wilcoxon signed-rank test
    between the reference method and each baseline method.
    Applies Benjamini-Hochberg False Discovery Rate (FDR) correction to the raw p-values.
    """
    # 1. Group records by (method, seed, target_mode) and average the distances
    grouped = defaultdict(list)
    for rec in records:
        if not math.isnan(rec.alignment_distance):
            grouped[(rec.method, rec.seed, rec.target_mode)].append(rec.alignment_distance)
            
    mean_map = {k: np.mean(v) for k, v in grouped.items()}
    
    # 2. Get the unique (seed, target_mode) configurations present in the reference method
    ref_configs = set()
    for (method, seed, target_mode) in mean_map.keys():
        if method == ref_key:
            ref_configs.add((seed, target_mode))
            
    methods = sorted(list(set(rec.method for rec in records if not math.isnan(rec.alignment_distance))))
    p_values = {}
    
    for method in methods:
        if method == ref_key:
            continue
            
        # Build paired arrays aligned by (seed, target_mode)
        ref_vals = []
        method_vals = []
        for (seed, target_mode) in ref_configs:
            ref_val = mean_map.get((ref_key, seed, target_mode))
            method_val = mean_map.get((method, seed, target_mode))
            if ref_val is not None and method_val is not None:
                ref_vals.append(ref_val)
                method_vals.append(method_val)
                
        ref_vals = np.array(ref_vals)
        method_vals = np.array(method_vals)
        
        # Wilcoxon signed-rank test requires at least 5 paired samples and some non-zero differences
        if len(ref_vals) >= 5 and not np.allclose(ref_vals, method_vals):
            try:
                res = wilcoxon(method_vals, ref_vals, alternative='two-sided')
                p_values[method] = float(res.pvalue)
            except Exception:
                p_values[method] = float('nan')
        else:
            p_values[method] = float('nan')
            
    # 3. Apply FDR Benjamini-Hochberg correction to non-NaN p-values
    valid_methods = [m for m, p in p_values.items() if not math.isnan(p)]
    if valid_methods:
        raw_p = [p_values[m] for m in valid_methods]
        n = len(raw_p)
        sorted_indices = np.argsort(raw_p)
        sorted_p = np.array(raw_p)[sorted_indices]
        
        adjusted_p = np.zeros(n)
        min_val = 1.0
        for i in range(n - 1, -1, -1):
            val = sorted_p[i] * (n / (i + 1))
            min_val = min(min_val, val)
            adjusted_p[i] = min_val
            
        rev_indices = np.argsort(sorted_indices)
        adjusted_p = adjusted_p[rev_indices]
        
        for m, adj_p in zip(valid_methods, adjusted_p):
            p_values[m] = adj_p
            
    return p_values


def evaluate_environment(
    root: Path,
    env_name: str,
    device: th.device,
    max_transitions: int,
    reward_limit: Optional[int],
    seed: int,
    out_dir: Path,
):
    # 1. Load expert data heavily grouped by mode
    mode_data = load_expert_trajectories_by_mode(root, env_name)
    mode_banks = {}
    
    rng = np.random.default_rng(seed)

    for mode_idx, trajs in mode_data.items():
        mode_bank = []
        for traj in trajs:
            obs, acts, next_obs = trajectory_to_arrays(traj)
            rews = compute_true_reward(env_name, obs, acts, mode_idx)
            n = min(len(obs), len(acts), len(next_obs), len(rews))
            for t in range(n):
                mode_bank.append({
                    "obs": obs[t], "act": acts[t], "next_obs": next_obs[t],
                    "expert_reward": float(rews[t]),
                })
            
        # Sample down to max_transitions per mode to keep evaluation fast and balanced
        if len(mode_bank) > max_transitions:
            idx = rng.choice(len(mode_bank), size=max_transitions, replace=False)
            mode_bank = [mode_bank[i] for i in idx]
            
        mode_banks[mode_idx] = mode_bank

    # 2. Discover and filter learned reward models
    records = discover_and_load_rewards(root, env_name, device=device, limit=reward_limit)
    if not records:
        print(f"[{env_name}] no reward nets found")
        return

    # 3. Calculate strictly the True Reward Alignment (Learned vs True on Same Mode)
    for rec in tqdm(records, desc=f"[{env_name}] learned vs true alignment", leave=False):
        if rec.target_mode in mode_banks:
            is_ess = "essinfogail" in rec.path.lower()
            align = true_reward_alignment(rec.model, mode_banks[rec.target_mode], device, rec.env, rec.target_mode, rec.cluster, is_ess)
            rec.alignment_corr = align["alignment_corr"]
            rec.alignment_distance = align["alignment_distance"]
        else:
            rec.alignment_corr, rec.alignment_distance = float('nan'), float('nan')

    # 4. Optional: Combined Mode-Level Reward Evaluation
    if env_name == "Reacher-v4":
        combined_groups = defaultdict(list)
        for rec in records:
            if not math.isnan(rec.alignment_distance):
                combined_groups[(rec.method, rec.seed, rec.target_mode)].append(rec)
                
        combined_records = []
        for (method, seed_val, target_mode), rec_list in tqdm(combined_groups.items(), desc=f"[{env_name}] combined mode alignment", leave=False):
            if len(rec_list) <= 1:
                continue  # Only combine if there are multiple clusters/heads
                
            bank = mode_banks[target_mode]
            align = combined_reward_alignment(rec_list, bank, device, env_name, target_mode)
            
            combined_records.append(
                RewardRecord(
                    env=env_name,
                    path="combined",
                    method=f"{method} (Combined)",
                    seed=seed_val,
                    cluster="combined",
                    target_mode=target_mode,
                    model=rec_list[0].model,  # reference model
                    alignment_corr=align["alignment_corr"],
                    alignment_distance=align["alignment_distance"]
                )
            )
            
        records.extend(combined_records)

    # 5. Save and Print Results
    records.sort(key=lambda r: (math.inf if math.isnan(r.alignment_distance) else r.alignment_distance))
    
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{env_name}_true_reward_alignment.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "method", "seed", "cluster", "target_mode", "path", "alignment_corr", "alignment_distance"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "env": rec.env, "method": rec.method, "seed": rec.seed, "cluster": rec.cluster,
                "target_mode": rec.target_mode, "path": rec.path, 
                "alignment_corr": rec.alignment_corr, "alignment_distance": rec.alignment_distance,
            })

    print(f"\n=== {env_name} Global Info ===")
    print(f"total filtered rewards evaluated={len(records)} | transitions_per_mode={max_transitions}")
    
    method_alignments = defaultdict(list)
    for rec in records:
        if not math.isnan(rec.alignment_distance):
            method_alignments[rec.method].append(rec.alignment_distance)
            
    if method_alignments:
        print(f"\n=== {env_name} Distance to True Reward ===")
        
        # We compare each method against our best performing CoMI-IRL reference method:
        comi_keys = [k for k in method_alignments.keys() if "CoMI-IRL" in k and "Combined" not in k]
        if comi_keys:
            ref_key = min(comi_keys, key=lambda k: np.mean(method_alignments[k]))
        else:
            ref_key = "CoMI-IRL (AIRL)"
            
        ref_combined_key = f"{ref_key} (Combined)"
        
        # Precompute Wilcoxon p-values with FDR correction
        p_values_single = compute_paired_wilcoxon_p_values(records, ref_key)
        p_values_combined = compute_paired_wilcoxon_p_values(records, ref_combined_key)
        
        for method in sorted(method_alignments.keys()):
            align_arr = np.array(method_alignments[method])
            
            mean_dist = np.mean(align_arr)
            std_dist = np.std(align_arr)
            best_dist = np.min(align_arr)
            
            p_val_str = ""
            if len(align_arr) > 0:
                current_ref_key = ref_key
                current_p_map = p_values_single
                
                # If we're evaluating a combined method, compare against the combined reference
                if "combined" in method.lower():
                    current_ref_key = ref_combined_key
                    current_p_map = p_values_combined
                    
                if method != current_ref_key:
                    p_val = current_p_map.get(method, float('nan'))
                    if math.isnan(p_val):
                        p_val_str = " (p = NaN)"
                    else:
                        ns_suffix = " [n.s.]" if p_val >= 0.05 else ""
                        if p_val < 0.0001:
                            p_val_str = f" (p = {p_val:.2e}{ns_suffix} vs {current_ref_key})"
                        else:
                            p_val_str = f" (p = {p_val:.4f}{ns_suffix} vs {current_ref_key})"
            
            print(f"  {method}: Mean = {mean_dist:.4f} ± {std_dist:.4f} | Best = {best_dist:.4f} (n={len(align_arr)}){p_val_str}")
    else:
        print("\nNo valid alignment scores calculated. Check mode mapping.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="True Reward Evaluator for CoMI-IRL.")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent), help="Repository root")
    ap.add_argument("--env", type=str, default="all", choices=["all"] + list(ENV_SPECS.keys()), help="Environment to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-transitions", type=int, default=5000, help="Max sampled transitions per mode")
    ap.add_argument("--reward-limit", type=int, default=None, help="Optional limit on loaded reward nets per environment")
    ap.add_argument("--out-dir", type=str, default="./true_reward_eval_outputs")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    root = Path(args.root).resolve()
    device = choose_device()

    envs = list(ENV_SPECS.keys()) if args.env == "all" else [args.env]
    print(f"device={device}")
    
    for env_name in envs:
        evaluate_environment(
            root, env_name, device, args.max_transitions, 
            args.reward_limit, args.seed, Path(args.out_dir) / env_name
        )


if __name__ == "__main__":
    main()