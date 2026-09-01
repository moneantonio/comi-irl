#!/usr/bin/env python3
"""
Expert & Agent Deployment and Visualization.

Deploys and renders trained expert policies, CoMI-IRL learners, and
geometric consensus agents across all supported environments.

Supports:
  - Reacher-v4, Pusher-v4 (legacy essinfogail envs)
  - Walker2d-v4 (legacy essinfogail env)
  - Hopper-v4, BipedalWalker-v3 (gymnasium envs with reward shaping)

Usage:
  # Render expert policies for Reacher
  python gc_deploy.py --env_id Reacher-v4 --agents expert --seeds 42 --render

  # Compare expert vs consensus agents for Pusher
  python gc_deploy.py --env_id Pusher-v4 --agents expert consensus --seeds 42 --render

  # Deploy Hopper experts (from gc_create_experts)
  python gc_deploy.py --env_id Hopper-v4 --agents expert --seeds 42 --render

  # Deploy BipedalWalker experts (same task, different styles)
  python gc_deploy.py --env_id BipedalWalker-v3 --agents expert --seeds 42 --render

  # Full comparison without rendering (just metrics)
  python gc_deploy.py --env_id Reacher-v4 --agents expert learner consensus --tr_names airl --seeds 42 0 1

  # List available configs for an environment
  python gc_deploy.py --env_id Hopper-v4 --list_configs
"""

import os
import re
import glob
import time
import argparse
import pickle
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from stable_baselines3 import PPO, SAC  # type: ignore[import]

# ═══════════════════════════════════════════════════════════════
# Import environment registries
# ═══════════════════════════════════════════════════════════════

# Legacy essinfogail environments
try:
    from envs import MultimodalEnvs
    from envs.env_norm import make_env
    from gail.algo import SACExpert
    HAS_ESSINFOGAIL = True
except ImportError:
    HAS_ESSINFOGAIL = False
    MultimodalEnvs = {}

# New gymnasium environments (Hopper, BipedalWalker)
# import mo_gymnasium as mo_gym # type: ignore[import]
# from mo_gymnasium.wrappers import LinearReward # type: ignore[import]
# try:
import gymnasium as gym  # type: ignore[import]
from new_create_experts import (
    ENV_SPECS, SHAPER_REGISTRY, ShapedRewardWrapper,
    ConfigInfo, EnvSpec, Trajectory, TrajectoryWithRew,
)
HAS_GC_EXPERTS = True
# except ImportError:
#     HAS_GC_EXPERTS = False
#     ENV_SPECS = {}
#     SHAPER_REGISTRY = {}


# ═══════════════════════════════════════════════════════════════
# Environment Configuration
# ═══════════════════════════════════════════════════════════════

# Map env_id to whether it's a legacy or gymnasium environment
LEGACY_ENVS = {"Reacher-v4", "Pusher-v4", "Walker2d-v4"}
GYMNASIUM_ENVS = {"Hopper-v4", "BipedalWalker-v3", "LunarLanderContinuous-v3","HalfCheetah-v5","Swimmer-v5","Walker2d-v5","mo-halfcheetah-v5"}

# Default number of modes (tasks) per environment
DEFAULT_N_MODES = {
    "Reacher-v4": 6,
    "Pusher-v4": 6,
    "Walker2d-v4": 3,
    "Hopper-v4": 3,        # 3 modes (forward, backward, stand)
    "BipedalWalker-v3": 1,  # 1 mode (forward)
    "LunarLanderContinuous-v3": 1,  # 1 mode (landing)
    "HalfCheetah-v5": 3,  # 3 modes (forward, backward, stand)
    "Swimmer-v5": 3,  # 3 modes (forward, backward, still)
    "Walker2d-v5": 3,  # 3 modes (forward, backward, stand)
    "mo-halfcheetah-v5": 3,  
}

# Default number of configs (mode × style) per environment
DEFAULT_N_CONFIGS = {
    "Reacher-v4": 12,
    "Pusher-v4": 6,
    "Walker2d-v4": 3,
    "Hopper-v4": 3,
    "BipedalWalker-v3": 4,  # 1 mode × 3 styles (DEPRECATED)
    "LunarLanderContinuous-v3": 3,  # 1 mode × 3 style (left, right, center) (DEPRECATED)
    "HalfCheetah-v5": 3,  # 1 mode × 3 styles
    "Swimmer-v5": 5,  # 2x2+1 (forward/backward × 2 styles + 1 neutral) (DEPRECATED)
    "Walker2d-v5": 5,  # 3 modes × 2 styles (forward/backward/stand × normal/low-torque) (DEPRECATED)
    "mo-halfcheetah-v5": 5,  # 3 mode × {2,2,1} styles (forward with different forward/control weightings)
}

# Max episode steps per environment
DEFAULT_MAX_STEPS = {
    "Reacher-v4": 50,
    "Pusher-v4": 100,
    "Walker2d-v4": 200,
    "Hopper-v4": 200,#500
    "BipedalWalker-v3": 1600,
    "LunarLanderContinuous-v3": 500,
    "HalfCheetah-v5": 100,#200
    "Swimmer-v5": 500,
    "Walker2d-v5": 500,
    "mo-halfcheetah-v5": 500,
}

# Behavior index for style analysis (obs dimension that distinguishes style)
BEHAVIOR_INDEX = {
    "Reacher-v4": 3,       # sin(theta2)
    "Pusher-v4": 3,
    "Walker2d-v4": 0,
    "Hopper-v4": 0,        # z_pos (height)
    "BipedalWalker-v3": 4,  # hip1_angle
    "LunarLanderContinuous-v3": 0,  # angle (to distinguish landing styles)
    "HalfCheetah-v5": 0,  # angle (to distinguish running styles)
    "Swimmer-v5": 0,  # angle (to distinguish forward/backward styles)
    "Walker2d-v5": 0,  # z_pos (height)
    "mo-halfcheetah-v5": 0,  # angle (to distinguish running styles)
}

# Environment structure type for reporting
ENV_STRUCTURE = {
    "Reacher-v4": "multi_task_multi_style",    # 6 tasks × 2 styles
    "Pusher-v4": "multi_task_single_style",    # 6 tasks × 1 style
    "Walker2d-v4": "multi_task_single_style",  # 3 tasks × 1 style
    "Hopper-v4": "multi_task_nonuniform",      # 3 tasks × {2,2,1} styles
    "BipedalWalker-v3": "single_task_multi_style",  # 1 task × 3 styles
    "LunarLanderContinuous-v3": "single_task_multi_style",  # 1 task × 3 styles
    "HalfCheetah-v5": "single_task_multi_style",  # 1 task × 3 styles
    "Swimmer-v5": "multi_task_nonuniform",  # 2 tasks × 2 styles + 1 neutral
    "Walker2d-v5": "multi_task_nonuniform",  # 3 tasks × {2,2,1} styles
    "mo-halfcheetah-v5": "multi_task_nonuniform",  # 3 tasks × {2,2,1} styles
}


# ═══════════════════════════════════════════════════════════════
# Environment Creation
# ═══════════════════════════════════════════════════════════════

def create_env(env_id: str, n_modes: int, config_id: int = None,
               mode_idx: int = None, render: bool = False):
    """
    Create an environment instance.

    For legacy envs: uses essinfogail's MultimodalEnvs + make_env.
    For gymnasium envs: uses gym.make with optional reward shaping.

    Args:
        env_id: Environment identifier.
        n_modes: Number of modes in the environment.
        config_id: Configuration ID (for gymnasium envs with reward shaping).
        mode_idx: Mode index to fix the environment to.
        render: Whether to enable rendering.

    Returns:
        env: The environment instance.
        is_legacy: Whether this is a legacy (old gym) environment.
    """

    if config_id is not None and env_id in SHAPER_REGISTRY:
        shaper_class = SHAPER_REGISTRY[env_id]
        cfg_info = ENV_SPECS[env_id].configs[config_id]
        shaper = shaper_class(
            config_id=config_id,
            mode_id=cfg_info.mode_id,
            style_id=cfg_info.style_id,
        )

    if env_id in LEGACY_ENVS:
        if not HAS_ESSINFOGAIL:
            raise ImportError(
                f"Cannot create {env_id}: essinfogail environment package not found. "
                f"Make sure 'envs' and 'gail' are importable."
            )
        EnvClass = MultimodalEnvs[env_id]
        raw_env = EnvClass(num_modes=n_modes)
        env = make_env(raw_env)
        return env, True

    elif env_id in GYMNASIUM_ENVS:
        render_mode = "human" if render else None
        if env_id == "mo-halfcheetah-v5":
            # Special handling for mo-halfcheetah-v5
            # base_env = mo_gym.make(env_id,max_episode_steps=DEFAULT_MAX_STEPS[env_id],render_mode=render_mode)
            # base_env = LinearReward(base_env, weight=np.array([shaper.CONFIGS[config_id].get("forward_weight", 1.0), shaper.CONFIGS[config_id].get("cost_weight", 1.0)], dtype=np.float32))
            pass
        elif env_id == "Hopper-v4" or env_id == "Walker2d-v5":
            base_env = gym.make(env_id, max_episode_steps=DEFAULT_MAX_STEPS[env_id], terminate_when_unhealthy=False, render_mode=render_mode)
        else:
            base_env = gym.make(env_id, render_mode=render_mode)

        # Apply reward shaping if config_id is provided
        if config_id is not None and env_id in SHAPER_REGISTRY:
            env = ShapedRewardWrapper(base_env, shaper)
        else:
            env = base_env

        return env, False

    else:
        raise ValueError(f"Unknown environment: {env_id}")


def reset_env(env, mode_idx: int, is_legacy: bool, seed: int = None):
    """
    Reset environment to a specific mode.

    For legacy envs: calls env.reset(mode_idx=mode_idx).
    For gymnasium envs: calls env.reset(seed=seed).
    """
    if is_legacy:
        out = env.reset(mode_idx=mode_idx)
        return out[0] if isinstance(out, tuple) else out
    else:
        obs, info = env.reset(seed=seed)
        return obs


# ═══════════════════════════════════════════════════════════════
# Agent Loading
# ═══════════════════════════════════════════════════════════════

def load_expert_legacy(env, env_id: str, n_modes: int, mode_idx: int,
                       device: torch.device):
    """Load legacy SAC expert from weights/{env_id}_{n_modes}_modes/{mode_idx}.pth"""
    expert_n_modes = 6 if env_id in ["Reacher-v4", "Pusher-v4"] else 3
    weight_path = f"weights/{env_id}_{expert_n_modes}_modes/{mode_idx}.pth"
    if not os.path.exists(weight_path):
        print(f"  Warning: Expert weights not found at {weight_path}")
        return None
    return SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=weight_path,
    )


def load_expert_gymnasium(env_id: str, config_id: int, seed: int = 42,
                          policy_dir: str = "./expert_policies_new"):
    """Load PPO expert trained by gc_create_experts."""
    policy_path = os.path.join(policy_dir, f"{env_id}_config_{config_id}_seed_{seed}")
    if not os.path.exists(policy_path + ".zip"):
        print(f"  Warning: Expert policy not found at {policy_path}.zip")
        return None
    return PPO.load(policy_path)


class BNIRLPolicyWrapper:
    def __init__(self, mode, checkpoint_path, env_name, device="cpu"):
        self.mode = mode
        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.state_dim = self.checkpoint["state_dim"]
        self.action_dim = self.checkpoint["action_dim"]
        
        # Import stress test classes dynamically
        import sys
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        from deep_dp_birl_stress_test import (
            DeepRewardNetwork, LocalControllerNetwork,
            ProbabilisticDynamicsModel, SoftPlanner
        )
        
        if mode == "bnirl_subgoal":
            self.policy_net = LocalControllerNetwork(self.state_dim, self.action_dim).to(self.device)
            self.policy_net.load_state_dict(self.checkpoint["reward_state_dict"])
            self.policy_net.eval()
        else:
            self.reward_net = DeepRewardNetwork(self.state_dim, self.action_dim).to(self.device)
            self.reward_net.load_state_dict(self.checkpoint["reward_state_dict"])
            self.reward_net.eval()
            
            self.dynamics = ProbabilisticDynamicsModel(self.state_dim, self.action_dim).to(self.device)
            self.dynamics.load_state_dict(self.checkpoint["dynamics_state_dict"])
            self.dynamics.eval()
            
            self.anchors = self.checkpoint["anchors"].to(self.device)
            self.action_candidates = self.checkpoint["action_candidates"].to(self.device)
            self.cached_value = self.checkpoint["cached_value"].to(self.device)
            
            self.planner = SoftPlanner(
                dynamics=self.dynamics,
                anchor_states=self.anchors,
                action_candidates=self.action_candidates,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gamma=0.99,
            )
            class MockComponent:
                def __init__(self, reward, cached_value):
                    self.reward = reward
                    self.cached_value = cached_value
            self.comp = MockComponent(self.reward_net, self.cached_value)
            
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            if self.mode == "bnirl_subgoal":
                action = self.policy_net(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
            else:
                cand = self.action_candidates.unsqueeze(0)
                q_vals = self.planner.q_values_for_actions(
                    self.comp, obs_tensor, cand, self.cached_value
                ).squeeze(0)
                best_idx = torch.argmax(q_vals).item()
                action = self.action_candidates[best_idx].cpu().numpy()
                
        return action, None


def load_learner(env_id: str, seed: int, stage: str, tr_name: str,
                 mode_idx: int, short: bool = True):
    """Load CoMI-IRL learner for a specific mode."""
    if tr_name in ["bnirl", "bnirl_subgoal", "bnirl_og", "choikim"]:
        algo_folder = {
            "bnirl": "learners_BNIRL",
            "bnirl_og": "learners_BNIRL_OG",
            "bnirl_subgoal": "learners_BNIRL_Subgoal",
            "choikim": "learners_ChoiKim"
        }[tr_name]
        search_pattern = os.path.join(f"./{algo_folder}", env_id, f"seed_{seed}", "K_*", "C*", f"*mode_{mode_idx}.pt")
        matching_files = glob.glob(search_pattern)
        if not matching_files:
            print(f"Warning: No BNIRL checkpoint found at {search_pattern}")
            return None
        checkpoint_path = matching_files[0]
        print(f"Loading BNIRL model from {checkpoint_path}...")
        return BNIRLPolicyWrapper(tr_name, checkpoint_path, env_id)
    base_dir = f'./learners/{env_id}/{seed}/{stage}' if short else \
               f'./learners_long/{env_id}/{seed}/{stage}'

    if not os.path.isdir(base_dir):
        if "finetuned" in stage:
            base_dir = f'./learners/{env_id}/{seed}/baseline' if short else \
                       f'./learners_long/{env_id}/{seed}/baseline'
        elif stage == "complete":
            for try_stage in ["finetuned_novel", "baseline"]:
                try_dir = f'./learners/{env_id}/{seed}/{try_stage}' if short else \
                          f'./learners_long/{env_id}/{seed}/{try_stage}'
                if os.path.isdir(try_dir):
                    base_dir = try_dir
                    break

    if not os.path.isdir(base_dir):
        return None

    ft_suffix = "_FT" if "finetuned" in stage else ""
    pattern = os.path.join(base_dir, f"{tr_name}_cluster_*_mode_{mode_idx}{ft_suffix}.zip")
    files = glob.glob(pattern)

    if not files and ft_suffix:
        pattern = os.path.join(base_dir, f"{tr_name}_cluster_*_mode_{mode_idx}.zip")
        files = glob.glob(pattern)

    if not files:
        return None

    path = files[0]
    if tr_name in ["gail", "airl"]:
        return PPO.load(path)
    elif tr_name == "sqil":
        return SAC.load(path)
    return None


def load_consensus_agent(env_id: str, seed: int, mode_idx: int,
                         save_dir: str = "./consensus_agents"):
    """Load consensus agent for a specific mode."""
    base_dir = os.path.join(save_dir, env_id)
    if not os.path.isdir(base_dir):
        # Try flat directory
        base_dir = save_dir
    
    pattern = os.path.join(base_dir, f"agent_intention_*_target_{mode_idx}_seed_{seed}.zip")
    print(pattern)
    files = glob.glob(pattern)

    if not files:
        # Broader pattern
        pattern = os.path.join(base_dir, f"*{env_id}*mode{mode_idx}*.zip")
        files = glob.glob(pattern)

    if not files:
        return None

    return PPO.load(files[0])


# ═══════════════════════════════════════════════════════════════
# Gait Metrics (BipedalWalker-specific)
# ═══════════════════════════════════════════════════════════════

def compute_gait_metrics(observations: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    """
    Compute gait-characterizing metrics from a BipedalWalker episode.

    Args:
        observations: (T+1, 24) array of observations.
        actions: (T, 4) array of actions.

    Returns:
        Dictionary with gait metrics.
    """
    if len(observations) < 2 or len(actions) < 1:
        return {}

    vel_x = observations[1:, 2]         # forward velocity
    vel_y = observations[1:, 3]         # vertical velocity
    hull_angle = observations[1:, 0]    # hull tilt
    hip1_angle = observations[1:, 4]    # hip1 angle
    hip2_angle = observations[1:, 9]    # hip2 angle
    contact1 = (observations[1:, 8] > 0.5).astype(float)
    contact2 = (observations[1:, 13] > 0.5).astype(float)

    # Hip spread
    hip_spread = np.abs(hip1_angle - hip2_angle)

    # Contact switches (step frequency proxy)
    c1_switches = np.sum(np.abs(np.diff(contact1)))
    c2_switches = np.sum(np.abs(np.diff(contact2)))
    total_switches = c1_switches + c2_switches
    T = len(actions)

    # Torque (action magnitude)
    torque = np.mean(np.abs(actions))

    return {
        "mean_vel_x": float(np.mean(vel_x)),
        "mean_vel_y_abs": float(np.mean(np.abs(vel_y))),
        "mean_hip_spread": float(np.mean(hip_spread)),
        "max_hip_spread": float(np.max(hip_spread)),
        "step_frequency": float(total_switches / max(T, 1)),
        "mean_torque": float(torque),
        "mean_hull_tilt": float(np.mean(np.abs(hull_angle))),
        "total_steps": int(total_switches),
        "episode_length": T,
    }


# ═══════════════════════════════════════════════════════════════
# Landing Metrics (LunarLanderContinuous-v2-specific)
# ═══════════════════════════════════════════════════════════════

def compute_landing_metrics(observations: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    """
    Compute approach-style metrics from a LunarLanderContinuous episode.

    Observation space (8-dim):
      obs[0] = x position
      obs[1] = y position
      obs[2] = x velocity
      obs[3] = y velocity
      obs[4] = angle
      obs[5] = angular velocity
      obs[6] = left leg contact (bool)
      obs[7] = right leg contact (bool)

    Action space (2-dim):
      act[0] = main engine thrust  [-1, 1]
      act[1] = lateral thrust      [-1, 1]

    Args:
        observations: (T+1, 8) array of observations.
        actions: (T, 2) array of actions.

    Returns:
        Dictionary with landing/approach metrics.
    """
    if len(observations) < 2 or len(actions) < 1:
        return {}

    x_pos = observations[1:, 0]
    y_pos = observations[1:, 1]
    x_vel = observations[1:, 2]
    y_vel = observations[1:, 3]
    angle = observations[1:, 4]
    ang_vel = observations[1:, 5]
    left_contact = observations[1:, 6]
    right_contact = observations[1:, 7]

    main_thrust = actions[:, 0]
    lateral_thrust = actions[:, 1]

    T = len(actions)

    # Approach direction: mean x position in the upper half of trajectory
    # (before descent phase)
    upper_mask = y_pos > 0.5
    if np.any(upper_mask):
        approach_x = float(np.mean(x_pos[upper_mask]))
        approach_x_std = float(np.std(x_pos[upper_mask]))
    else:
        approach_x = float(np.mean(x_pos[:T // 4])) if T > 4 else 0.0
        approach_x_std = float(np.std(x_pos[:T // 4])) if T > 4 else 0.0

    # Final landing position
    final_x = float(x_pos[-1])
    final_y = float(y_pos[-1])

    # Landing precision (distance from pad center at episode end)
    landing_dist = float(np.sqrt(final_x ** 2 + final_y ** 2))

    # Whether both legs made contact
    landed = bool(left_contact[-1] > 0.5 and right_contact[-1] > 0.5)

    # Fuel usage (total thrust magnitude)
    total_main_thrust = float(np.sum(np.clip(main_thrust, 0, 1)))
    total_lateral_thrust = float(np.sum(np.abs(lateral_thrust)))
    fuel_used = total_main_thrust + total_lateral_thrust

    # Mean lateral thrust direction (positive = right, negative = left)
    mean_lateral_thrust = float(np.mean(lateral_thrust))

    # Approach curvature: how much the x-position trajectory curves
    # High curvature = sweeping approach; low = straight descent
    if T > 2:
        dx = np.diff(x_pos)
        curvature = float(np.mean(np.abs(np.diff(dx))))
    else:
        curvature = 0.0

    # Maximum lateral deviation during episode
    max_lateral_dev = float(np.max(np.abs(x_pos)))

    # Mean descent speed
    mean_descent_speed = float(np.mean(-y_vel))

    # Angular stability
    mean_angle_abs = float(np.mean(np.abs(angle)))
    max_angle = float(np.max(np.abs(angle)))

    return {
        "approach_x": approach_x,
        "approach_x_std": approach_x_std,
        "final_x": final_x,
        "final_y": final_y,
        "landing_dist": landing_dist,
        "landed": float(landed),
        "fuel_used": fuel_used,
        "mean_lateral_thrust": mean_lateral_thrust,
        "curvature": curvature,
        "max_lateral_dev": max_lateral_dev,
        "mean_descent_speed": mean_descent_speed,
        "mean_angle_abs": mean_angle_abs,
        "max_angle": max_angle,
        "episode_length": T,
    }


# ═══════════════════════════════════════════════════════════════
# Cheetah Metrics (HalfCheetah-specific)
# ═══════════════════════════════════════════════════════════════

def compute_cheetah_metrics(observations: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    """
    Compute behavior-characterizing metrics from a HalfCheetah episode.

    Standard HalfCheetah-v4/v5 Observation space (17-dim):
      obs[0] = z coordinate (height)
      obs[1] = angle (pitch)
      ...
      obs[8] = x velocity (forward speed)
      obs[9] = z velocity (vertical speed)

    Action space (6-dim): Torques applied to 6 joints.

    Args:
        observations: (T+1, 17) array of observations.
        actions: (T, 6) array of actions.

    Returns:
        Dictionary with running metrics.
    """
    if len(observations) < 2 or len(actions) < 1:
        return {}

    # Standard Gym HalfCheetah-v4/v5 places x-velocity at index 8
    vel_x = observations[1:, 8]
    height = observations[1:, 0]
    pitch = observations[1:, 1]
    
    T = len(actions)

    # 1. Velocity Metrics
    mean_vel_x = float(np.mean(vel_x))
    max_vel_x = float(np.max(vel_x))
    min_vel_x = float(np.min(vel_x)) # Useful for checking backward running

    # 2. Energy / Action Metrics
    # Action cost in HalfCheetah is typically 0.1 * ||a||^2
    action_cost = float(np.sum(0.1 * np.square(actions)) / max(T, 1))
    mean_torque_abs = float(np.mean(np.abs(actions)))
    max_torque = float(np.max(np.abs(actions)))

    # 3. Posture Metrics
    mean_height = float(np.mean(height))
    mean_pitch_abs = float(np.mean(np.abs(pitch)))

    # Determine if it actually moved or just collapsed
    # If mean absolute velocity is tiny, it's standing still.
    is_standing_still = float(np.abs(mean_vel_x) < 0.5)

    return {
        "mean_vel_x": mean_vel_x,
        "max_vel_x": max_vel_x,
        "min_vel_x": min_vel_x,
        "action_cost_rate": action_cost,
        "mean_torque_abs": mean_torque_abs,
        "max_torque": max_torque,
        "mean_height": mean_height,
        "mean_pitch_abs": mean_pitch_abs,
        "is_standing_still": is_standing_still,
        "episode_length": T,
    }

# ═══════════════════════════════════════════════════════════════
# Rollout & Rendering
# ═══════════════════════════════════════════════════════════════

def rollout_episode(
    env,
    policy,
    mode_idx: int,
    is_legacy: bool,
    agent_type: str = "learner",
    max_steps: int = 100,
    render: bool = False,
    render_delay: float = 0.02,
    seed: int = None,
    deterministic: bool = True,
    env_id: str = None,
) -> Dict[str, Any]:
    """
    Roll out a single episode and collect metrics.

    Args:
        env: The environment.
        policy: The agent/policy.
        mode_idx: Mode to evaluate on.
        is_legacy: Whether this is a legacy environment.
        agent_type: "expert" (SAC .exploit), "learner"/"consensus" (SB3 .predict).
        max_steps: Maximum episode steps.
        render: Whether to render.
        render_delay: Delay between rendered frames (seconds).
        seed: Random seed for reset.
        deterministic: Whether to use deterministic actions.
        env_id: Environment ID (for computing environment-specific metrics).

    Returns:
        Dictionary with episode metrics.
    """
    obs = reset_env(env, mode_idx, is_legacy, seed=seed)

    total_reward = 0.0
    total_eval_reward = 0.0
    step_count = 0
    observations = [obs.copy()]
    actions_list = []
    rewards_list = []

    for step in range(max_steps):
        # Get action based on agent type
        if agent_type == "expert":
            action = policy.exploit(obs)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)

        # Step
        if is_legacy:
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate
        r_eval = info.get("reward_eval", reward) if isinstance(info, dict) else reward
        total_reward += reward
        total_eval_reward += r_eval
        step_count += 1

        observations.append(obs.copy())
        actions_list.append(action.copy() if hasattr(action, 'copy') else action)
        rewards_list.append(r_eval)

        # Render
        if render:
            if is_legacy:
                env.render()
            if render_delay > 0:
                time.sleep(render_delay)

        if terminated or truncated:
            break

    result_dict = {
        "total_reward": total_reward,
        "eval_reward": total_eval_reward,
        "steps": step_count,
        "observations": np.array(observations),
        "actions": np.array(actions_list) if actions_list else np.array([]),
        "rewards": np.array(rewards_list),
    }

    # Add environment-specific metrics
    if env_id == "BipedalWalker-v3" and len(actions_list) > 0:
        gait = compute_gait_metrics(
            np.array(observations), np.array(actions_list))
        result_dict["gait_metrics"] = gait
    if env_id == "LunarLanderContinuous-v3" and len(actions_list) > 0:
        landing = compute_landing_metrics(
            np.array(observations), np.array(actions_list))
        result_dict["landing_metrics"] = landing

    if env_id in ["HalfCheetah-v5", "mo-halfcheetah-v5"] and len(actions_list) > 0:
        cheetah = compute_cheetah_metrics(
            np.array(observations), np.array(actions_list))
        result_dict["cheetah_metrics"] = cheetah


    return result_dict


def deploy_agent(
    env_id: str,
    agent_name: str,
    policy,
    mode_idx: int,
    n_modes: int,
    num_episodes: int = 10,
    render: bool = False,
    render_delay: float = 0.02,
    seed: int = 42,
    config_id: int = None,
    short: bool = True,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Deploy an agent on a specific mode for multiple episodes.

    Returns a DataFrame with per-episode results.
    """
    # Determine agent type
    if agent_name == "expert":
        agent_type = "expert" if env_id in LEGACY_ENVS else "learner"
    elif agent_name.startswith("learner"):
        agent_type = "learner"
    elif agent_name == "consensus":
        agent_type = "learner"
    else:
        agent_type = "learner"

    # Determine environment mode index
    env_mode_idx = mode_idx
    if env_id == "Walker2d-v4" and mode_idx >= 3:
        env_mode_idx = mode_idx % 3
    if env_id in ["Reacher-v4", "Pusher-v4"] and mode_idx >= 6:
        env_mode_idx = mode_idx % 6

    # Create environment
    is_legacy = env_id in LEGACY_ENVS
    env, is_leg = create_env(
        env_id, n_modes,
        config_id=config_id if not is_legacy else None,
        mode_idx=env_mode_idx,
        render=render and not is_legacy,
    )

    if is_legacy:
        env.seed(seed)

    results = []
    for ep in range(num_episodes):
        ep_result = rollout_episode(
            env=env,
            policy=policy,
            mode_idx=env_mode_idx,
            is_legacy=is_leg,
            agent_type=agent_type,
            max_steps=DEFAULT_MAX_STEPS.get(env_id, 200),
            render=render,
            render_delay=render_delay,
            seed=seed + ep,
            deterministic=deterministic,
            env_id=env_id,
        )

        row = {
            "agent": agent_name,
            "mode_idx": mode_idx,
            "config_id": config_id if config_id is not None else mode_idx,
            "episode": ep,
            "eval_reward": ep_result["eval_reward"],
            "total_reward": ep_result["total_reward"],
            "steps": ep_result["steps"],
        }

        # Add gait metrics for BipedalWalker
        if "gait_metrics" in ep_result:
            for k, v in ep_result["gait_metrics"].items():
                row[f"gait_{k}"] = v
        # Add landing metrics for LunarLanderContinuous
        if "landing_metrics" in ep_result:
            for k, v in ep_result["landing_metrics"].items():
                row[f"landing_{k}"] = v
        # Add landing metrics for LunarLanderContinuous
        if "cheetah_metrics" in ep_result:
            for k, v in ep_result["cheetah_metrics"].items():
                row[f"cheetah_{k}"] = v
        results.append(row)

    env.close()
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# Mode / Config Mapping
# ═══════════════════════════════════════════════════════════════

def get_mode_config_mapping(env_id: str) -> List[Dict[str, Any]]:
    """
    Get the mapping from mode_idx to config_ids for an environment.

    For legacy envs: each mode has its own mode_idx.
    For gymnasium envs: modes map to one or more config_ids (mode×style).

    Returns a list of dicts with keys: mode_idx, config_ids, mode_name, style_names.
    """
    if env_id in GYMNASIUM_ENVS and HAS_GC_EXPERTS and env_id in ENV_SPECS:
        spec = ENV_SPECS[env_id]
        modes = {}
        for cfg in spec.configs:
            if cfg.mode_id not in modes:
                modes[cfg.mode_id] = {
                    "mode_idx": cfg.mode_id,
                    "mode_name": cfg.mode_name,
                    "config_ids": [],
                    "style_names": [],
                    "descs": [],
                }
            modes[cfg.mode_id]["config_ids"].append(cfg.config_id)
            modes[cfg.mode_id]["style_names"].append(cfg.style_name)
            modes[cfg.mode_id]["descs"].append(cfg.desc)
        return list(modes.values())

    else:
        n_modes = DEFAULT_N_MODES.get(env_id, 6)
        return [
            {"mode_idx": i, "config_ids": [i], "mode_name": f"mode_{i}",
             "style_names": ["default"], "descs": [f"mode_{i}"]}
            for i in range(n_modes)
        ]


# ═══════════════════════════════════════════════════════════════
# Batch Deployment
# ═══════════════════════════════════════════════════════════════

def deploy_all(
    env_id: str,
    agent_names: List[str],
    seeds: List[int],
    tr_names: List[str] = None,
    stage: str = "complete",
    num_episodes: int = 10,
    render: bool = False,
    render_delay: float = 0.02,
    only_modes: List[int] = None,
    only_configs: List[int] = None,
    short: bool = True,
    deterministic: bool = True,
    expert_policy_dir: str = "./expert_policies_new",
) -> pd.DataFrame:
    """
    Deploy and evaluate multiple agents across modes and seeds.
    """
    if tr_names is None:
        tr_names = ["airl"]

    device = torch.device("cpu")
    n_modes = DEFAULT_N_MODES.get(env_id, 6)
    is_legacy = env_id in LEGACY_ENVS
    structure = ENV_STRUCTURE.get(env_id, "unknown")
    mode_config_map = get_mode_config_mapping(env_id)

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} | {env_id} | Structure: {structure}")
        print(f"{'='*60}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        for mode_info in mode_config_map:
            mode_idx = mode_info["mode_idx"]
            config_ids = mode_info["config_ids"]
            mode_name = mode_info["mode_name"]
            style_names = mode_info["style_names"]

            if only_modes and mode_idx not in only_modes:
                continue

            # Header depends on structure
            if structure == "single_task_multi_style":
                print(f"\n--- Task: {mode_name} | "
                      f"Styles: {', '.join(style_names)} ---")
            else:
                print(f"\n--- Mode {mode_idx} ({mode_name}) | "
                      f"Configs: {config_ids} ---")

            for ci, config_id in enumerate(config_ids):
                if only_configs and config_id not in only_configs:
                    continue

                style_name = style_names[ci] if ci < len(style_names) else ""

                for agent_name in agent_names:
                    # ── Load agent ──
                    policy = None

                    if agent_name == "expert":
                        if is_legacy:
                            tmp_env, _ = create_env(env_id, n_modes)
                            adj_mode = mode_idx % 6 if env_id in ["Reacher-v4", "Pusher-v4"] else mode_idx
                            policy = load_expert_legacy(
                                tmp_env, env_id, n_modes, adj_mode, device)
                            tmp_env.close()
                        else:
                            policy = load_expert_gymnasium(
                                env_id, config_id, seed=seed,
                                policy_dir=expert_policy_dir)

                    elif agent_name == "learner" or agent_name.startswith("learner_"):
                        if agent_name.startswith("learner_"):
                            tr = agent_name.split("_", 1)[1]
                        else:
                            tr = tr_names[0] if tr_names else "airl"

                        adj_mode = mode_idx
                        if env_id == "Walker2d-v4" and mode_idx >= 3:
                            adj_mode = mode_idx % 3
                        if env_id in ["Reacher-v4", "Pusher-v4"] and mode_idx >= 6:
                            adj_mode = mode_idx % 6
                        if env_id == "HalfCheetah-v5" or env_id == "Hopper-v4" or env_id == "mo-halfcheetah-v5":
                            adj_mode = config_id
                        print(f"Learner {agent_name} has adj_mode {adj_mode} for env mode {mode_idx}")
                        policy = load_learner(
                            env_id, seed, stage, tr, adj_mode, short=short)

                        if policy is None:
                            print(f"  Warning: No {tr} learner for mode {adj_mode}")

                    elif agent_name == "consensus":
                        adj_mode = mode_idx
                        if env_id == "Walker2d-v4" and mode_idx >= 3:
                            adj_mode = mode_idx % 3
                        if env_id in ["Reacher-v4", "Pusher-v4"] and mode_idx >= 6:
                            adj_mode = mode_idx % 6
                        if env_id == "HalfCheetah-v5" or env_id == "Hopper-v4":
                            adj_mode = config_id

                        policy = load_consensus_agent(env_id, seed, adj_mode)

                        if policy is None:
                            print(f"  Info: No consensus agent for mode {adj_mode}")

                    if policy is None:
                        continue

                    # ── Deploy ──
                    if structure == "single_task_multi_style":
                        display_name = f"{agent_name} [{style_name}]"
                    elif style_name:
                        display_name = f"{agent_name} ({style_name})"
                    else:
                        display_name = agent_name

                    print(f"  Deploying {display_name} "
                          f"(config {config_id})...")

                    ep_results = deploy_agent(
                        env_id=env_id,
                        agent_name=agent_name,
                        policy=policy,
                        mode_idx=mode_idx,
                        n_modes=n_modes,
                        num_episodes=num_episodes,
                        render=render,
                        render_delay=render_delay,
                        seed=seed,
                        config_id=config_id,
                        short=short,
                        deterministic=deterministic,
                    )

                    # Add metadata
                    ep_results["seed"] = seed
                    ep_results["mode_name"] = mode_name
                    ep_results["style_name"] = style_name
                    ep_results["structure"] = structure

                    mean_r = ep_results["eval_reward"].mean()
                    std_r = ep_results["eval_reward"].std()
                    mean_steps = ep_results["steps"].mean()

                    summary_str = f"    → {mean_r:.2f} ± {std_r:.2f} ({mean_steps:.0f} steps)"

                    # Add gait summary for BipedalWalker
                    if env_id == "BipedalWalker-v3" and "gait_mean_torque" in ep_results.columns:
                        avg_torque = ep_results["gait_mean_torque"].mean()
                        avg_spread = ep_results["gait_mean_hip_spread"].mean()
                        avg_freq = ep_results["gait_step_frequency"].mean()
                        avg_vel = ep_results["gait_mean_vel_x"].mean()
                        summary_str += (f" | vel={avg_vel:.2f} "
                                        f"torque={avg_torque:.3f} "
                                        f"hip_spread={avg_spread:.2f} "
                                        f"step_freq={avg_freq:.3f}")

                    # Add landing summary for LunarLander
                    if env_id == "LunarLanderContinuous-v2" and "landing_approach_x" in ep_results.columns:
                        avg_approach = ep_results["landing_approach_x"].mean()
                        avg_fuel = ep_results["landing_fuel_used"].mean()
                        avg_dist = ep_results["landing_landing_dist"].mean()
                        land_rate = ep_results["landing_landed"].mean()
                        avg_curve = ep_results["landing_curvature"].mean()
                        summary_str += (f" | approach_x={avg_approach:.2f} "
                                        f"fuel={avg_fuel:.0f} "
                                        f"dist={avg_dist:.3f} "
                                        f"landed={land_rate:.0%} "
                                        f"curve={avg_curve:.4f}")


                    # Add landing summary for HalfCheetah
                    if env_id in ["HalfCheetah-v5", "mo-halfcheetah-v5"] and "mean_vel_x" in ep_results.columns:
                        avg_vel = ep_results["mean_vel_x"].mean()
                        summary_str += (f" | vel_x={avg_vel:.2f} ")


                    print(summary_str)

                    all_results.append(ep_results)

    if not all_results:
        print("\nNo results collected.")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# Summary & Reporting
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, env_id: str):
    """Print a formatted summary of deployment results."""
    if df.empty:
        print("No results to summarize.")
        return

    structure = ENV_STRUCTURE.get(env_id, "unknown")

    print(f"\n{'='*70}")
    print(f"  DEPLOYMENT SUMMARY: {env_id}")
    print(f"  Structure: {structure}")
    print(f"{'='*70}")

    if structure == "single_task_multi_style":
        _print_summary_single_task(df, env_id)
    else:
        _print_summary_multi_task(df, env_id)


def _print_summary_multi_task(df: pd.DataFrame, env_id: str):
    """Summary for multi-task environments (Reacher, Pusher, Hopper, Walker2d)."""
    agents = df["agent"].unique()
    modes = sorted(df["mode_idx"].unique())

    # Header
    header = f"{'Mode':>6} {'Name':>12}"
    for a in agents:
        header += f" | {a:>18}"
    print(header)
    print("-" * len(header))

    for mode_idx in modes:
        mode_df = df[df["mode_idx"] == mode_idx]
        mode_name = mode_df["mode_name"].iloc[0] if "mode_name" in mode_df else f"mode_{mode_idx}"

        row = f"{mode_idx:>6} {mode_name:>12}"
        for a in agents:
            agent_df = mode_df[mode_df["agent"] == a]
            if agent_df.empty:
                row += f" | {'N/A':>18}"
            else:
                mean = agent_df["eval_reward"].mean()
                std = agent_df["eval_reward"].std()
                row += f" | {mean:>8.2f} ± {std:>5.2f}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'':>6} {'OVERALL':>12}"
    for a in agents:
        agent_df = df[df["agent"] == a]
        if agent_df.empty:
            row += f" | {'N/A':>18}"
        else:
            mean = agent_df["eval_reward"].mean()
            std = agent_df["eval_reward"].std()
            row += f" | {mean:>8.2f} ± {std:>5.2f}"
    print(row)

    # Expert ratio
    if "expert" in agents:
        print(f"\n--- Performance Ratio vs Expert ---")
        for a in agents:
            if a == "expert":
                continue
            ratios = []
            for mode_idx in modes:
                expert_mean = df[(df["agent"] == "expert") &
                                (df["mode_idx"] == mode_idx)]["eval_reward"].mean()
                agent_mean = df[(df["agent"] == a) &
                               (df["mode_idx"] == mode_idx)]["eval_reward"].mean()

                if np.isnan(expert_mean) or np.isnan(agent_mean):
                    continue

                invert = env_id in ["Reacher-v4", "Pusher-v4"]
                if invert:
                    ratio = expert_mean / agent_mean if abs(agent_mean) > 1e-9 else 0.0
                else:
                    ratio = agent_mean / expert_mean if abs(expert_mean) > 1e-9 else 0.0
                ratios.append(ratio)

            if ratios:
                print(f"  {a}: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")


def _print_summary_single_task(df: pd.DataFrame, env_id: str):
    """
    Summary for single-task multi-style environments (BipedalWalker, LunarLander).
    Summary for single-task multi-style environments (BipedalWalker, LunarLander).

    All configs share the same task, so we compare styles, not modes.
    The key question: do different styles achieve similar task performance
    but with distinct behavioral signatures?
    """
    agents = df["agent"].unique()
    configs = sorted(df["config_id"].unique())

    # ── Reward Table (by style/config) ──
    print(f"\n  Task Performance by Style:")
    print(f"  {'Config':>6} {'Style':>16}", end="")
    print(f"  {'Config':>6} {'Style':>16}", end="")
    for a in agents:
        print(f" | {a:>18}", end="")
    print()
    print("  " + "-" * (24 + 21 * len(agents)))
    print("  " + "-" * (24 + 21 * len(agents)))

    for config_id in configs:
        cfg_df = df[df["config_id"] == config_id]
        style_name = cfg_df["style_name"].iloc[0] if "style_name" in cfg_df else f"config_{config_id}"

        row = f"  {config_id:>6} {style_name:>16}"
        row = f"  {config_id:>6} {style_name:>16}"
        for a in agents:
            agent_df = cfg_df[cfg_df["agent"] == a]
            if agent_df.empty:
                row += f" | {'N/A':>18}"
            else:
                mean = agent_df["eval_reward"].mean()
                std = agent_df["eval_reward"].std()
                row += f" | {mean:>8.2f} ± {std:>5.2f}"
        print(row)

    # Overall
    print("  " + "-" * (24 + 21 * len(agents)))
    row = f"  {'':>6} {'ALL STYLES':>16}"
    print("  " + "-" * (24 + 21 * len(agents)))
    row = f"  {'':>6} {'ALL STYLES':>16}"
    for a in agents:
        agent_df = df[df["agent"] == a]
        if agent_df.empty:
            row += f" | {'N/A':>18}"
        else:
            mean = agent_df["eval_reward"].mean()
            std = agent_df["eval_reward"].std()
            row += f" | {mean:>8.2f} ± {std:>5.2f}"
    print(row)

    # ── Cheetah Metrics Table (HalfCheetah only) ──
    cheetah_cols = [c for c in df.columns if c.startswith("cheetah_")]
    if cheetah_cols and env_id in ["HalfCheetah-v5", "mo-halfcheetah-v5"]:
        print(f"\n  Cheetah Metrics by Style:")
        key_metrics = [
            ("cheetah_mean_vel_x", "Mean Vel X"),
            ("cheetah_max_vel_x", "Max Vel X"),
            ("cheetah_min_vel_x", "Min Vel X"),
            ("cheetah_action_cost_rate", "Action Cost"),
            ("cheetah_mean_torque_abs", "|Torque|"),
            ("cheetah_mean_height", "Height"),
            ("cheetah_is_standing_still", "Standing%"),
        ]

    # ── Gait Metrics Table (BipedalWalker only) ──
    gait_cols = [c for c in df.columns if c.startswith("gait_")]
    if gait_cols and env_id == "BipedalWalker-v3":
        print(f"\n  Gait Metrics by Style (expert only shown if available):")

        key_metrics = [
            ("gait_mean_vel_x", "Fwd Velocity"),
            ("gait_mean_torque", "Mean Torque"),
            ("gait_mean_hip_spread", "Hip Spread"),
            ("gait_max_hip_spread", "Max Hip Spread"),
            ("gait_step_frequency", "Step Freq"),
            ("gait_mean_vel_y_abs", "|Vert Vel|"),
            ("gait_episode_length", "Ep Length"),
        ]

        available = [(col, name) for col, name in key_metrics if col in df.columns]

        if available:
            header = f"  {'Style':>16}"
            header = f"  {'Style':>16}"
            for _, name in available:
                header += f" | {name:>12}"
            print(header)
            print("  " + "-" * (18 + 15 * len(available)))
            print("  " + "-" * (18 + 15 * len(available)))

            for config_id in configs:
                cfg_df = df[df["config_id"] == config_id]
                style_name = cfg_df["style_name"].iloc[0] if "style_name" in cfg_df else f"cfg_{config_id}"

                if "expert" in cfg_df["agent"].values:
                    metric_df = cfg_df[cfg_df["agent"] == "expert"]
                else:
                    metric_df = cfg_df

                row = f"  {style_name:>16}"
                row = f"  {style_name:>16}"
                for col, _ in available:
                    if col in metric_df.columns and not metric_df[col].isna().all():
                        val = metric_df[col].mean()
                        row += f" | {val:>12.3f}"
                    else:
                        row += f" | {'N/A':>12}"
                print(row)

    # ── Landing Metrics Table (LunarLander only) ──
    landing_cols = [c for c in df.columns if c.startswith("landing_")]
    if landing_cols and env_id == "LunarLanderContinuous-v2":
        print(f"\n  Landing Metrics by Style:")

        key_metrics = [
            ("landing_approach_x", "Approach X"),
            ("landing_max_lateral_dev", "Max Lateral"),
            ("landing_curvature", "Curvature"),
            ("landing_mean_lateral_thrust", "Lat Thrust"),
            ("landing_fuel_used", "Fuel Used"),
            ("landing_landing_dist", "Land Dist"),
            ("landing_landed", "Land Rate"),
            ("landing_mean_descent_speed", "Desc Speed"),
            ("landing_mean_angle_abs", "|Angle|"),
        ]

        available = [(col, name) for col, name in key_metrics if col in df.columns]

        if available:
            header = f"  {'Style':>16}"
            for _, name in available:
                header += f" | {name:>12}"
            print(header)
            print("  " + "-" * (18 + 15 * len(available)))

            for config_id in configs:
                cfg_df = df[df["config_id"] == config_id]
                style_name = cfg_df["style_name"].iloc[0] if "style_name" in cfg_df else f"cfg_{config_id}"

                if "expert" in cfg_df["agent"].values:
                    metric_df = cfg_df[cfg_df["agent"] == "expert"]
                else:
                    metric_df = cfg_df

                row = f"  {style_name:>16}"
                for col, _ in available:
                    if col in metric_df.columns and not metric_df[col].isna().all():
                        val = metric_df[col].mean()
                        if col == "landing_landed":
                            row += f" | {val:>11.0%} "
                        else:
                            row += f" | {val:>12.3f}"
                    else:
                        row += f" | {'N/A':>12}"
                print(row)

    # ── Style Distinctness Score (works for both BW3 and LLC) ──
    all_metric_cols = gait_cols if env_id == "BipedalWalker-v3" else landing_cols
    if all_metric_cols:
        if env_id == "BipedalWalker-v3":
            distinctness_metrics = [
                ("gait_mean_vel_x", "Fwd Velocity"),
                ("gait_mean_torque", "Mean Torque"),
                ("gait_mean_hip_spread", "Hip Spread"),
                ("gait_step_frequency", "Step Freq"),
                ("gait_mean_vel_y_abs", "|Vert Vel|"),
            ]
        else:  # LunarLander
            distinctness_metrics = [
                ("landing_approach_x", "Approach X"),
                ("landing_max_lateral_dev", "Max Lateral"),
                ("landing_curvature", "Curvature"),
                ("landing_mean_lateral_thrust", "Lat Thrust"),
                ("landing_fuel_used", "Fuel Used"),
                ("landing_mean_descent_speed", "Desc Speed"),
            ]

        print(f"\n  Style Distinctness (coefficient of variation across styles):")
        for col, name in distinctness_metrics:
            if col not in df.columns:
                continue
            style_means = []
            for config_id in configs:
                cfg_df = df[df["config_id"] == config_id]
                if "expert" in cfg_df["agent"].values:
                    vals = cfg_df[cfg_df["agent"] == "expert"][col]
                else:
                    vals = cfg_df[col]
                if not vals.isna().all():
                    style_means.append(vals.mean())
            if len(style_means) >= 2:
                cv = np.std(style_means) / (abs(np.mean(style_means)) + 1e-9)
                marker = " ★" if cv > 0.3 else " ✓" if cv > 0.1 else ""
                print(f"    {name:>12}: CV = {cv:.3f}{marker}")
    # ── Landing Metrics Table (LunarLander only) ──
    landing_cols = [c for c in df.columns if c.startswith("landing_")]
    if landing_cols and env_id == "LunarLanderContinuous-v2":
        print(f"\n  Landing Metrics by Style:")

        key_metrics = [
            ("landing_approach_x", "Approach X"),
            ("landing_max_lateral_dev", "Max Lateral"),
            ("landing_curvature", "Curvature"),
            ("landing_mean_lateral_thrust", "Lat Thrust"),
            ("landing_fuel_used", "Fuel Used"),
            ("landing_landing_dist", "Land Dist"),
            ("landing_landed", "Land Rate"),
            ("landing_mean_descent_speed", "Desc Speed"),
            ("landing_mean_angle_abs", "|Angle|"),
        ]

        available = [(col, name) for col, name in key_metrics if col in df.columns]

        if available:
            header = f"  {'Style':>16}"
            for _, name in available:
                header += f" | {name:>12}"
            print(header)
            print("  " + "-" * (18 + 15 * len(available)))

            for config_id in configs:
                cfg_df = df[df["config_id"] == config_id]
                style_name = cfg_df["style_name"].iloc[0] if "style_name" in cfg_df else f"cfg_{config_id}"

                if "expert" in cfg_df["agent"].values:
                    metric_df = cfg_df[cfg_df["agent"] == "expert"]
                else:
                    metric_df = cfg_df

                row = f"  {style_name:>16}"
                for col, _ in available:
                    if col in metric_df.columns and not metric_df[col].isna().all():
                        val = metric_df[col].mean()
                        if col == "landing_landed":
                            row += f" | {val:>11.0%} "
                        else:
                            row += f" | {val:>12.3f}"
                    else:
                        row += f" | {'N/A':>12}"
                print(row)

    # ── Style Distinctness Score (works for both BW3 and LLC) ──
    all_metric_cols = gait_cols if env_id == "BipedalWalker-v3" else landing_cols
    if all_metric_cols:
        if env_id == "BipedalWalker-v3":
            distinctness_metrics = [
                ("gait_mean_vel_x", "Fwd Velocity"),
                ("gait_mean_torque", "Mean Torque"),
                ("gait_mean_hip_spread", "Hip Spread"),
                ("gait_step_frequency", "Step Freq"),
                ("gait_mean_vel_y_abs", "|Vert Vel|"),
            ]
        else:  # LunarLander
            distinctness_metrics = [
                ("landing_approach_x", "Approach X"),
                ("landing_max_lateral_dev", "Max Lateral"),
                ("landing_curvature", "Curvature"),
                ("landing_mean_lateral_thrust", "Lat Thrust"),
                ("landing_fuel_used", "Fuel Used"),
                ("landing_mean_descent_speed", "Desc Speed"),
            ]

        print(f"\n  Style Distinctness (coefficient of variation across styles):")
        for col, name in distinctness_metrics:
            if col not in df.columns:
                continue
            style_means = []
            for config_id in configs:
                cfg_df = df[df["config_id"] == config_id]
                if "expert" in cfg_df["agent"].values:
                    vals = cfg_df[cfg_df["agent"] == "expert"][col]
                else:
                    vals = cfg_df[col]
                if not vals.isna().all():
                    style_means.append(vals.mean())
            if len(style_means) >= 2:
                cv = np.std(style_means) / (abs(np.mean(style_means)) + 1e-9)
                marker = " ★" if cv > 0.3 else " ✓" if cv > 0.1 else ""
                print(f"    {name:>12}: CV = {cv:.3f}{marker}")

    # ── Consensus Test ──
    print(f"\n  Consensus Test:")
    print(f"  All {len(configs)} configs should map to 1 intention "
          f"(same task, different styles).")
    if "consensus" in agents:
        consensus_df = df[df["agent"] == "consensus"]
        if not consensus_df.empty:
            cons_mean = consensus_df["eval_reward"].mean()
            expert_means = []
            for config_id in configs:
                expert_df = df[(df["agent"] == "expert") &
                               (df["config_id"] == config_id)]
                if not expert_df.empty:
                    expert_means.append(expert_df["eval_reward"].mean())
            if expert_means:
                best_expert = max(expert_means)
                ratio = cons_mean / best_expert if abs(best_expert) > 1e-9 else 0.0
                print(f"    Consensus reward:     {cons_mean:.2f}")
                print(f"    Best expert reward:   {best_expert:.2f}")
                print(f"    Ratio:                {ratio:.3f}")
                if ratio > 0.8:
                    print(f"    ✓ Consensus successfully recovers the task")
                else:
                    print(f"    ✗ Consensus underperforms — may have "
                          f"failed to merge styles")
        else:
            print(f"    No consensus agent results available.")
    else:
        print(f"    Consensus agent not evaluated.")


def list_configs(env_id: str):
    """List all available configurations for an environment."""
    print(f"\nConfigurations for {env_id}:")
    print(f"{'='*60}")

    structure = ENV_STRUCTURE.get(env_id, "unknown")
    print(f"  Structure: {structure}")

    if env_id in GYMNASIUM_ENVS and HAS_GC_EXPERTS and env_id in ENV_SPECS:
        spec = ENV_SPECS[env_id]
        print(f"  Modes (tasks):  {spec.n_modes}")
        print(f"  Styles/mode:    {dict(spec.styles_per_mode)}")
        print(f"  Configurations: {spec.n_configs}")
        print()
        print(f"  {'Config':>6}  {'Mode':>4}  {'Style':>5}  {'Mode Name':>12}  "
              f"{'Style Name':>12}  Description")
        print(f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*20}")
        for c in spec.configs:
            print(f"  {c.config_id:>6}  {c.mode_id:>4}  {c.style_id:>5}  "
                  f"{c.mode_name:>12}  {c.style_name:>12}  {c.desc}")

    elif env_id in LEGACY_ENVS:
        n_modes = DEFAULT_N_MODES.get(env_id, 6)
        n_configs = DEFAULT_N_CONFIGS.get(env_id, n_modes)
        styles = 2 if env_id == "Reacher-v4" else 1

        print(f"  Modes (tasks):  {n_modes}")
        print(f"  Styles/mode:    {styles}")
        print(f"  Configurations: {n_configs}")
        print()

        if env_id == "Reacher-v4":
            for mode in range(n_modes):
                for style in range(styles):
                    config = mode * styles + style
                    style_name = "elbow-up" if style == 0 else "elbow-down"
                    print(f"  Config {config:>2}: Mode {mode} (target_{mode}) "
                          f"+ Style {style} ({style_name})")
        elif env_id == "Pusher-v4":
            for mode in range(n_modes):
                print(f"  Config {mode:>2}: Mode {mode} (target_{mode}) "
                      f"+ Style 0 (default)")
        elif env_id == "Walker2d-v4":
            mode_names = ["forward", "backward", "stand"]
            for mode in range(n_modes):
                print(f"  Config {mode:>2}: Mode {mode} "
                      f"({mode_names[mode] if mode < len(mode_names) else f'mode_{mode}'})")

    else:
        print(f"  Unknown environment: {env_id}")

    # Check what's available on disk
    print(f"\n--- Available on disk ---")

    expert_dir = "./expert_policies_new"
    if os.path.isdir(expert_dir):
        expert_files = glob.glob(os.path.join(expert_dir, f"{env_id}_*.zip"))
        if expert_files:
            print(f"  Expert policies: {len(expert_files)} found")
            for f in sorted(expert_files):
                print(f"    {os.path.basename(f)}")
        else:
            print(f"  Expert policies: none found in {expert_dir}")

    for n_modes in [3, 6]:
        weight_dir = f"weights/{env_id}_{n_modes}_modes"
        if os.path.isdir(weight_dir):
            weight_files = glob.glob(os.path.join(weight_dir, "*.pth"))
            print(f"  Legacy expert weights ({n_modes} modes): "
                  f"{len(weight_files)} found")

    consensus_dir = f"./consensus_agents/{env_id}"
    if os.path.isdir(consensus_dir):
        consensus_files = glob.glob(os.path.join(consensus_dir, "*.zip"))
        print(f"  Consensus agents: {len(consensus_files)} found")
        for f in sorted(consensus_files):
            print(f"    {os.path.basename(f)}")

    # Also check flat consensus_agents dir
    consensus_dir_flat = "./consensus_agents"
    if os.path.isdir(consensus_dir_flat):
        consensus_files = glob.glob(os.path.join(consensus_dir_flat, f"*{env_id}*.zip"))
        if consensus_files:
            print(f"  Consensus agents (flat): {len(consensus_files)} found")
            for f in sorted(consensus_files):
                print(f"    {os.path.basename(f)}")

    print()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Deploy and visualize expert, learner, and consensus agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render Reacher expert on all modes
  python gc_deploy.py --env_id Reacher-v4 --agents expert --seeds 42 --render

  # Compare all agents on BipedalWalker (single task, multi style)
  python gc_deploy.py --env_id BipedalWalker-v3 --agents expert consensus \\
    --seeds 42 --num_episodes 50

  # Deploy Hopper experts with rendering
  python gc_deploy.py --env_id Hopper-v4 --agents expert --seeds 42 --render

  # Only evaluate specific configs
  python gc_deploy.py --env_id BipedalWalker-v3 --agents expert \\
    --only_configs 0 1 2 --seeds 42

  # List available configurations
  python gc_deploy.py --env_id Hopper-v4 --list_configs
        """,
    )

    p.add_argument("--env_id", type=str, required=True,
                   choices=list(LEGACY_ENVS | GYMNASIUM_ENVS),
                   help="Environment ID.")
    p.add_argument("--agents", type=str, nargs="+",
                   default=["expert"],
                   help="Agent types to deploy: expert, learner, "
                        "learner_gail, learner_airl, consensus.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Random seeds.")
    p.add_argument("--tr_names", type=str, nargs="+", default=["airl"],
                   choices=["gail", "airl", "sqil", "bnirl", "bnirl_subgoal", "bnirl_og", "choikim"],
                   help="IRL algorithm names for learners.")
    p.add_argument("--stage", type=str, default="complete",
                   help="Stage for loading learners.")
    p.add_argument("--num_episodes", type=int, default=5,
                   help="Episodes per agent per mode.")
    p.add_argument("--render", action="store_true",
                   help="Enable rendering.")
    p.add_argument("--render_delay", type=float, default=0.02,
                   help="Delay between rendered frames (seconds).")
    p.add_argument("--only_modes", type=int, nargs="*",
                   help="Subset of mode indices to evaluate.")
    p.add_argument("--only_configs", type=int, nargs="*",
                   help="Subset of config IDs to evaluate.")
    p.add_argument("--short", action="store_true", default=True,
                   help="Use short log paths.")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions.")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic actions.")
    p.add_argument("--expert_policy_dir", type=str,
                   default="./expert_policies_new",
                   help="Directory with trained expert policies.")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Save results to CSV file.")
    p.add_argument("--list_configs", action="store_true",
                   help="List available configurations and exit.")

    args = p.parse_args()

    if args.list_configs:
        list_configs(args.env_id)
        return

    deterministic = not args.stochastic

    # Expand "learner" to include all tr_names
    agent_names = []
    for a in args.agents:
        if a == "learner":
            agent_names.extend([f"learner_{tr}" for tr in args.tr_names])
        else:
            agent_names.append(a)

    print(f"\n{'='*70}")
    print(f"  Agent Deployment & Visualization")
    print(f"  Environment:   {args.env_id}")
    print(f"  Structure:     {ENV_STRUCTURE.get(args.env_id, 'unknown')}")
    print(f"  Agents:        {agent_names}")
    print(f"  Seeds:         {args.seeds}")
    print(f"  Episodes:      {args.num_episodes}")
    print(f"  Render:        {args.render}")
    print(f"  Deterministic: {deterministic}")
    print(f"{'='*70}")

    results_df = deploy_all(
        env_id=args.env_id,
        agent_names=agent_names,
        seeds=args.seeds,
        tr_names=args.tr_names,
        stage=args.stage,
        num_episodes=args.num_episodes,
        render=args.render,
        render_delay=args.render_delay,
        only_modes=args.only_modes,
        only_configs=args.only_configs,
        short=args.short,
        deterministic=deterministic,
        expert_policy_dir=args.expert_policy_dir,
    )

    if not results_df.empty:
        print_summary(results_df, args.env_id)

        # Save
        if args.save_csv:
            results_df.to_csv(args.save_csv, index=False)
            print(f"Results saved to {args.save_csv}")
        else:
            results_dir = "gc_deployment_results"
            os.makedirs(results_dir, exist_ok=True)
            seeds_str = "_".join(map(str, args.seeds))
            agents_str = "_".join(agent_names)
            filename = (f"deploy_{args.env_id}_{agents_str}_"
                        f"seeds_{seeds_str}.csv")
            filepath = os.path.join(results_dir, filename)
            results_df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()