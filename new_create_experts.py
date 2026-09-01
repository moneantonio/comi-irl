#!/usr/bin/env python3
# filepath: create_experts.py
"""
Expert Trajectory Generator for Geometric Consensus Experiments.

To be used only for the Hopper and BipedalWalker trajectories, as Reacher and Pusher exist already

Generates multi-modal expert demonstrations for testing the geometric
consensus approach on new environments.

Terminology:
  - Mode (Task):   the goal/intention (e.g., reach target A, go forward)
  - Style:         the behavioral strategy (e.g., elbow-up, hopping)
  - Configuration: a specific (mode, style) pair → one expert policy

Supported environments:
  - Reacher-v4   (6 modes × 2 styles = 12 configurations)
  - Pusher-v4    (6 modes × 1 style  =  6 configurations)
  - Hopper-v4    (3 modes × {2,2,1} styles = 5 configurations)
  - BipedalWalker-v3 (1 modes × 3 styles = 3 configurations)

Usage:
  python create_experts.py --env Reacher-v4 --n_trajs 100
  python create_experts.py --env Hopper-v4 --n_trajs 100
  python create_experts.py --env BipedalWalker-v3 --n_trajs 100
  python create_experts.py --list_envs
"""

import os
import argparse
import pickle
import math
import numpy as np  # type: ignore[import]
import gymnasium as gym  # type: ignore[import]
import mo_gymnasium as mo_gym  # type: ignore[import]
from mo_gymnasium.wrappers import LinearReward # type: ignore[import]
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from stable_baselines3 import PPO  # type: ignore[import]
from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore[import]
from stable_baselines3.common.callbacks import EvalCallback  # type: ignore[import]

from imitation.data.types import Trajectory, TrajectoryWithRew  # type: ignore[import]


import warnings
warnings.filterwarnings("ignore")

DEFAULT_MAX_STEPS = {
    "Reacher-v4": 50,
    "Pusher-v4": 100,
    # "Walker2d-v4": 200,
    "Hopper-v4": 200, #500
    # "BipedalWalker-v3": 1600,
    # "LunarLanderContinuous-v3": 500,
    "HalfCheetah-v5": 100, #200
    # "Swimmer-v5": 500,
    # "Walker2d-v5": 500,
    # "mo-halfcheetah-v5": 500,
}

# ═══════════════════════════════════════════════════════════════
# Configuration Metadata
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfigInfo:
    """Metadata for a single (mode, style) configuration."""
    config_id: int        # unique index across all configs
    mode_id: int          # task/intention index
    style_id: int         # style index within mode
    mode_name: str        # e.g., "forward", "backward", "stand"
    style_name: str       # e.g., "hopping", "crawling"
    desc: str             # human-readable description


@dataclass
class EnvSpec:
    """Full specification of an environment's mode×style structure."""
    env_id: str
    n_modes: int                          # number of distinct tasks
    styles_per_mode: Dict[int, int]       # mode_id → number of styles
    n_configs: int                        # total (mode, style) pairs
    configs: List[ConfigInfo]             # all configurations
    training_timesteps: int = 500_000

    @property
    def mode_names(self) -> List[str]:
        seen = {}
        for c in self.configs:
            if c.mode_id not in seen:
                seen[c.mode_id] = c.mode_name
        return [seen[i] for i in sorted(seen)]


# ═══════════════════════════════════════════════════════════════
# Reward Shapers (Define Different Modes/Styles)
# ═══════════════════════════════════════════════════════════════

class RewardShaper:
    """Base class for reward shaping to create different expert configurations."""

    def __init__(self, config_id: int, mode_id: int, style_id: int = 0):
        self.config_id = config_id
        self.mode_id = mode_id
        self.style_id = style_id

    def shape_reward(self, obs, action, reward, info) -> float:
        raise NotImplementedError

    def get_description(self) -> str:
        raise NotImplementedError


# ─── Hopper-v4: 3 modes × {2,2,1} styles = 5 configurations ───

class HopperRewardShaper(RewardShaper):
    """
    Hopper with non-uniform task×style structure:

      Mode 0 (Forward):   2 styles — hopping, crawling
      Mode 1 (Backward):  2 styles — hopping, crawling
      Mode 2 (Stand):     1 style  — standing

    Configuration table:
      config 0: Forward  + Hopping   (mode=0, style=0) ─┐
      config 1: Forward  + Crawling  (mode=0, style=1) ─┘ same task
      config 2: Backward + Hopping   (mode=1, style=0) ─┐
      config 3: Backward + Crawling  (mode=1, style=1) ─┘ same task
      config 4: Stand    + Standing  (mode=2, style=0)     singleton orbit

    Style separation in the behavioral encoder:
      - Hopping:   high z-variance, periodic vertical oscillation
      - Crawling:  low z-variance, quasi-static forward lean
      - Standing:  minimal state change, high stability
      → Encoder clusters by style (hopping vs crawling vs standing)
      → Gradient consensus recovers task (forward vs backward vs stand)
    """

    CONFIGS = {
        # ── Mode 0: FORWARD ───────────────────────────────────
        0: { #perfect
            "mode": 0, "style": 0,
            "mode_name": "forward", "style_name": "hopping",
            "desc": "Forward+Hopping",
            "vel_target": 2.0,
            "vel_weight": 3.0,
            "height_bonus": 5.0,
            "height_baseline": 1.3,
            "action_penalty": 0.01,
            "alive_bonus": 1.0,
            "crouch_penalty": 0.0,
        },
        # 1: { #perfect
        #     "mode": 0, "style": 1,
        #     "mode_name": "forward", "style_name": "crawling",
        #     "desc": "Forward+Crawling",
        #     "vel_target": 1.5,
        #     "vel_weight": 3.0,
        #     "height_bonus": 0.0,
        #     "height_baseline": 1.3,
        #     "action_penalty": 0.5,
        #     "alive_bonus": 1.0,
        #     "crouch_penalty": 3.0,
        # },
        # ── Mode 1: BACKWARD ──────────────────────────────────
        1: {
            "mode": 1, "style": 0,
            "mode_name": "backward", "style_name": "hopping",
            "desc": "Backward+Hopping",
            "vel_target":       -2.5,
            "vel_weight":        3.0,
            "height_bonus":     15.0,
            "height_baseline":   1.3,
            "action_penalty":    0.001,  # almost zero — don't punish trying
            "alive_bonus":      10.0,    # large — staying upright must always pay
            "crouch_penalty":    0.0,
            "upright_penalty":   5.0,    # reduced from 10 — less punishing when tilting
            "z_vel_penalty":     0.0,
        },
        # 3: {
        #     "mode": 1, "style": 2,
        #     "mode_name": "backward", "style_name": "hooking",
        #     "desc": "Backward+Hooking",
        #     "vel_target":       -1.5,
        #     "vel_weight":        2.0,
        #     "height_bonus":      0.0,
        #     "height_baseline":   1.3,
        #     "action_penalty":    0.01,
        #     "alive_bonus":       3.0,   # increased — reward staying alive
        #     "crouch_penalty":    3.0,   
        #     "upright_penalty":   0.0,
        #     "z_vel_penalty":     2.0,   # keep — suppress hopping
        #     "lean_bonus":        1.0,   # NEW — reward backward lean (negative angle)
        # },
        # ── Mode 2: STAND STILL ───────────────────────────────
        2: { 
            "mode": 2, "style": 1,
            "mode_name": "stand", "style_name": "standing",
            "desc": "StandStill",
            "vel_target": 0.0,
            "vel_weight": 0.0,
            "height_bonus": 2.0,
            "height_baseline": 1.3,
            "action_penalty": 2.0,
            "alive_bonus": 5.0,
            "crouch_penalty": 0.0,
            "upright_penalty":   3.0,
            "z_vel_penalty":     1.0,   
        },
    }

    def shape_reward(self, obs, action, reward, info):
        cfg = self.CONFIGS[self.config_id]

        # Hopper obs: [z_pos, angle, thigh_joint, leg_joint, foot_joint,
        #              x_vel, z_vel, thigh_vel, leg_vel, foot_vel, ...]
        height = obs[0] if len(obs) > 0 else 1.3
        angle  = obs[1]   # torso angle: 0=upright, positive=leaning forward
        thigh = obs[2]   # thigh joint angle
        z_vel  = obs[6]   # vertical velocity
        x_vel = info.get('x_velocity', obs[5] if len(obs) > 5 else 0)

        # Velocity component: track target velocity
        vel_error = abs(x_vel - cfg["vel_target"])
        vel_reward = -cfg["vel_weight"] * vel_error
        # print("VEL_REWARD", vel_reward, "X_VEL", x_vel, "TARGET", cfg["vel_target"])

        # For standing: penalize ANY velocity
        if cfg["mode_name"] == "stand":
            vel_reward = -5.0 * abs(x_vel)

        # Height component
        height_delta = height - cfg["height_baseline"]
        height_reward = cfg["height_bonus"] * max(0, height_delta)

        # Crouch penalty (for crawling style: penalize being above baseline)
        crouch_cost = cfg["crouch_penalty"] * max(0, height_delta)

        # Action penalty
        action_cost = cfg["action_penalty"] * np.sum(action ** 2)

        # Alive bonus
        alive = cfg["alive_bonus"]

        # Upright penalty: penalise torso angle deviation from vertical.
        # Prevents "fall-forward-and-bounce" hack in backward hopping.
        # Zero for configs that don't define it.
        upright_cost = cfg.get("upright_penalty", 0.0) * (angle ** 2)

        # Vertical velocity penalty: discourages periodic bouncing.
        # Used in backward crawling to suppress hopping motion.
        # Zero for configs that don't define it.
        z_vel_cost = cfg.get("z_vel_penalty", 0.0) * (z_vel ** 2)

        shaped = (alive + vel_reward + height_reward
                - crouch_cost - action_cost - upright_cost - z_vel_cost)
        supine_cost = cfg.get("supine_penalty", 0.0) * ((angle - (-1.57)) ** 2)
        shaped -= supine_cost
        lean_bonus = cfg.get("lean_bonus", 0.0) * max(0.0, -angle)
        shaped += lean_bonus
        thigh_bonus = cfg.get("thigh_bonus", 0.0) * max(0.0, thigh)
        shaped += thigh_bonus
        return shaped

    def get_description(self):
        return self.CONFIGS[self.config_id]["desc"]


# ─── HalfCheetah-v5: 1 modes × 3 styles = 3 configurations ───

class HalfCheetahRewardShaper(RewardShaper):
    """
    HalfCheetah with 1 task × 3 styles.

    Task: go forward (shared across all configs).
    Styles differ in HOW the cheetah moves forward.

    HalfCheetah-v4 obs (17-dim):
      [0]  rootz      (torso height, relative)
      [1]  rooty      (torso pitch angle)
      [2]  bthigh     (back thigh joint)
      [3]  bshin      (back shin joint)
      [4]  bfoot      (back foot joint)
      [5]  fthigh     (front thigh joint)
      [6]  fshin      (front shin joint)
      [7]  ffoot      (front foot joint)
      [8]  rootx vel  (forward velocity)  ← main task signal
      [9]  rootz vel  (vertical velocity)
      [10] rooty vel  (pitch angular vel)
      [11..16] joint angular velocities (back: 11-13, front: 14-16)

    HalfCheetah never terminates early (no unhealthy condition),
    so NoEarlyTerm wrapper is not needed. We still pass
    terminate_when_unhealthy=False explicitly for safety.

    Configuration table:
      config 0: Forward + Bouncing   (mode=0, style=0)
        Large back-leg drive, unrestricted torque → long leaping strides
      config 1: Forward + Prowling   (mode=0, style=1)
        Heavy action penalty → small, energy-efficient mincing steps
      config 2: Forward + Crouching  (mode=0, style=2)
        Penalise high torso → agent stays low, producing a crawling gait
    """

    CONFIGS = {
        0: { #good, behaves normally
            "mode": 0, "style": 0,
            "mode_name": "forward", "style_name": "prowling",
            "desc": "Forward+Prowling",
            # Penalise all large torques → tiny careful steps
            "back_joint_bonus":   0.0,
            "action_penalty":     2.0,
            "crouch_bonus":       0.0,
            "forward_vel_bonus":  0.0,
        },
        1: { # backwards
            "mode": 1, "style": 0,
            "mode_name": "backward", "style_name": "prowling",
            "desc": "Backward+Prowling",
            # Penalise all large torques → tiny careful steps
            "back_joint_bonus":   0.0,
            "action_penalty":     2.0,
            "crouch_bonus":       0.0,
            "forward_vel_bonus":  -1.0,
        },
        2: { #good, stays in place
            "mode": 2, "style": 1,
            "mode_name": "standing", "style_name": "still",
            "desc": "Standing+Still",
            "back_joint_bonus":   0.0,
            "action_penalty":     10.0,
            "crouch_bonus":       0.0,
            "forward_vel_bonus":  0.0,
        },
    }

    # HalfCheetah's natural torso height at rest is around 0.0
    # (rootz is already relative to the ground in v4)
    _HEIGHT_BASELINE = 0.0

    def shape_reward(self, obs, action, reward, info):
        cfg = self.CONFIGS[self.config_id]

        rootz   = obs[0]   # torso height (relative)
        rooty   = obs[1]   # torso pitch angle
        x_vel   = obs[8]   # forward velocity
        joint_vels   = obs[11:] # 6 joint angular velocities

        shaped = reward  # keep env's own ctrl_cost and forward_reward

        # Extra forward velocity bonus (shared task signal, same for all styles)
        shaped += cfg["forward_vel_bonus"] * x_vel #max(x_vel, 0.0)

        # Style 0 — Bouncing: reward large back-leg actuations
        # actions[3:6] correspond to back thigh, shin, foot
        if cfg["back_joint_bonus"] > 0:
            shaped += cfg["back_joint_bonus"] * np.sum(action[3:6] ** 2)

        # Style 1 — Prowling: penalise all joint torques
        shaped -= cfg["action_penalty"] * np.sum(action ** 2)

        # Style 2 — Walking: track a target velocity (replaces generic forward bonus)
        vel_target = cfg.get("vel_target", None)
        if vel_target is not None:
            shaped -= cfg.get("vel_weight", 0.0) * (x_vel - vel_target) ** 2
        else:
            shaped += cfg["forward_vel_bonus"] * x_vel#max(x_vel, 0.0)

        return shaped

    def get_description(self):
        return self.CONFIGS[self.config_id]["desc"]

# ═══════════════════════════════════════════════════════════════
# Environment Specification Registry
# ═══════════════════════════════════════════════════════════════

def _build_env_spec(env_id: str, shaper_class, training_timesteps: int) -> EnvSpec:
    """Build an EnvSpec from a shaper class's CONFIGS dict."""
    configs_raw = shaper_class.CONFIGS
    configs = []
    mode_styles: Dict[int, int] = {}

    for cid in sorted(configs_raw.keys()):
        c = configs_raw[cid]
        mid = c["mode"]
        sid = c["style"]
        configs.append(ConfigInfo(
            config_id=cid,
            mode_id=mid,
            style_id=sid,
            mode_name=c["mode_name"],
            style_name=c["style_name"],
            desc=c["desc"],
        ))
        mode_styles[mid] = max(mode_styles.get(mid, 0), sid + 1)

    n_modes = len(mode_styles)
    n_configs = len(configs)

    return EnvSpec(
        env_id=env_id,
        n_modes=n_modes,
        styles_per_mode=mode_styles,
        n_configs=n_configs,
        configs=configs,
        training_timesteps=training_timesteps,
    )


# Shaper classes for environments that need reward shaping
SHAPER_REGISTRY = {
    "Hopper-v4": HopperRewardShaper,
    # "BipedalWalker-v3": BipedalWalkerRewardShaper,
    # "LunarLanderContinuous-v3": LunarLanderRewardShaper,
    "HalfCheetah-v5": HalfCheetahRewardShaper,
    # "Swimmer-v5": SwimmerRewardShaper,
    # "Walker2d-v5": Walker2dRewardShaper,
    # "mo-halfcheetah-v5": MOHalfCheetahRewardShaper,
}

# Full environment specifications
ENV_SPECS: Dict[str, EnvSpec] = {}

# Auto-build specs from shaper registry
_TRAINING_TIMESTEPS = {
    "Hopper-v4": 1_000_000,
    "BipedalWalker-v3": 500_000,
    "LunarLanderContinuous-v3": 500_000,
    "HalfCheetah-v5": 1_000_000,
    "Swimmer-v5": 500_000,
    "Walker2d-v5": 1_000_000,
    "mo-halfcheetah-v5": 1_000_000,
}

for _env_id, _shaper_cls in SHAPER_REGISTRY.items():
    ENV_SPECS[_env_id] = _build_env_spec(
        _env_id, _shaper_cls, _TRAINING_TIMESTEPS[_env_id])

# Built-in multimodal environments (no reward shaping needed)
ENV_SPECS["Reacher-v4"] = EnvSpec(
    env_id="Reacher-v4",
    n_modes=6,
    styles_per_mode={i: 2 for i in range(6)},
    n_configs=12,
    configs=[
        ConfigInfo(i, mode_id=i // 2, style_id=i % 2,
                   mode_name=f"target_{i // 2}", style_name=f"arm_config_{i % 2}",
                   desc=f"Target{i // 2}+Config{i % 2}")
        for i in range(12)
    ],
    training_timesteps=250_000,
)

ENV_SPECS["Pusher-v4"] = EnvSpec(
    env_id="Pusher-v4",
    n_modes=6,
    styles_per_mode={i: 1 for i in range(6)},
    n_configs=6,
    configs=[
        ConfigInfo(i, mode_id=i, style_id=0,
                   mode_name=f"target_{i}", style_name="default",
                   desc=f"Target{i}+Default")
        for i in range(6)
    ],
    training_timesteps=250_000,
)




# ═══════════════════════════════════════════════════════════════
# Shaped Environment Wrapper
# ═══════════════════════════════════════════════════════════════

class ShapedRewardWrapper(gym.Wrapper):
    """Wrapper that replaces the environment reward with a shaped reward."""

    def __init__(self, env, shaper: RewardShaper):
        super().__init__(env)
        self.shaper = shaper

    def reset(self, *, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        if hasattr(self.shaper, 'reset_state'):
            self.shaper.reset_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.shaper.shape_reward(obs, action, reward, info)
        return obs, shaped_reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════
# Expert Training
# ═══════════════════════════════════════════════════════════════

def train_expert(
    env_id: str,
    config_id: int,
    seed: int = 42,
    timesteps: Optional[int] = None,
    save_dir: str = "./expert_policies_new",
    verbose: bool = True,
) -> str:
    """
    Train an expert policy for a specific configuration (mode×style pair).

    Returns the path to the saved policy.
    """
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(save_dir, f"{env_id}_config_{config_id}_seed_{seed}")

    if os.path.exists(policy_path + ".zip"):
        if verbose:
            print(f"  Expert for config {config_id} already exists. Skipping.")
        return policy_path

    # Build environment
    if env_id in SHAPER_REGISTRY:
        shaper_class = SHAPER_REGISTRY[env_id]
        cfg_info = ENV_SPECS[env_id].configs[config_id]
        shaper = shaper_class(
            config_id=config_id,
            mode_id=cfg_info.mode_id,
            style_id=cfg_info.style_id,
        )

        def make_env():
            if env_id == "Hopper-v4" or env_id == "Walker2d-v5":
                # HalfCheetah-v5 has no early termination, but we still set max_episode_steps
                env = gym.make(env_id,max_episode_steps=DEFAULT_MAX_STEPS[env_id],terminate_when_unhealthy=False)
            elif env_id == "mo-halfcheetah-v5":
                # Special handling for mo-halfcheetah-v5
                env = mo_gym.make(env_id,max_episode_steps=DEFAULT_MAX_STEPS[env_id])
                env = LinearReward(env, weight=np.array([shaper.CONFIGS[config_id].get("forward_weight", 1.0), shaper.CONFIGS[config_id].get("cost_weight", 1.0)], dtype=np.float32))
            else:
                env = gym.make(env_id,max_episode_steps=DEFAULT_MAX_STEPS[env_id])
            return ShapedRewardWrapper(env, shaper)

        env = DummyVecEnv([make_env])
        if verbose:
            print(f"  Training expert for {env_id} config {config_id} "
                  f"({shaper.get_description()})...")
    else:
        # For environments with built-in multimodal support (Reacher, Pusher)
        if verbose:
            cfg_info = ENV_SPECS[env_id].configs[config_id]
            print(f"  Training expert for {env_id} config {config_id} "
                  f"({cfg_info.desc})...")
        env = DummyVecEnv([lambda: gym.make(env_id, max_episode_steps=DEFAULT_MAX_STEPS[env_id])])

    spec = ENV_SPECS.get(env_id)
    if timesteps is None:
        timesteps = spec.training_timesteps if spec else 500_000

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=timesteps, progress_bar=verbose)
    model.save(policy_path)

    if verbose:
        print(f"  Saved to {policy_path}.zip")

    return policy_path


def collect_trajectories(
    env_id: str,
    config_id: int,
    policy_path: str,
    n_trajs: int = 200,
    seed: int = 42,
    verbose: bool = True,
    save_dir: str = "./expert_imitation_trajectories",
) -> Tuple[List[Trajectory], List[TrajectoryWithRew]]:
    """
    Collect demonstration trajectories from a trained expert.

    Returns:
        trajs: List[Trajectory]
        trajs_with_rew: List[TrajectoryWithRew]
    """
    # Build environment
    if env_id in SHAPER_REGISTRY:
        cfg_info = ENV_SPECS[env_id].configs[config_id]
        shaper = SHAPER_REGISTRY[env_id](
            config_id=config_id,
            mode_id=cfg_info.mode_id,
            style_id=cfg_info.style_id,
        )
        if env_id == "Hopper-v4" or env_id == "Walker2d-v5":
            base_env = gym.make(env_id, max_episode_steps=DEFAULT_MAX_STEPS[env_id], terminate_when_unhealthy=False)
        elif env_id == "mo-halfcheetah-v5":
            base_env = mo_gym.make(env_id, max_episode_steps=DEFAULT_MAX_STEPS[env_id])
            base_env = LinearReward(base_env, weight=np.array([shaper.CONFIGS[config_id].get("forward_weight", 1.0), shaper.CONFIGS[config_id].get("cost_weight", 1.0)], dtype=np.float32))
        else:
            base_env = gym.make(env_id, max_episode_steps=DEFAULT_MAX_STEPS[env_id])
        env = ShapedRewardWrapper(base_env, shaper)
    else:
        env = gym.make(env_id)

    model = PPO.load(policy_path)

    trajs = []
    trajs_rew = []

    for ep in range(n_trajs):
        obs_list = []
        act_list = []
        rew_list = []

        obs, info = env.reset(seed=seed + ep)
        obs_list.append(obs.copy())

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            obs_list.append(obs.copy())
            act_list.append(action.copy())
            rew_list.append(reward)

            done = terminated or truncated

        obs_arr = np.array(obs_list)
        act_arr = np.array(act_list)
        rew_arr = np.array(rew_list, dtype=np.float64)

        traj = Trajectory(obs=obs_arr, acts=act_arr, infos=None, terminal=True)
        traj_rew = TrajectoryWithRew(
            obs=obs_arr, acts=act_arr, rews=rew_arr, infos=None, terminal=True,
        )

        trajs.append(traj)
        trajs_rew.append(traj_rew)

    if verbose:
        cfg_info = ENV_SPECS[env_id].configs[config_id]
        mean_rew = np.mean([np.sum(t.rews) for t in trajs_rew])
        mean_len = np.mean([len(t.acts) for t in trajs])
        print(f"  Config {config_id} (task={cfg_info.mode_name}, "
              f"style={cfg_info.style_name}): {n_trajs} trajs, "
              f"mean_reward={mean_rew:.2f}, mean_len={mean_len:.1f}")

    return trajs, trajs_rew


def generate_all_experts(
    env_id: str,
    n_trajs: int = 200,
    seed: int = 42,
    timesteps: Optional[int] = None,
    save_dir: str = "./expert_trajectories_new",
    policy_dir: str = "./expert_policies_new",
    verbose: bool = True,
):
    """
    Generate expert demonstrations for all configurations of an environment.

    Saves trajectories in the unified format:
      {save_dir}/{env_id}_task_{mode_id}_style_{style_id}.pkl
      {save_dir}/{env_id}_task_{mode_id}_style_{style_id}_withrew.pkl

    Also saves the EnvSpec metadata for downstream use.
    """
    os.makedirs(save_dir, exist_ok=True)

    spec = ENV_SPECS[env_id]

    print(f"\n{'='*60}")
    print(f"Generating Expert Trajectories")
    print(f"  Environment:    {env_id}")
    print(f"  Modes (tasks):  {spec.n_modes}")
    print(f"  Styles/mode:    {dict(spec.styles_per_mode)}")
    print(f"  Configurations: {spec.n_configs}")
    print(f"  Trajs/Config:   {n_trajs}")
    print(f"  Seed:           {seed}")
    print(f"{'='*60}")

    # Print config table
    print(f"\n  {'Config':>6}  {'Task':>4}  {'Style':>5}  Description")
    print(f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*30}")
    for c in spec.configs:
        print(f"  {c.config_id:>6}  {c.mode_id:>4}  {c.style_id:>5}  "
              f"{c.mode_name}+{c.style_name}")
    print()

    for cfg in spec.configs:
        cid = cfg.config_id
        print(f"\n--- Config {cid}/{spec.n_configs - 1}: "
              f"{cfg.mode_name}+{cfg.style_name} ---")

        # New unified naming
        traj_path = os.path.join(
            save_dir,
            f"{env_id}_task_{cfg.mode_id}.pkl")
        traj_rew_path = traj_path.replace(".pkl", "_withrew.pkl")

        if os.path.exists(traj_path) and os.path.exists(traj_rew_path):
            print(f"  Trajectories already exist. Skipping.")
            continue

        # Train expert
        policy_path = train_expert(
            env_id=env_id,
            config_id=cid,
            seed=seed,
            timesteps=timesteps,
            save_dir=policy_dir,
            verbose=verbose,
        )

        # Collect trajectories
        trajs, trajs_rew = collect_trajectories(
            env_id=env_id,
            config_id=cid,
            policy_path=policy_path,
            n_trajs=n_trajs,
            seed=seed,
            verbose=verbose,
            save_dir=save_dir,
        )

        # Save
        with open(traj_path, "wb") as f:
            pickle.dump(trajs, f)
        with open(traj_rew_path, "wb") as f:
            pickle.dump(trajs_rew, f)

        print(f"  Saved to {traj_path}")

    # Save the environment spec for downstream use
    spec_path = os.path.join(save_dir, f"env_spec_{env_id}.pkl")
    with open(spec_path, "wb") as f:
        pickle.dump(spec, f)
    print(f"\nSaved EnvSpec to {spec_path}")

    print(f"\n{'='*60}")
    print(f"Expert generation complete!")
    print(f"  {spec.n_configs} configurations × {n_trajs} trajectories "
          f"= {spec.n_configs * n_trajs} total demonstrations")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate expert trajectories for GC experiments')

    parser.add_argument('--env', type=str, default=None,
                        help='Environment ID (e.g., Hopper-v4)')
    parser.add_argument('--n_trajs', type=int, default=101,
                        help='Trajectories per configuration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps per expert')
    parser.add_argument('--save_dir', type=str,
                        default='./expert_trajectories_new')
    parser.add_argument('--policy_dir', type=str,
                        default='./expert_policies_new')
    parser.add_argument('--list_envs', action='store_true',
                        help='List supported environments and exit')

    args = parser.parse_args()

    if args.list_envs:
        print("\nSupported environments for Geometric Consensus:")
        print("=" * 60)
        for env_id, spec in sorted(ENV_SPECS.items()):
            print(f"\n{env_id}")
            print(f"  Modes (tasks):  {spec.n_modes}")
            print(f"  Styles/mode:    {dict(spec.styles_per_mode)}")
            print(f"  Configurations: {spec.n_configs}")
            print(f"  Timesteps:      {spec.training_timesteps:,}")
            print(f"  {'Config':>8}  {'Mode':>12}  {'Style':>12}  Description")
            print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*20}")
            for c in spec.configs:
                print(f"  {c.config_id:>8}  {c.mode_name:>12}  "
                      f"{c.style_name:>12}  {c.desc}")
        return

    if args.env is None:
        parser.error("--env is required (or use --list_envs)")

    if args.env not in ENV_SPECS:
        print(f"Error: '{args.env}' not in registry. "
              f"Available: {list(ENV_SPECS.keys())}")
        print("Use --list_envs to see details.")
        return

    generate_all_experts(
        env_id=args.env,
        n_trajs=args.n_trajs,
        seed=args.seed,
        timesteps=args.timesteps,
        save_dir=args.save_dir,
        policy_dir=args.policy_dir,
    )


if __name__ == "__main__":
    main()