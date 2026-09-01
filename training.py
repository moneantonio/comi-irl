import torch as th  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from torch.utils import data # type: ignore[import]
from tqdm import tqdm # type: ignore[import]
import copy
import random
from stable_baselines3.ppo import MlpPolicy #type: ignore[import]
from stable_baselines3.sac import SAC #type: ignore[import]
from stable_baselines3.common.env_util import make_vec_env #type: ignore[import]
from stable_baselines3 import PPO #type: ignore[import]
from imitation.algorithms.adversarial.airl import AIRL #type: ignore[import]
from imitation.algorithms.adversarial.gail import GAIL #type: ignore[import]
from imitation.algorithms import sqil #type: ignore[import]
from imitation.rewards.reward_nets import BasicShapedRewardNet #type: ignore[import]
from imitation.util.networks import RunningNorm #type: ignore[import]

from loss import *
from utils import *
from essinfogail.envs.wrappers import *
import essinfogail.envs.pusher as pusher_mod
import essinfogail.envs.walker as walker2d_mod
import essinfogail.envs.reacher as reacher_mod
import essinfogail.envs.hopper_v4 as hopper_mod
import essinfogail.envs.halfcheetah_v5 as halfcheetah_mod

def make_env_by_name(env_name: str, num_modes: int, render: bool = False):
    render_kwargs = {"render_mode": "human"} if render else {}
    if env_name == "Reacher-v4":
        try:
            return reacher_mod.MultimodalReacher(num_modes, **render_kwargs)
        except TypeError:
            return reacher_mod.MultimodalReacher(num_modes=num_modes)
    if env_name == "Pusher-v4":
        try:
            return pusher_mod.MultimodalPusher(num_modes, **render_kwargs)
        except TypeError:
            return pusher_mod.MultimodalPusher(num_modes=num_modes)
    if env_name == "Walker2d-v4":
        try:
            env = walker2d_mod.MultimodalWalker(num_modes, **render_kwargs)
        except TypeError:
            env = walker2d_mod.MultimodalWalker(num_modes=num_modes)
        return FixedLengthEnvWrapper(env)
    if env_name == "Hopper-v4":
        try:
            return hopper_mod.MultimodalHopper(num_modes, **render_kwargs)
        except TypeError:
            return hopper_mod.MultimodalHopper(num_modes=num_modes)
    if env_name == "HalfCheetah-v5":
        try:
            return halfcheetah_mod.MultimodalHalfCheetah(num_modes, **render_kwargs)
        except TypeError:
            return halfcheetah_mod.MultimodalHalfCheetah(num_modes=num_modes)
    raise ValueError(f"Unsupported env: {env_name}")

def encoder_training(
    env_id: str,
    encoder: nn.Module,
    dataloader: data.DataLoader,
    device: th.device,
    epochs: int = 150,
    lr: float = 1e-3,
    alpha: float = 0.3,    # contrastive
    beta: float = 1.0,   # InfoMax
    gamma: float = 0.5,   # segmentation
    zeta: float = 0.0,   # NN
):
    """
    Encoder-only training (no decoder, no reconstruction loss).
    Uses: SimCSE-style contrastive on CLS, DeepInfoMax(global-local), optional segment contrastive,
    K-Means init + CDEC-style clustering loss in formal stage.
    Returns: (encoder, None, None, cluster_centroids, None)
    """
    
    tau = 0.3 if env_id == "Walker2d-v4" else 0.3
    infomax_loss_fn = DeepInfoMaxLoss(encoder.d_model).to(device)
    contrastive_loss_fn = InstanceLoss(temperature=tau, device=device)
    optimizer = th.optim.Adam(
        list(encoder.parameters()) + list(infomax_loss_fn.discriminator.parameters()),
        lr=lr
    )

    # ---------------- Stage 1: Pre-Training (no recon) ----------------
    epoch_pbar = tqdm(range(epochs), desc="Encoder Training")
    n_seg = 4 if env_id == "Walker2d-v4" else 4
    for epoch in epoch_pbar:
        encoder.train(); infomax_loss_fn.discriminator.train()
        total_loss = 0.0
        loss_contrastive = th.tensor(0.0)
        loss_infomax = th.tensor(0.0)
        loss_seg = th.tensor(0.0)

        for batch in dataloader:
            # Handle both 5-element (old) and 6-element (new with is_old_flags) batches
            if len(batch) == 6:
                states, actions, masks, _, _, _ = [b.to(device) for b in batch]
            else:
                states, actions, masks, _, _ = [b.to(device) for b in batch]
            
            # segment lengths
            L_min, L_max = (
                (8, 16) if env_id in ("Reacher-v4","Pusher-v4")
                else (16, 32) if env_id == "Walker2d-v4" else (8, 16)
            )
            T = states.shape[1]
            current_L_max = min(L_max, T - 1)
            L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min

            optimizer.zero_grad()

            # Two stochastic passes (dropout)
            norm1, _, _, _, cls1, _ = encoder(states, actions, src_key_padding_mask=masks)
            norm2, _, _, _, cls2, _ = encoder(states, actions, src_key_padding_mask=masks)

            # Deep InfoMax on local tokens (exclude CLS @ pos 0)
            local_mask1 = expand_interleaved_mask(masks)
            local_mask2 = expand_interleaved_mask(masks)
            if beta > 0.0:
                loss_infomax = (
                    infomax_loss_fn(cls1, norm1[:, 1:, :], local_mask1) +
                    infomax_loss_fn(cls2, norm2[:, 1:, :], local_mask2)
                ) / 2.0

            # Instance contrastive on CLS
            if alpha > 0.0:
                loss_contrastive = contrastive_loss_fn(cls1, cls2)

            # Segment contrastive (optional)
            if gamma > 0.0:
                loss_seg_1 = segment_contrastive_loss(cls1, encoder, states, actions, masks, L=L,
                                                    temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=n_seg,
                                                    pairwise_segments=True)[0]
                loss_seg_2 = segment_contrastive_loss(cls2, encoder, states, actions, masks, L=L,
                                                    temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=n_seg,
                                                    pairwise_segments=True)[0]
                loss_seg = (loss_seg_1 + loss_seg_2) / 2.0

            loss = alpha * loss_contrastive + beta * loss_infomax + gamma * loss_seg
            loss.backward()
            th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_pbar.set_description(
            f"PT {epoch+1}/{epochs} Avg: {total_loss/len(dataloader):.4f} "
            f"Ct:{alpha*loss_contrastive.item():.4f} DIM:{beta*loss_infomax.item():.4f} SG:{gamma*loss_seg.item():.4f}"
        )
 
    return encoder

def encoder_finetuning(
    env_id: str,
    encoder: nn.Module,
    dataloader: data.DataLoader,
    K: int,
    device: th.device,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha: float = 0.3,    # contrastive
    beta: float = 1.0,   # InfoMax
    gamma: float = 0.5,   # segmentation
    delta: float = 1.0,   # stability
):
    """
    Encoder-only finetuning (no decoder, no reconstruction loss).
    Uses: SimCSE-style contrastive on CLS, DeepInfoMax(global-local), optional segment contrastive,
    K-Means init + CDEC-style clustering loss in formal stage.
    Returns: (encoder, None, None, cluster_centroids, None)
    """
    if getattr(encoder, "model_type", "") == "BE_SA_VDT":
        raise ValueError("Encoder-only loop not supported for BE_SA_VDT (variational) encoders. Use alpha=0 in the AE loop instead.")

    frozen_encoder = copy.deepcopy(encoder)
    frozen_encoder.eval()
    for param in frozen_encoder.parameters():
        param.requires_grad = False
    finetuning_encoder = copy.deepcopy(encoder)
    finetuning_encoder.train()

    STABILITY_TYPE = "cosine"

    tau = 0.3 if env_id == "Walker2d-v4" else 0.3
    contrastive_loss_fn = InstanceLoss(temperature=tau, device=device)
    infomax_loss_fn = DeepInfoMaxLoss(encoder.d_model).to(device)
    stability_loss = nn.MSELoss() if STABILITY_TYPE == "mse" else stability_loss_fn_cos
    optimizer = th.optim.Adam(
        list(finetuning_encoder.parameters()) + list(infomax_loss_fn.discriminator.parameters()),
        lr=lr
    )

    # ---------------- Finetuning ----------------
    epoch_pbar = tqdm(range(epochs), desc="Finetuning Encoder ")
    for epoch in epoch_pbar:
        encoder.train(); infomax_loss_fn.discriminator.train()
        total_loss = 0.0
        loss_contrastive = th.tensor(0.0)
        loss_infomax = th.tensor(0.0)
        loss_seg = th.tensor(0.0)
        loss_stability = th.tensor(0.0)

        for batch in dataloader:
            states, actions, masks, _, _, is_old_flags = [b.to(device) for b in batch]
            L_min, L_max = (
                (8, 16) if env_id in ("Reacher-v4","Pusher-v4")
                else (16, 32) if env_id == "Walker2d-v4" else (8, 16)
            )
            T = states.shape[1]
            current_L_max = min(L_max, T - 1)
            L = th.randint(L_min, current_L_max + 1, (1,)).item() if current_L_max >= L_min else L_min

            optimizer.zero_grad()

            # Two stochastic passes (dropout)
            norm1, _, _, _, cls1, _ = finetuning_encoder(states, actions, src_key_padding_mask=masks)
            norm2, _, _, _, cls2, _ = finetuning_encoder(states, actions, src_key_padding_mask=masks)

            # Deep InfoMax on local tokens (exclude CLS @ pos 0)
            local_mask1 = expand_interleaved_mask(masks)
            local_mask2 = expand_interleaved_mask(masks)
            if beta > 0.0:
                loss_infomax = (
                    infomax_loss_fn(cls1, norm1[:, 1:, :], local_mask1) +
                    infomax_loss_fn(cls2, norm2[:, 1:, :], local_mask2)
                ) / 2.0

            # Instance contrastive on CLS
            if alpha > 0.0:
                loss_contrastive = contrastive_loss_fn(cls1, cls2)

            # Segment contrastive (optional)
            if gamma > 0.0:
                loss_seg_1 = segment_contrastive_loss(cls1, finetuning_encoder, states, actions, masks, L=L,
                                                    temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=4,
                                                    pairwise_segments=True)[0]
                loss_seg_2 = segment_contrastive_loss(cls2, finetuning_encoder, states, actions, masks, L=L,
                                                    temperature=tau, contrastive_loss_fn=contrastive_loss_fn, num_segments=4,
                                                    pairwise_segments=True)[0]
                # loss_seg_1 = segment_infomax_and_contrastive(cls1,encoder,states,actions,masks,L,infomax_loss_seg_fn,contrastive_loss_fn,num_segments=4,pairwise_segments=True)[0]
                # loss_seg_2 = segment_infomax_and_contrastive(cls2,encoder,states,actions,masks,L,infomax_loss_seg_fn,contrastive_loss_fn,num_segments=4,pairwise_segments=True)[0]
                loss_seg = (loss_seg_1 + loss_seg_2) / 2.0

            # Use the boolean flags to select only the old trajectories
            old_states,old_actions,old_masks = states[is_old_flags], actions[is_old_flags], masks[is_old_flags]

            # Only compute stability loss if there are old samples in the batch
            if old_states.shape[0] > 0:
                # Get their embeddings from the new, training encoder
                _, _, _, _, old_cls_ft, _ = finetuning_encoder(old_states, old_actions, src_key_padding_mask=old_masks)

                # Get their original embeddings from the frozen encoder
                # Use torch.no_grad() to be absolutely sure no gradients are computed for E_old
                with th.no_grad():
                    _, _, _, _, old_cls_fz, _ = frozen_encoder(old_states, old_actions, src_key_padding_mask=old_masks)

                # Calculate the stability loss
                loss_stability = stability_loss(old_cls_ft, old_cls_fz)

            loss = alpha * loss_contrastive + beta * loss_infomax + gamma * loss_seg + delta * loss_stability
            loss.backward()
            th.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_pbar.set_description(
            f"PT {epoch+1}/{epochs} Avg: {total_loss/len(dataloader):.4f} "
            f"Ct:{alpha*loss_contrastive.item():.4f} DIM:{beta*loss_infomax.item():.4f} SG:{gamma*loss_seg.item():.4f} ST:{delta*loss_stability.item():.4f}"
        )

    return finetuning_encoder

def train_and_evaluate_irl_agent(
    cluster_id: int,
    trajectories: list,
    trajectories_with_rew: list,
    true_labels: list,
    env_name: str,
    env_id: str,
    tr_name: str,
    seed: int,
    base_learner_policy,
    stage:str,
    rl_timesteps: int = 1_000_000,
):
    """
    Trains and evaluates an IRL agent for a specific cluster of trajectories.

    Args:
        cluster_id: The ID of the cluster being trained.
        trajectories: The list of expert trajectories for this cluster.
        trajectories_with_rew: Trajectories with original rewards stored.
        true_labels: The list of ground-truth expert modes for the trajectories.
        env_name: The name of the environment (e.g., "Reacher-v4").
        env_id: The gym environment ID (e.g., "Reacher-v4").
        tr_name: The IRL algorithm to use ('gail', 'airl', 'sqil').
        seed: The random seed for reproducibility.
        long: Whether to use long dataset.
        base_learner_policy: The SB3 policy class (e.g., MlpPolicy).
        stage: The training stage (e.g., "baseline", "finetuned_novel").
        rl_timesteps: The number of timesteps for RL training.
        render: Whether to render the environment during evaluation.

    Returns:
        A dictionary containing the expert and learner reward statistics.
    """
    print(f"\n--- Processing Cluster {cluster_id} (Stage: {stage}) for {env_name} using {tr_name.upper()} ---")
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 1. Determine the dominant expert mode for this cluster
    if not true_labels:
        print(f"Warning: Could not determine expert mode for cluster {cluster_id} because no labels were provided. Skipping.")
        return None
    
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

    expert_mode_idx = Counter(true_labels).most_common(1)[0][0] - 10 #-10 because the labels have a +10 offset
    # print(np.unique(true_labels,return_counts=True))
    print(f"Cluster {cluster_id} corresponds to expert mode {expert_mode_idx} ({len(trajectories)} trajectories).")

    # 2. Setup Environment
    # This part needs to be adapted based on how your custom environments are instantiated.
    num_modes = 3 if env_name == "Walker2d-v4" else 6
    env = make_env_by_name(env_name, num_modes)
    try:
        env.reset(mode_idx=expert_mode_idx)
    except Exception:
        _ = env.reset()
    venv = make_vec_env(lambda: make_env_by_name(env_name, num_modes), n_envs=1, seed=seed)

    # EnvClass = {
    #     "Pusher-v4": pusher_mod.MultimodalPusher,
    #     "Walker2d-v4": walker2d_mod.MultimodalWalker,
    #     "Reacher-v4": reacher_mod.MultimodalReacher,
    # }[env_name]
    # if env_name in ["Walker2d-v4"]:
    #     env = FixedLengthEnvWrapper(EnvClass(num_modes=num_modes))
    #     env.reset(mode_idx=expert_mode_idx)
    #     venv = make_vec_env(lambda: FixedLengthEnvWrapper(EnvClass(num_modes=num_modes)), n_envs=1, seed=seed)
    # else:
    #     env = EnvClass(num_modes=num_modes)
    #     env.reset(mode_idx=expert_mode_idx)
    #     venv = make_vec_env(lambda: EnvClass(num_modes=num_modes), n_envs=1, seed=seed)

    # 3. Calculate Expert Reward & Normalizer
    avg_expert_reward, std_expert_reward = calculate_original_expert_reward_stats(trajectories_with_rew)
    print(f"  Expert Reward: {avg_expert_reward:.4f} ± {std_expert_reward:.4f}")
    # 4. Setup and Train IRL Agent
    learner_dir = f"./learners/{env_id}/{seed}/{stage}/"
    ft_suffix = "_FT" if stage == "finetuned_novel" else ""
    learner_filename = f"{tr_name}_cluster_{cluster_id}_mode_{expert_mode_idx}{ft_suffix}.zip"
    learner_save_path = os.path.join(learner_dir, learner_filename)
    os.makedirs(os.path.dirname(learner_save_path), exist_ok=True)

    if os.path.exists(learner_save_path):
        print(f"  Loading existing learner from {learner_save_path}")
        if tr_name in ['gail','airl']:
            learner = PPO.load(learner_save_path, env=venv)
            reward_net = th.load(learner_save_path.replace(".zip", "_reward_net.zip"))
            reward_net.load_state_dict(th.load(learner_save_path.replace(".zip", "_reward_net_state_dict.zip")))
            reward_net.eval()
        elif tr_name == "sqil":
            loaded_sac = SAC.load(learner_save_path, env=env)
            learner = loaded_sac.policy
    else:
        print(f"  Training new learner. Will save to {learner_save_path}")
        # Define a base RL learner. You can customize hyperparameters here.
        if env_name == "HalfCheetah-v5":
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
        elif env_name == "Hopper-v4":
            policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
        else:
            policy_kwargs = dict()
        learner = PPO(
            env=venv, policy=base_learner_policy, batch_size=64, ent_coef=0.01,
            learning_rate=0.0003, gamma=0.99, clip_range=0.2, n_epochs=10, seed=seed,policy_kwargs=policy_kwargs,
            verbose=0
        )
        reward_hid = (32,)
        potential_hid =(32, 32)

        reward_net = BasicShapedRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
                reward_hid_sizes=reward_hid,
                potential_hid_sizes=potential_hid,
            )
        training_func = AIRL if tr_name == "airl" else GAIL if tr_name == "gail" else None
        if tr_name == "sqil":
            trainer = sqil.SQIL(
                demonstrations=trajectories,
                venv=venv,
                rl_algo_class=SAC,
                policy="MlpPolicy",
            )
        else:
            trainer = training_func(
                    demonstrations=trajectories,
                    demo_batch_size=batch_size,
                    gen_replay_buffer_capacity=2048,
                    n_disc_updates_per_round=32 if env_name not in ["Hopper-v4", "HalfCheetah-v5"] else 8,
                    venv=venv,
                    gen_algo=learner,
                    reward_net=reward_net,
                )     

        trainer.train(total_timesteps=rl_timesteps)
        if tr_name == "sqil":
            trainer.rl_algo.save(learner_save_path)
        else:
        # trainer.policy.save(learner_save_path)
        # learner = trainer.policy
            learner.save(learner_save_path)
            th.save(learner.policy.state_dict(), learner_save_path.replace(".zip", "_policy.zip"))
            th.save(reward_net.state_dict(), learner_save_path.replace(".zip", "_reward_net_state_dict.zip"))
            th.save(reward_net, learner_save_path.replace(".zip", "_reward_net.zip"))

    # 5. Evaluate Learner (now with normalizer)
    avg_learner_reward, std_learner_reward = evaluate_policy_reward(
        learner, env, num_episodes=max(5, min(20, len(trajectories))),
        mode_idx=expert_mode_idx, env_name=env_name
    )
    print(f"  Learner Reward: {avg_learner_reward:.4f} ± {std_learner_reward:.4f}")
    normalized_ratio = (avg_learner_reward / avg_expert_reward) if avg_expert_reward != 0 else 0.0

    return {
        "Expert Mode": expert_mode_idx,
        "Expert Reward Mean": avg_expert_reward,
        "Expert Reward Std": std_expert_reward,
        "Learner Reward Mean": avg_learner_reward,
        "Learner Reward Std": std_learner_reward,
        "Normalized Learner Reward": normalized_ratio,
    }

