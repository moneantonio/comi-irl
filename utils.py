import math
import os
import pandas as pd # type: ignore[import]
from collections import Counter
import numpy as np # type: ignore[import]
import torch as th # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import torch.nn.functional as F # type: ignore[import]
from torch.utils import data # type: ignore[import]
from torch.utils.data import DataLoader # type: ignore[import]
from sklearn.preprocessing import ( #type: ignore[import]
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from tqdm import tqdm # type: ignore[import]
from imitation.data.types import Trajectory, TrajectoryWithRew # type: ignore[import]
from datasets import TrajectoryDataset, TrajectoryDatasetSeenUnseen
from sklearn.covariance import LedoitWolf # type: ignore[import]
from sklearn.feature_selection import chi2 # type: ignore[import]
from sklearn.cluster import HDBSCAN # type: ignore[import]
import matplotlib.pyplot as plt # type: ignore[import]
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment # type: ignore[import]
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score # type: ignore[import]

def save_ablation_results(env_id: str, loss_type: str, seed: int, nmi: float, ari: float, silhouette: float, stage: str = "baseline"):
    """
    Save ablation clustering metrics to a CSV file for later aggregation.
    One file per environment: ablation_results_{env_id}.csv
    """
    ablation_dir = "./ablation_results"
    os.makedirs(ablation_dir, exist_ok=True)
    
    csv_path = os.path.join(ablation_dir, f"ablation_results_{env_id}.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_path)
    
    # Create the record
    record = {
        "env_id": env_id,
        "loss_type": loss_type,
        "seed": seed,
        "stage": stage,
        "NMI": nmi,
        "ARI": ari,
        "Silhouette": silhouette,
    }
    
    # Append to CSV
    df_new = pd.DataFrame([record])
    df_new.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    print(f"[Ablation] Saved results to {csv_path}: {loss_type} seed={seed} NMI={nmi:.4f} ARI={ari:.4f} Sil={silhouette:.4f}")


def calculate_original_expert_reward_stats(trajectories_with_rew: list[TrajectoryWithRew]):
    """
    Calculates the mean and standard deviation of rewards directly from a list
    of trajectories that have rewards stored in them. This avoids the
    stochasticity of replaying actions in the environment.
    """
    total_rewards = [np.sum(traj.rews) for traj in trajectories_with_rew if hasattr(traj, 'rews') and traj.rews is not None]
    if not total_rewards:
        return 0.0, 0.0
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

def prepare_sa_trajectories(
    env_id: str,
    trajectories: list[Trajectory],
    true_labels: np.ndarray,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Prepares state-action trajectories for a typed transformer model.
    This function handles padding for variable-length trajectories and separates
    states, actions, and labels into distinct tensors. It also masks out any steps
    that occur after the first time a known goal state is reached. For environments
    like Traj2d with variable lengths, it pads shorter trajectories by repeating the
    last valid state and action.

    Args:
        env_id (str): The ID of the environment, used to identify the goal state.
        trajectories (list): A list of imitation.Trajectory objects.
        true_labels (np.ndarray): An array of ground-truth integer labels for each trajectory.
        max_len (Optional[int]): The maximum length to pad/truncate to. If None,
                                 it's determined by the longest trajectory.

    Returns:
        A tuple containing:
        - all_states (th.Tensor): Padded state sequences. Shape: [N, T, ...state_dims].
        - all_actions (th.Tensor): Padded action sequences. Shape: [N, T].
        - all_masks (th.Tensor): Boolean mask, True for padded/post-goal elements. Shape: [N, T].
        - all_labels (th.Tensor): Ground-truth labels. Shape: [N].
    """
    if not trajectories:
        return th.empty(0), th.empty(0), th.empty(0), th.empty(0)

    max_len = max(len(traj.obs) for traj in trajectories)
    min_len = min(len(traj.obs) for traj in trajectories)

    # Define ending_state based on env_id
    ending_state = None
    if env_id == "PuddleWorld-v0":
        ending_state = np.array([2, 4])
        # ending_state = np.array([3, 6])
    elif env_id == "TwoLakesFishing-v0":
        ending_state = np.array([6, 3, 1])
    elif env_id == "ConditionalAssemblyLine-v0":
        ending_state = np.array([3, 14, 0])

    # Infer shapes and dtypes from the first trajectory
    first_traj = trajectories[0]
    state_shape = np.array(first_traj.obs).shape[1:]
    state_dtype = th.float32
    action_dtype = th.int64

    num_trajs = len(trajectories)
    all_states = th.zeros((num_trajs, max_len, *state_shape), dtype=state_dtype)
    # all_actions = th.zeros((num_trajs, max_len), dtype=action_dtype)
    first_act = np.array(trajectories[0].acts)
    if first_act.ndim == 0 or first_act.ndim == 1:
        # discrete: shape (L,) ints
        discrete = True
        action_dim = 1
        all_actions = th.zeros((num_trajs, max_len), dtype=th.long)
    else:
        # continuous: shape (L, A) floats
        discrete = False
        action_dim = first_act.shape[1]
        all_actions = th.zeros((num_trajs, max_len, action_dim),
                                  dtype=th.float32)
    all_masks = th.ones((num_trajs, max_len), dtype=th.bool)  # True means padded/masked

    for i, traj in enumerate(tqdm(trajectories, desc="Preparing State-Action Trajectories")):
        obs_np = np.array(traj.obs)
        acts_np = np.array(traj.acts)
        original_seq_len = len(obs_np)

        # Determine the valid length of the trajectory (before padding)
        valid_len = original_seq_len
        if ending_state is not None:
            matches = np.where(np.all(obs_np == ending_state, axis=1))[0]
            if len(matches) > 0:
                goal_index = matches[0]
                valid_len = goal_index + 1
        
        # The effective length is the part of the trajectory we will use, capped by max_len
        effective_len = min(valid_len, max_len)
        if effective_len == 0:
            continue

        # --- Fill valid part of the tensors ---
        
        # States
        all_states[i, :effective_len] = th.tensor(obs_np[:effective_len], dtype=state_dtype)

        if discrete:
            # previous logic for ints
            if len(acts_np) > 0:
                # same as before, but wrap in a 1-D view
                num_copy = min(len(acts_np), effective_len-1)
                if num_copy > 0:
                    a = th.tensor(acts_np[:num_copy], dtype=th.long)
                    all_actions[i, :num_copy] = a
                    last = a[-1]
                    all_actions[i, num_copy:effective_len] = last
                else:
                    # only one step
                    all_actions[i, :effective_len] = int(acts_np[0])
            else:
                all_actions[i, :effective_len] = 0
        else:
            # continuous
            if acts_np.ndim == 1:
                # flatten scalar per step → make it [L,1]
                acts_np = acts_np.reshape(-1, 1)
            # now acts_np is [L, action_dim]
            num_copy = min(len(acts_np), effective_len-1)
            if num_copy > 0:
                a = th.tensor(acts_np[:num_copy], dtype=th.float32)  # [num_copy, A]
                all_actions[i, :num_copy, :] = a
                last = a[-1]  # [A]
                all_actions[i, num_copy:effective_len, :] = last
            else:
                # only one step
                all_actions[i, :effective_len, :] = th.tensor(acts_np[0], dtype=th.float32)


        # Mask: False for valid steps, True for padded/post-goal steps
        all_masks[i, :effective_len] = False

        # --- Fill padded part of the tensors ---
        if effective_len < max_len:
            last_state_val = all_states[i, effective_len - 1]
            last_action_val = all_actions[i, effective_len - 1]
            
            all_states[i, effective_len:] = last_state_val
            all_actions[i, effective_len:] = last_action_val

    all_labels = th.tensor(true_labels, dtype=th.long)

    return all_states, all_actions, all_masks, all_labels, max_len

def infer_dims(traj: Trajectory): #for matrix
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

def interleave_flatten(traj: Trajectory, max_steps: int, pad_value: float = 0.0): #for matrix
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

def is_coord_states(x) -> bool:
    x = np.asarray(x)
    return x.ndim == 2  # [T, D]

def is_cont_actions(a_list) -> bool:
    a0 = np.asarray(a_list[0])
    return a0.ndim == 2 and np.issubdtype(a0.dtype, np.floating)

def build_scaler(name: str, seed: int = 0, feature_range=(-1.0, 1.0), n_quantiles: int = 1000):
    name = name.lower()
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler(feature_range=tuple(feature_range))
    if name == 'maxabs': return MaxAbsScaler()
    if name == 'robust': return RobustScaler(quantile_range=(25, 75))
    if name == 'quantile_uniform': return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform', random_state=seed)
    if name == 'quantile_normal': return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=seed)
    if name == 'power_yeo': return PowerTransformer(method='yeo-johnson', standardize=True)
    if name == 'l2norm': return Normalizer(norm='l2')
    if name == 'none': return None
    raise ValueError(f"Unknown scaler: {name}")

def fit_on_flat(list_of_td, scaler, winsorize_bounds=(1, 99)):
    """Fit scaler on training data only and return clipping bounds."""
    # Flatten: stack all [T,D] -> [sum_T, D]
    X_train = np.vstack([np.asarray(s) for s in list_of_td])
    scaler.fit(X_train)

    # Determine Winsorization bounds from the transformed training data
    X_train_transformed = scaler.transform(X_train)
    min_bound = np.percentile(X_train_transformed, winsorize_bounds[0], axis=0)
    max_bound = np.percentile(X_train_transformed, winsorize_bounds[1], axis=0)
    
    return scaler, (min_bound, max_bound)

def apply_per_traj(list_of_td, scaler, clip_range=None):
    if scaler is None: return list_of_td
    out = []
    for s in list_of_td:
        s_np = np.asarray(s)
        if s_np.ndim != 2:
            out.append(s)  # leave grids or discrete as-is
            continue
        T, D = s_np.shape
        transformed_s = scaler.transform(s_np.reshape(-1, D))
        if clip_range is not None:
            min_val, max_val = clip_range
            np.clip(transformed_s, min_val, max_val, out=transformed_s)
        out.append(transformed_s.reshape(T, D))
    return out

def datasets_preparation_sa(
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    labels: th.Tensor,
    train_size: int,
    val_size: int,
    test_size: int,
    loader_batch: int,
    val_bptt: int,
    test_bptt: int,
    seed: int = 42,
) -> tuple:
    """
    Creates train, validation, and test dataloaders from prepared state-action tensors.
    This is the new equivalent of your `datasets_preparation` function.

    Args:
        states, actions, masks, labels: Tensors from prepare_sa_data_from_trajectories.
        train_size, val_size, test_size: Number of samples for each split.
        loader_batch, val_bptt, test_bptt: Batch sizes for the dataloaders.
        seed: Random seed for splitting.

    Returns:
        A tuple matching the original function's output structure for easy integration:
        (full_dataset, train_dataset, val_dataset, test_dataset,
         total_dataloader, train_dataloader, val_dataloader, test_dataloader)
    """
    # Create the full dataset using the new TrajectoryDataset class
    full_dataset = TrajectoryDataset(states, actions, masks, labels)

    # Split the dataset
    generator = th.Generator().manual_seed(seed)
    print((len(full_dataset), train_size, val_size, test_size))
    train_dataset, val_dataset, test_dataset = th.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    total_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=loader_batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_bptt, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_bptt, shuffle=False)

    return full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader

def datasets_preparation_sa_seen_unseen(
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    labels: th.Tensor,
    train_size: int,
    val_size: int,
    test_size: int,
    loader_batch: int,
    val_bptt: int,
    test_bptt: int,
    seed: int = 42,
    is_old_data: bool = False
) -> tuple:
    """
    Creates train, validation, and test dataloaders from prepared state-action tensors.
    This is the new equivalent of your `datasets_preparation` function.

    Args:
        states, actions, masks, labels: Tensors from prepare_sa_data_from_trajectories.
        train_size, val_size, test_size: Number of samples for each split.
        loader_batch, val_bptt, test_bptt: Batch sizes for the dataloaders.
        seed: Random seed for splitting.

    Returns:
        A tuple matching the original function's output structure for easy integration:
        (full_dataset, train_dataset, val_dataset, test_dataset,
         total_dataloader, train_dataloader, val_dataloader, test_dataloader)
    """
    # Create the full dataset using the new TrajectoryDataset class
    full_dataset = TrajectoryDatasetSeenUnseen(states, actions, masks, labels,is_old_data)

    # Split the dataset
    generator = th.Generator().manual_seed(seed)
    print((len(full_dataset), train_size, val_size, test_size))
    train_dataset, val_dataset, test_dataset = th.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    total_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=loader_batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_bptt, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_bptt, shuffle=False)

    return full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader

def expand_interleaved_mask(original_mask: th.Tensor) -> th.Tensor:
    """
    Expands a mask for interleaved states and actions.
    
    Args:
        original_mask (Tensor): Original mask of shape [B, T] (e.g., [B, 126]).
    
    Returns:
        Tensor: Expanded mask of shape [B, 2*T] (e.g., [B, 252]).
    """
    B, T = original_mask.shape
    # Repeat each mask element twice (for state and action)
    interleaved_mask = original_mask.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * T)
    return interleaved_mask

def fit_hdbscan_seen(X_seen: np.ndarray, granularity: float, seed: int):
    nX = len(X_seen)
    min_cluster_size = max(5, int(granularity * nX))
    min_samples = max(1, int(math.sqrt(min_cluster_size)))
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cosine'
    ).fit(X_seen)
    labels = model.labels_
    unique_core = np.unique(labels[labels != -1])
    # cluster centers as medians (robust), renormalize
    centers = []
    for c in unique_core:
        pts = X_seen[labels == c]
        centers.append(np.median(pts, axis=0))
    if len(centers) > 0:
        centers = np.vstack(centers)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    else:
        centers = np.zeros((0, X_seen.shape[1]))
    return model, labels, unique_core, centers

def build_registry(X_seen: np.ndarray, labels: np.ndarray, core_ids: np.ndarray, centers: np.ndarray) -> dict:
    # For each cluster: store center, cosine radius (95th pct), Mahalanobis stats, and ECDF for cosine distances
    reg = {}
    for i, cid in enumerate(core_ids):
        pts = X_seen[labels == cid]
        if len(pts) == 0:
            continue
        c = centers[i]  # unit-norm center
        # L2-normalize points for cosine geometry
        pts_n = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
        # cosine distance on unit sphere: d = 1 - cos
        cos = np.clip(pts_n @ c, -1.0, 1.0)
        d_cos = 1.0 - cos
        r95 = float(np.quantile(d_cos, 0.95))  # acceptance radius
        dcos_sorted = np.sort(d_cos.astype(np.float32))  # for ECDF lookup

        # covariance for Mahalanobis on normalized space (shrinkage)
        try:
            lw = LedoitWolf().fit(pts_n)
            mu = lw.location_
            prec = lw.precision_
        except Exception:
            print("Registry exception")
            mu = pts_n.mean(axis=0)
            cov = np.cov(pts_n.T) + 1e-6 * np.eye(pts_n.shape[1])
            prec = np.linalg.pinv(cov)
        reg[int(cid)] = {
            "center": c,   # unit
            "r95": r95,
            "mu": mu,      # normalized space
            "prec": prec,  # normalized space
            "count": len(pts),
            "dcos_sorted": dcos_sorted,
        }
    return reg

def visualize_controller_output(
    Z_seen: np.ndarray,
    Z_online: np.ndarray,
    labels_seen: np.ndarray,
    labels_online: np.ndarray,
    novelty_scores: np.ndarray,
    registry: dict,
    reducer,
    title: str,
    is_3d: bool = False,
    palette: list = None
):
    """Visualizes the output of the controller logic.

    Uses the provided palette or HIGH_CONTRAST_PREDICTED_COLORS (if available)
    to assign consistent, colorblind-friendly colors to clusters.
    """
    # determine palette (hex strings or matplotlib colors). fallback to tab10 if missing.
    if palette is None:
        palette = globals().get("HIGH_CONTRAST_PREDICTED_COLORS", None)
    if palette is None:
        # fallback palette using matplotlib's tab10
        palette = [plt.cm.tab10(i) for i in range(10)]

    # build deterministic mapping from cluster id -> palette color
    seen_ids = np.unique(labels_seen) if labels_seen is not None else np.array([])
    online_ids = np.unique(labels_online) if labels_online is not None and len(labels_online) > 0 else np.array([])
    all_ids = sorted(set(int(x) for x in np.concatenate([seen_ids, online_ids]) if int(x) != -1))
    id_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(all_ids)}
    # noise color
    noise_color = 'gray'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    # 1. Plot Seen Data (training set for the controller)
    for cid in np.unique(labels_seen):
        mask = labels_seen == cid
        if int(cid) == -1:
            color, marker, label = noise_color, 'x', "Seen (Noise)"
        else:
            color, marker, label = id_to_color.get(int(cid), palette[0]), 'o', f"Seen (Cluster {int(cid)})"

        if is_3d:
            ax.scatter(Z_seen[mask, 0], Z_seen[mask, 1], Z_seen[mask, 2], c=[color], s=20, alpha=0.6, marker=marker, label=label)
        else:
            ax.scatter(Z_seen[mask, 0], Z_seen[mask, 1], c=[color], s=20, alpha=0.6, marker=marker, label=label)

    # 2. Plot Online Data
    if len(Z_online) > 0:
        # Assigned points
        assigned_mask = labels_online != -1
        for cid in np.unique(labels_online[assigned_mask]):
            mask = labels_online == cid
            color, marker, label = id_to_color.get(int(cid), palette[0]), '^', f"Online (Assigned to {int(cid)})"
            if is_3d:
                ax.scatter(Z_online[mask, 0], Z_online[mask, 1], Z_online[mask, 2], c=[color], s=60, alpha=0.9, marker=marker, label=label)
            else:
                ax.scatter(Z_online[mask, 0], Z_online[mask, 1], c=[color], s=60, alpha=0.9, marker=marker, label=label)

        # Novel points (sized by novelty score)
        novel_mask = labels_online == -1
        if np.any(novel_mask):
            # Normalize scores for better size visibility
            scores_norm = (novelty_scores[novel_mask] - novelty_scores[novel_mask].min()) / (novelty_scores[novel_mask].max() - novelty_scores[novel_mask].min() + 1e-6)
            sizes = 50 + 200 * scores_norm

            if is_3d:
                p = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], Z_online[novel_mask, 2], c='red', s=sizes, marker='*', label='Online (Novel)')
            else:
                p = ax.scatter(Z_online[novel_mask, 0], Z_online[novel_mask, 1], c='red', s=sizes, marker='*', label='Online (Novel)')

            # Add a colorbar legend for the novelty scores
            cbar = plt.colorbar(p, ax=ax, shrink=0.6)
            cbar.set_label('Novelty Score')
            # Set ticks to show the original score range
            ticks = np.linspace(novelty_scores[novel_mask].min(), novelty_scores[novel_mask].max(), 5)
            cbar.set_ticks(np.linspace(0, 1, 5))
            cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    # 3. Plot Cluster Centers from Registry
    if registry:
        centers_emb = np.stack([v['center'] for v in registry.values()])
        Z_centers = reducer.transform(centers_emb)
        # map registry keys to colors (registry order may differ - use sorted keys)
        reg_keys = sorted(registry.keys())
        for i, rk in enumerate(reg_keys):
            color = id_to_color.get(int(rk), palette[i % len(palette)])
            if is_3d:
                ax.scatter(Z_centers[i:i+1, 0], Z_centers[i:i+1, 1], Z_centers[i:i+1, 2], c=[color], s=250, marker='P', edgecolor='white', label=f'Cluster Center {int(rk)}')
            else:
                ax.scatter(Z_centers[i:i+1, 0], Z_centers[i:i+1, 1], c=[color], s=250, marker='P', edgecolor='white', label=f'Cluster Center {int(rk)}')

    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    if is_3d: ax.set_zlabel("UMAP-3")
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

def save_cluster_splits(
    env_id: str,
    seed: int,
    trajectory_manager: dict,
    labels: np.ndarray,
    indices: np.ndarray,
    registry: dict,
    phase_name: str = "phase1_baseline",
    output_root: str = "./cluster_splits"
):
    """
    Save per-cluster trajectory splits to disk.
    
    Args:
        env_id: Environment identifier (e.g., "Reacher-v4")
        seed: Random seed for reproducibility
        trajectory_manager: Dictionary mapping trajectory indices to trajectory data
        labels: Cluster labels for each trajectory (shape: [N,])
        indices: Trajectory indices corresponding to labels (shape: [N,])
        registry: Dictionary mapping cluster IDs to cluster metadata
        phase_name: Name of the training phase (default: "phase1_baseline")
        output_root: Root directory for saving clusters (default: "./cluster_splits")
    
    Returns:
        List of saved file paths
    """
    
    out_dir = os.path.join(output_root, env_id, str(seed), phase_name)
    os.makedirs(out_dir, exist_ok=True)
    
    reg_ids = sorted(registry.keys())
    saved_files = []
    
    print(f"[Save clusters] Writing per-cluster splits to {out_dir}")
    
    for cid in reg_ids:
        mask = (labels == cid)
        manager_indices = indices[mask]
        
        if len(manager_indices) == 0:
            continue
        
        # Gather trajectories and true modes (labels have +10 offset)
        cluster_trajectories = [
            trajectory_manager[int(i)]['original_trajectory'] 
            for i in manager_indices
        ]
        cluster_trajs_with_rew = [
            trajectory_manager[int(i)]['original_trajectory_with_rew'] 
            for i in manager_indices
        ]
        per_modes = [
            int(trajectory_manager[int(i)]['real_cluster_label']) - 10 
            for i in manager_indices
        ]
        
        # Determine dominant mode
        mode_counts = Counter(per_modes)
        dom_mode = mode_counts.most_common(1)[0][0] if mode_counts else -1
        
        # File paths
        prefix = f"cluster_{int(cid)}_mode_{int(dom_mode)}_n{len(manager_indices)}"
        path_trajs = os.path.join(out_dir, prefix + "_trajs.pth")
        path_trajs_rew = os.path.join(out_dir, prefix + "_trajs_withrew.pth")
        path_meta = os.path.join(out_dir, prefix + "_meta.pth")
        
        # Save payloads
        th.save(cluster_trajectories, path_trajs)
        th.save(cluster_trajs_with_rew, path_trajs_rew)
        
        meta = {
            "env_id": env_id,
            "seed": int(seed),
            "cluster_id": int(cid),
            "dominant_mode": int(dom_mode),
            "mode_counts": {int(k): int(v) for k, v in mode_counts.items()},
            "num_trajs": int(len(manager_indices)),
            "indices_in_trajectory_manager": [int(i) for i in manager_indices.tolist()],
            "phase": phase_name,
        }
        th.save(meta, path_meta)
        
        saved_files.extend([path_trajs, path_trajs_rew, path_meta])
        
        print(f"  Saved C{int(cid)} -> {len(manager_indices)} trajs | "
              f"dom mode={int(dom_mode)} | "
              f"files: {os.path.basename(path_trajs)}, {os.path.basename(path_trajs_rew)}")
    
    print(f"[Save clusters] Saved {len(reg_ids)} clusters to {out_dir}")
    return saved_files

def calculate_expert_reward(trajectories, env, mode_idx=None, env_name=""):
    """
    Calculates the average reward of a set of expert trajectories by replaying them.
    This version resets the environment to the trajectory's actual start state.
    """
    total_rewards = []
    for traj in trajectories:
        # Reset the environment to the trajectory's specific start state.
        initial_obs = traj.obs[0]
        if env_name == "Traj2d":
            env.unwrapped.reset(options={"mode_idx": mode_idx})
        elif env_name == "Reacher-v4":
            env.unwrapped.reset(mode_idx=mode_idx)
        else:
            env.unwrapped.reset(mode_idx=mode_idx, start_state=initial_obs)

        episode_reward = 0
        # Replay the actions from the trajectory
        for action in traj.acts:
            if env_name == "Traj2d" or env_name == "Reacher-v4":
                action = action
            else:
                if isinstance(action, np.ndarray):
                    action = int(action.item())
            
            # Use unwrapped env for step if it exists
            if hasattr(env.unwrapped, "step"):
                obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            reward = info['reward_eval']

            # if env_name == "Traj2d":
            #     reward = info['reward_eval']
            # elif env_name == "Reacher-v4":
            #     reward = info['reward_train']
            
            episode_reward += reward
            if terminated or truncated:
                break
        
        if env_name == "Traj2d":
            lap_lengths = [32, 63, 32, 63, 126, 126]
            if mode_idx is not None and 0 <= mode_idx < len(lap_lengths):
                max_steps = lap_lengths[mode_idx]
            else:
                # Fallback to the environment's max steps if mode_idx is invalid or not provided
                print("Expert reward fallback")
                max_steps = env.unwrapped._max_episode_steps if hasattr(env, 'unwrapped') else env._max_episode_steps
            # Average the normalized step rewards over the episode length
            total_rewards.append(episode_reward / max_steps)
        else:
            total_rewards.append(episode_reward)
    if env_name == "Traj2d":
        mean_reward = np.mean(np.exp(total_rewards)) if total_rewards else 0.0
        std_reward = np.std(np.exp(total_rewards)) if total_rewards else 0.0
    else:
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
    return mean_reward, std_reward

def evaluate_policy_reward(policy, env, num_episodes=50, mode_idx=None, env_name=""):
    """
    Simple evaluation:
      For Traj2d: reproduces previous exp(normalized_episode) behavior.
      For others: raw summed reward_eval per episode (no scaling).
    Returns:
      mean_reward, std_reward
    """
    returns = []
    use_traj2d = (env_name == "Traj2d")
    for _ in range(num_episodes):
        if env_name == "Traj2d":
            obs, info = env.unwrapped.reset(options={"mode_idx": mode_idx})
        elif env_name in ['Hopper-v4', 'HalfCheetah-v5']:
            obs, info = env.reset(options={"mode_idx": mode_idx})
        elif hasattr(env.unwrapped, "reset"):
            reset_out = env.unwrapped.reset(mode_idx=mode_idx)
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
        else:
            obs, _ = env.reset(mode_idx=mode_idx)

        terminated = False
        truncated = False
        episode_reward = 0.0
        max_steps = getattr(env.unwrapped, "_max_episode_steps", 1000) if hasattr(env, "unwrapped") else getattr(env, "_max_episode_steps", 1000)

        for _t in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            if hasattr(env.unwrapped, "step"):
                obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            reward = info.get("reward_eval", reward)
            episode_reward += reward
            if terminated or truncated:
                break

        if use_traj2d:
            lap_lengths = [32, 63, 32, 63, 126, 126]
            if mode_idx is not None and 0 <= mode_idx < len(lap_lengths):
                norm_steps = lap_lengths[mode_idx]
            else:
                norm_steps = max_steps
            normalized_episode = episode_reward / norm_steps
            returns.append(float(np.exp(normalized_episode)))
        else:
            returns.append(episode_reward)

    if len(returns) == 0:
        return 0.0, 0.0
    return float(np.mean(returns)), float(np.std(returns))

def assign_or_flag_online(X_online: np.ndarray, registry: dict, maha_p: float = 0.99):
    # Assign by nearest center (cosine over L2-normalized vectors), then gate; score by calibrated ECDFs
    if len(registry) == 0 or len(X_online) == 0:
        empty = np.full((len(X_online),), -1, dtype=int)
        return empty, np.ones((len(X_online),), dtype=bool), np.zeros((len(X_online),), dtype=float)
    cids = sorted(registry.keys())
    C = np.stack([registry[c]["center"] for c in cids], axis=0)  # [C, D], unit-norm
    Xn = X_online / (np.linalg.norm(X_online, axis=1, keepdims=True) + 1e-8)

    # nearest center by cosine similarity
    sims = Xn @ C.T  # [N, C]
    argmax = sims.argmax(axis=1)
    best_cids = np.array([cids[j] for j in argmax], dtype=int)
    best_centers = C[argmax]

    # cosine gate
    d_cos = 1.0 - np.clip(np.sum(Xn * best_centers, axis=1), -1.0, 1.0)
    r95 = np.array([registry[int(cid)]["r95"] for cid in best_cids])
    pass_cos = d_cos <= r95

    # Mahalanobis gate
    df = Xn.shape[1]
    chi_thr = float(chi2.ppf(maha_p, df))
    maha_vals = []
    for x, cid in zip(Xn, best_cids):
        mu = registry[int(cid)]["mu"]
        prec = registry[int(cid)]["prec"]
        diff = x - mu
        m2 = float(diff @ prec @ diff)
        maha_vals.append(m2)
    maha_vals = np.array(maha_vals)
    pass_maha = maha_vals <= chi_thr

    # final labels
    accepted = pass_cos & pass_maha
    labels_online = np.where(accepted, best_cids, -1)
    novel_mask = labels_online == -1

    # probability-calibrated novelty score in [0,1], higher => more novel
    # F_cos: per-cluster ECDF(d_cos); F_maha: chi2 CDF(maha)
    F_cos = np.zeros_like(d_cos, dtype=np.float32)
    for i, (dc, cid) in enumerate(zip(d_cos, best_cids)):
        entry = registry[int(cid)]
        dsorted = entry.get("dcos_sorted", None)
        if dsorted is None or len(dsorted) == 0:
            # Fallback when no ECDF available for a cluster (e.g., newly spawned)
            # Use normalized distance vs. its radius as a proxy in [0,1]
            r_local = float(entry.get("r95", 1.0))
            F_cos[i] = float(np.clip(dc / max(r_local, 1e-6), 0.0, 1.0))
        else:
            r = np.searchsorted(dsorted, dc, side='right')
            F_cos[i] = r / max(1, len(dsorted))
    F_maha = chi2.cdf(maha_vals, df=df).astype(np.float32)

    novelty_scores = np.maximum(F_cos, F_maha)  # unified, comparable across clusters
    return labels_online, novel_mask, novelty_scores


def fit_clustering(
    embeddings: np.ndarray,
    method: str = 'graph',
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    granularity: float = 0.05,
    k: int = 15,
    use_behavioral: bool = False,
    behavioral_alpha: float = 0.3,
    target_clusters: int = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Unified clustering interface supporting both HDBSCAN and graph-based methods.
    
    Args:
        embeddings: [N, D] trajectory embeddings
        method: 'hdbscan' or 'graph'
        trajectory_manager: For graph method with behavioral features
        indices: Trajectory indices for behavioral feature computation
        granularity: HDBSCAN granularity parameter
        k: Graph k-NN parameter
        use_behavioral: Use Jacobian-based edge reweighting (graph only)
        behavioral_alpha: Weight for behavioral features
        target_clusters: Hint for expected cluster count
        seed: Random seed
    
    Returns:
        labels, registry, core_ids, centers
    """
    if method == 'hdbscan':
        model, labels, core_ids, centers = fit_hdbscan_seen(
            embeddings, granularity=granularity, seed=seed
        )
        registry = build_registry(embeddings, labels, core_ids, centers)
        return labels, registry, core_ids, centers
    
    elif method == 'graph':
        from graph_clustering import fit_graph_clustering
        return fit_graph_clustering(
            embeddings=embeddings,
            trajectory_manager=trajectory_manager,
            indices=indices,
            k=k,
            use_behavioral_features=use_behavioral,
            behavioral_alpha=behavioral_alpha,
            resolution_sweep=True,
            target_clusters=target_clusters,
            min_cluster_size=max(5, int(0.02 * len(embeddings))),
            seed=seed,
        )
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
def evaluate_novel_detection(labels_pred, y_true, unseen_modes_list, n_seen, registry=None):
    """
    Evaluate how well novel clusters capture unseen modes.
    
    Args:
        labels_pred: Predicted cluster labels [N_all]
        y_true: True mode labels [N_all] (format: 10+mode_id)
        unseen_modes_list: List of mode IDs that are unseen (e.g., [2] for Walker)
        n_seen: Number of seen trajectories
        registry: Optional registry dict to use is_novel flag
    
    Returns:
        Dictionary with precision, recall, f1, and detailed stats
    """
    # True unseen mask (based on ground truth labels)
    unseen_true_labels = [10 + m for m in unseen_modes_list]
    is_truly_unseen = np.isin(y_true, unseen_true_labels)
    
    # Predicted novel mask
    if registry is not None:
        # Use registry's is_novel flag if available
        novel_pred_clusters = {cid for cid, info in registry.items() if info.get('is_novel', False)}
    else:
        # Fallback: clusters that are majority unseen
        unique_clusters = np.unique(labels_pred[labels_pred != -1])
        novel_pred_clusters = set()
        
        for cid in unique_clusters:
            cluster_mask = labels_pred == cid
            unseen_in_cluster = is_truly_unseen[cluster_mask].sum()
            total_in_cluster = cluster_mask.sum()
            if total_in_cluster > 0 and unseen_in_cluster / total_in_cluster > 0.5:
                novel_pred_clusters.add(cid)
    
    is_pred_novel = np.isin(labels_pred, list(novel_pred_clusters))
    
    # Compute precision/recall for novel detection
    tp = np.sum(is_truly_unseen & is_pred_novel)  # Correctly identified as novel
    fp = np.sum(~is_truly_unseen & is_pred_novel)  # Seen trajectories incorrectly marked as novel
    fn = np.sum(is_truly_unseen & ~is_pred_novel)  # Unseen trajectories missed (assigned to existing clusters)
    tn = np.sum(~is_truly_unseen & ~is_pred_novel)  # Seen trajectories correctly kept in existing clusters
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'n_truly_unseen': int(is_truly_unseen.sum()),
        'n_pred_novel': int(is_pred_novel.sum()),
        'novel_clusters_detected': sorted(novel_pred_clusters),
    }
