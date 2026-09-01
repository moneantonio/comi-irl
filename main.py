import os
import math
import argparse
import torch as th #type: ignore[import]
import torch.nn as nn # type: ignore[import]
import torch.nn.functional as F # type: ignore[import]
import numpy as np # type: ignore[import]
import random
import pickle
import pandas as pd # type: ignore[import]
from tqdm import tqdm # type: ignore[import]
from collections import Counter
from sklearn.cluster import HDBSCAN # type: ignore[import]
from sklearn.cluster import AgglomerativeClustering # type: ignore[import]
from sklearn.decomposition import PCA # type: ignore[import]
import networkx as nx # type: ignore[import]

#visualization imports
import matplotlib.pyplot as plt # type: ignore[import]
from scipy.spatial.distance import cdist # type: ignore[import]
from scipy.optimize import linear_sum_assignment # type: ignore[import]
import umap # type: ignore[import]

from torch.utils.data import ConcatDataset, DataLoader # type: ignore[import]

from sklearn.metrics import (silhouette_score, davies_bouldin_score, # type: ignore[import]
    calinski_harabasz_score, adjusted_rand_score, precision_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    recall_score,f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, pairwise_distances
)
from sklearn.metrics.pairwise import cosine_similarity # type: ignore[import]
import seaborn as sns # type: ignore[import]
from scipy.sparse.csgraph import connected_components # type: ignore[import]

from stable_baselines3.ppo import MlpPolicy # type: ignore[import]
from stable_baselines3 import PPO # type: ignore[import]


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from essinfogail.envs import *
from utils import *
from be import *
from loss import *
from training import *
from inference import *
from graph_clustering import *
from irl_parallel import *

plt.rcParams.update({
    "font.size": 32,             # base font size
    "axes.titlesize": 20,        # axes title
    "axes.labelsize": 16,        # x/y labels
    "legend.fontsize": 16,       # legend text
    "legend.title_fontsize": 14, # legend title (if used)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})


def setup_parser():
    parser = argparse.ArgumentParser(description='Model-Free Multi Intention Maximum Likelihood IRL: Experiments Runner')
    
    arg_env = parser.add_argument_group('Environment Selection')
    arg_env.add_argument("-Rv4","--Reacherv4", help="Apply the selected algorithm to the Reacher-v4 environment",action="store_true")
    arg_env.add_argument("-Pv4","--Pusherv4", help="Apply the selected algorithm to the Pusher-v4 environment",action="store_true")
    arg_env.add_argument("-W2D","--Walker2dv4", help="Apply the selected algorithm to the Walker2d-v4 environment",action="store_true")
    arg_env.add_argument("-Ho4","--Hopperv4", help="Apply the selected algorithm to the Hopper-v4 environment",action="store_true")
    arg_env.add_argument("-HC5","--HalfCheetahv5", help="Apply the selected algorithm to the HalfCheetah-v5 environment",action="store_true")

    arg_alg = parser.add_argument_group('Algorithm Selection')
    arg_alg.add_argument("-finetuning","--finetuning", action="store_true", 
                    help="Use seen-unseen split for training. If not set, train on the entire dataset.")
    arg_alg.add_argument("-nUnModes","--num_unseen_modes", type=int, default=1, help="Number of unseen modes in the unseen split for the dataset.")
    arg_alg.add_argument("-ablation","--ablation", action="store_true", help="Use ablation version of the model without RFF and TemporalConv for actions.")
    arg_alg.add_argument("-irl_training","--irl_training", action="store_true", help="Run the IRL training phase.")
    arg_alg.add_argument("-parallel_irl","--parallel_irl", action="store_true", help="Run the IRL training phase in parallel.")

    arg_irl = parser.add_argument_group('IRL Selection')
    arg_irl.add_argument("-gail","--GAIL", help="Run GAIL IRL",action="store_true")
    arg_irl.add_argument("-sqil","--SQIL", help="Run SQIL IRL",action="store_true")
    arg_irl.add_argument("-airl","--AIRL", help="Run AIRL IRL",action="store_true")

    arg_hyp = parser.add_argument_group('Hyperparameters')
    arg_hyp.add_argument('-nT','--num_trajs', type=int,default=100,help='int: Number of expert trajectories to generate')
    arg_hyp.add_argument('-seed','--seed', type=int,default=42,help='int: Random seed for reproducibility') 
    arg_hyp.add_argument('--ratio', type=int, default=1,help="Ratio for splitting trajectories between modes. 1 uniform 3 first gets most last gets least, etc.")
    arg_hyp.add_argument('-alpha','--alpha', type=float,default=0.5,help='float: Weight for the contrastive loss term')
    arg_hyp.add_argument('-beta','--beta', type=float,default=1.0,help='float: Weight for the infomax loss term')
    arg_hyp.add_argument('-gamma','--gamma', type=float,default=0.5,help='float: Weight for the segmentation loss term')
    arg_hyp.add_argument('-delta','--delta', type=float,default=1.0,help='float: Weight for the stability loss term')

    arg_vis = parser.add_argument_group('Visualization')
    arg_vis.add_argument('-vC','--visualize_clusters', help='Visualize the clusters',action="store_true")
    arg_vis.add_argument('-vG','--visualize_graphs', help='Visualize the cluster quality graphs',action="store_true")
    arg_vis.add_argument('-vO','--visualize_original', help='Visualize original expert trajectories colored by cluster assignments',action="store_true")
    arg_vis.add_argument('-3d','--threeD', help='Visualize UMAP in 3D',action="store_true",default=False)
    arg_vis.add_argument('-r','--render', help='Render the environment',action="store_true",default=False)
    arg_alg.add_argument("-save_abl_results","--save_abl_results", action="store_true", 
                help="Save ablation clustering metrics (NMI, ARI, Silhouette) to a CSV file for later aggregation.")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    K = 3 if args.Walker2dv4 else 3 if args.Hopperv4  else 3 if args.HalfCheetahv5 else 6
    env_name =  "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "Walker2d-v4" if args.Walker2dv4 else "Hopper-v4" if args.Hopperv4 else "HalfCheetah-v5" if args.HalfCheetahv5 else "Unknown"
    env_id =  "Reacher-v4" if args.Reacherv4 else "Pusher-v4" if args.Pusherv4 else "Walker2d-v4" if args.Walker2dv4 else "Hopper-v4" if args.Hopperv4 else "HalfCheetah-v5" if args.HalfCheetahv5 else "UnknownEnv"
    env_code = "Rv4" if args.Reacherv4 else "Pv4" if args.Pusherv4 else "W2D" if args.Walker2dv4 else "Hv4" if args.Hopperv4 else "HC5" if args.HalfCheetahv5 else "UNK"
    tr_name = "gail" if args.GAIL else "airl" if args.AIRL else "sqil"
    device = th.device("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
    print(f"*** CoMIIRL approach on {env_name} on {device} device ***")

    RL_TS = 250_000 if args.Reacherv4 or args.Pusherv4 else 750_000 if args.Hopperv4 else 500_000 if args.HalfCheetahv5 else 1_500_000
    baseline_stage_name = "baseline" if args.finetuning else "ablation" if args.ablation else "complete"
    SEED = args.seed
    th.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    HIGH_CONTRAST_PREDICTED_COLORS = [
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # bluish green
        '#F0E442',  # yellow
        '#0072B2',  # blue
        '#D55E00',  # vermilion
        '#CC79A7',  # reddish purple (optional)
    ]

    trajectory_manager = {}

    demos = []
    demos_withrew = []
    labels = []

    ####################################################
    ################## DATA LOADING ####################
    ####################################################

    # Load all modes
    for i in range(K):
        if env_code == "Rv4" or env_code == "Pv4" or env_code == "W2D":
            file_path = f"essinfogail/expert_imitation_trajectories/expert_imitation_trajectories_{env_id}_mode_{i}.pkl"
        elif env_code == "Hv4" or env_code == "HC5":
            file_path = f"expert_trajectories_new/{env_id}_task_{i}.pkl"
        file_path_withrew = file_path.replace(".pkl", "_withrew.pkl")
        
        with open(file_path, "rb") as f:
            d = pickle.load(f)
            print(f"Loaded {len(d)} expert trajectories for mode {i}")
        with open(file_path_withrew, "rb") as f:
            dwr = pickle.load(f)
        
        demos.append(d)
        demos_withrew.append(dwr)
        labels.append(np.array([10 + i] * len(d)))

    n_per_mode = args.num_trajs
    modes = list(range(K))

    if args.finetuning:
        unseen_n = args.num_unseen_modes
        if unseen_n < 0 or unseen_n >= K:
            raise ValueError(f"num_unseen_modes must be in [0, {K-1}] for {env_id}")
        
        if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5:
            # Walker2d-v4: Sequential selection (baseline gets first modes, unseen gets last modes)
            # For num_unseen_modes=1: seen=[0,1], unseen=[2]
            # For num_unseen_modes=2: seen=[0], unseen=[1,2]
            all_modes = list(range(K))
            seen_modes = all_modes[:K - unseen_n]
            unseen_modes_list = all_modes[K - unseen_n:]
        else:
            # Reacher-v4 and Pusher-v4: Alternating selection
            # For num_unseen_modes=3: seen=[0,2,4], unseen=[1,3,5]
            all_modes = list(range(K))
            seen_modes = [m for m in all_modes if m % 2 == 0][:K - unseen_n]  # Even indices: 0, 2, 4, ...
            unseen_modes_list = [m for m in all_modes if m % 2 == 1][:unseen_n]  # Odd indices: 1, 3, 5, ...
            
            # Ensure we have exactly the right number of modes
            if len(unseen_modes_list) < unseen_n:
                # If we don't have enough odd modes, fill from even modes
                remaining = [m for m in all_modes if m not in seen_modes and m not in unseen_modes_list]
                unseen_modes_list.extend(remaining[:unseen_n - len(unseen_modes_list)])
        
        print(f"Finetuning mode: seen={seen_modes}, unseen={unseen_modes_list}, n_per_mode={n_per_mode}")
    else:
        seen_modes = modes
        unseen_modes_list = []
        print(f"No finetuning: using all modes {seen_modes}, n_per_mode={n_per_mode}")

    # Prepare training data
    trajectories = []
    trajectories_withrew = []
    true_labels_list = []

    for m in seen_modes:
        trajectories.extend(demos[m][:n_per_mode])
        trajectories_withrew.extend(demos_withrew[m][:n_per_mode])
        true_labels_list.append(labels[m][:n_per_mode])

    true_labels = np.concatenate(true_labels_list) if true_labels_list else np.array([])

    # Prepare online/unseen data
    unseen_trajectories_for_online = []
    trajectories_withrew_for_online = []
    true_labels_online_list = []

    for m in unseen_modes_list:
        unseen_trajectories_for_online.extend(demos[m][:n_per_mode])
        trajectories_withrew_for_online.extend(demos_withrew[m][:n_per_mode])
        true_labels_online_list.append(labels[m][:n_per_mode])

    true_labels_online = np.concatenate(true_labels_online_list) if true_labels_online_list else np.array([])

    print(f"Splitting into {len(seen_modes)} seen modes and {len(unseen_modes_list)} unseen modes. Total training trajectories: {len(trajectories)}")

    print(f"\n--- Calculating Original Expert Reward Statistics for {env_id} ---")
    mean_expert_reward, std_expert_reward = calculate_original_expert_reward_stats(trajectories_withrew)
    print(f"Original Expert Reward (from stored .rews): Mean={mean_expert_reward:.4f} ± Std={std_expert_reward:.4f}")

    input_coord_dims = demos[0][0].obs[0].shape[0]
    env_num_step = 1 
    num_actions = demos[0][0].acts[0].shape[0]
    num_trajs = len(trajectories)
    print(f"Generated {num_trajs} expert trajectories for {env_id} with {K} modes.")
    print("--- Preparing State-Action Tensors")
    all_states,all_actions,all_masks,all_labels,max_len = prepare_sa_trajectories(env_id,trajectories,true_labels)
    if args.finetuning:
        all_states_online,all_actions_online,all_masks_online,all_labels_online,max_len_online = prepare_sa_trajectories(env_id,unseen_trajectories_for_online,true_labels_online)

    if args.visualize_original: #scalers or original spaces visualization with flattening
        print("Trajectory flattening . . .")
        traj_flat_seen = [np.array(traj).flatten() for traj in all_states]
        traj_flat_seen_actions = [np.array(traj).flatten() for traj in all_actions]
        traj_flat_seen = np.stack(traj_flat_seen)
        traj_flat_seen_actions = np.stack(traj_flat_seen_actions)

        if args.finetuning:
            traj_flat_online = [np.array(traj).flatten() for traj in all_states_online]
            traj_flat_online_actions = [np.array(traj).flatten() for traj in all_actions_online]
            traj_flat_online = np.stack(traj_flat_online)
            traj_flat_online_actions = np.stack(traj_flat_online_actions)

        print("Original space visualization (using interleaved state-actions)...")
        # Use the same interleaving logic as khgail.py for a fair comparison
        interleaved_features, meta = build_interleaved_matrix(trajectories, max_steps=None, pad_value=0.0)
        print(f"Interleaved feature shape: {interleaved_features.shape} (T_max={meta['max_steps']})")

        # Calculate mean intra and inter distances on the interleaved features
        concatenations_labels_distances = np.array(true_labels)
        dists = cdist(interleaved_features, interleaved_features, 'euclidean')
        intra_mask = concatenations_labels_distances[:, None] == concatenations_labels_distances[None, :]
        inter_mask = ~intra_mask
        intra = dists[intra_mask]
        inter = dists[inter_mask]
        print("mean intra (interleaved s-a)", intra.mean(), "mean inter (interleaved s-a)", inter.mean())

        # UMAP projection on the interleaved features
        reducer = umap.UMAP(random_state=SEED, n_neighbors=20, min_dist=0.5, n_components=2, metric='euclidean')
        umap_proj = reducer.fit_transform(interleaved_features)

        # Color mapping for true labels
        label_colors = {10: HIGH_CONTRAST_PREDICTED_COLORS[0], 11: HIGH_CONTRAST_PREDICTED_COLORS[1],\
                        12: HIGH_CONTRAST_PREDICTED_COLORS[2], 13: HIGH_CONTRAST_PREDICTED_COLORS[3],\
                        14: HIGH_CONTRAST_PREDICTED_COLORS[4], 15: HIGH_CONTRAST_PREDICTED_COLORS[5]}
        colors = [label_colors.get(lbl, "gray") for lbl in true_labels]

        plt.figure(figsize=(8, 6))
        for lbl in np.unique(true_labels):
            idx = true_labels == lbl
            plt.scatter(umap_proj[idx, 0], umap_proj[idx, 1], c=label_colors.get(lbl, "gray"), label=f"Mode {lbl-10}", alpha=0.7)
        plt.title(f"Original {env_id} Trajectory Space")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    ############################################################
    #################### DATA NORMALIZATION ####################
    ############################################################

    state_scaler = action_scaler = "quantile_normal"
    simple_scaler_fit = "both"  if not args.finetuning else "seen"  # options: 'seen', 'both'

    if is_coord_states(all_states[0]) and state_scaler != 'none':
        ss = build_scaler(state_scaler)
        print(f"[Simple] Fitting state scaler='{state_scaler}' on {simple_scaler_fit}.")
        ss, state_clip_bounds = fit_on_flat(all_states, ss)
        all_states = apply_per_traj(all_states, ss)
        if args.finetuning:
            all_states_online = apply_per_traj(all_states_online, ss, state_clip_bounds)
        print("[Simple] States scaled.")

    # Actions: only if continuous [T,A]
    if is_cont_actions(all_actions) and action_scaler != 'none':
        sa = build_scaler(action_scaler)
        print(f"[Simple] Fitting action scaler='{action_scaler}' on {simple_scaler_fit}.")
        sa, actions_clip_bounds = fit_on_flat(all_actions, sa)
        all_actions = apply_per_traj(all_actions, sa)
        if args.finetuning:
            all_actions_online = apply_per_traj(all_actions_online, sa, actions_clip_bounds)
        print("[Simple] Actions scaled.")
    
    print("--- Populating Trajectory Manager ---")
    for i in range(len(trajectories)):
        trajectory_manager[i]= {
            'id': i,
            'original_trajectory': trajectories[i],
            'original_trajectory_with_rew': trajectories_withrew[i],
            'prepared_states': all_states[i],
            'prepared_actions': all_actions[i],
            'prepared_masks': all_masks[i],
            'real_cluster_label': all_labels[i]
        }
    print("Len trajectory manager:", len(trajectory_manager))
    if args.finetuning:
        print("--- Populating Trajectory Manager with Unseen Trajectories for Online Testing---")

        for i in range(len(unseen_trajectories_for_online)):
            idx = i + len(trajectories)
            trajectory_manager[idx]= {
                'id': idx,
                'original_trajectory': unseen_trajectories_for_online[i],
                'original_trajectory_with_rew': trajectories_withrew_for_online[i],
                'prepared_states': all_states_online[i],
                'prepared_actions': all_actions_online[i],
                'prepared_masks': all_masks_online[i],
                'real_cluster_label': all_labels_online[i]
            }
        print("Len trajectory manager after online:", len(trajectory_manager))

    obs_shape = trajectories[0].obs[0].shape
    print(f"Generated {len(trajectories)} expert trajectories.")
    transformer_folder = "./models/"
    model_folder = transformer_folder+f"/{env_code}/ntrj_{num_trajs}"
    csv_folder = f"./csvs/{env_id}_CoMIIRL_results"
    csv_file_path = os.path.join(csv_folder, f"{env_code}_{'AIRL' if args.AIRL else 'GAIL' if args.GAIL else 'SQIL'}_seed_{SEED}_FULL.csv")
    csv_file_path = csv_file_path.replace("_FULL.csv", f"_BL{int(K-args.num_unseen_modes)}.csv") if args.finetuning else csv_file_path
    csv_file_path = csv_file_path.replace(".csv", f"_ABL.csv") if args.ablation else csv_file_path
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    results = []
    num_steps = env_num_step
    seq_max_len = max_len #it was if SA
    train_size = int(num_trajs*1.0)
    train_size_online = int(len(unseen_trajectories_for_online)*1.0)
    val_size = 0 #int(num_trajs*0.0)
    val_size_online = int(len(unseen_trajectories_for_online)*0.0)
    test_size = 0 #int(len(trajectories) - train_size - val_size)
    test_size_online = int(len(unseen_trajectories_for_online)- train_size_online - val_size_online)

    ###############################################################
    #################### MODEL HYPERPARAMETERS ####################
    ###############################################################

    tr_lr = 0.0001
    emb_dim = 32
    num_heads = 4
    nlayers = 2
    d_hid = 1024
    dropout = 0.1

    gaussian_m_state = 64 
    gaussian_m_action = 32 
    gaussian_sigma_state = 0.001 if args.Walker2dv4 or args.Pusherv4 else 0.01  
    gaussian_sigma_action = 0.001 if args.Walker2dv4 or args.Pusherv4 else 0.01
    input_channels = obs_shape[0]
    cnn_output_dim = emb_dim

    # Loss weights
    #contrastive
    alpha_training = args.alpha
    #infomax
    beta_training = args.beta
    #segmentation
    gamma_training = args.gamma
    #stability
    delta_training = args.delta #stability

    #training
    epochs_pre = 100
    epochs_ft = 100
    loader_batch = 64
    val_bptt = 8
    test_bptt = 1

    config_name = f"max_len_{seq_max_len}_seed_{SEED}"
    config_name = config_name + f"_FULL" if not args.finetuning else config_name #FULL if training on all data
    config_name = config_name + f"_BL{int(K-args.num_unseen_modes)}" if args.finetuning else config_name #BL if finetuning
    config_name = config_name + f"_ABL" if args.ablation else config_name
    config_name = config_name + f"_modes_{unseen_n}" if args.finetuning else config_name

    print("--- Creating Dataloaders for State-Action Trajectories ---")
    if not args.finetuning:
        full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader = datasets_preparation_sa(all_states,all_actions,all_masks,true_labels,train_size,val_size,test_size,loader_batch,val_bptt,test_bptt,SEED)
    elif args.finetuning:
        full_dataset, train_dataset, val_dataset, test_dataset, total_dataloader, train_dataloader, val_dataloader, test_dataloader = datasets_preparation_sa_seen_unseen(all_states,all_actions,all_masks,true_labels,train_size,val_size,test_size,loader_batch,val_bptt,test_bptt,SEED,is_old_data=True)
        full_dataset_online, train_dataset_online, val_dataset_online, test_dataset_online, total_dataloader_online, train_dataloader_online, val_dataloader_online, test_dataloader_online = datasets_preparation_sa_seen_unseen(all_states_online,all_actions_online,all_masks_online,true_labels_online,train_size_online,val_size_online,test_size_online,loader_batch,val_bptt,test_bptt,SEED,is_old_data=False)
        combined_dataset = ConcatDataset([full_dataset, full_dataset_online])
        finetuning_dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    ###############################################################
    #################### MODEL&LOSS DEFINITION ####################
    ###############################################################

    behaviorencoder = BehaviorEncoderCLSattnSATyped(
        input_channels=input_channels,
        cnn_output_dim=cnn_output_dim,
        max_len=max_len,
        steps=max_len,
        nhead=num_heads,
        d_hid=d_hid,
        emb_dim=emb_dim,
        num_actions=num_actions,
        nlayers=nlayers,
        dropout=dropout,
        # pe_type=pe_type,
        input_coord_dims=input_coord_dims,
        gaussian_m_state=gaussian_m_state,
        gaussian_m_action=gaussian_m_action,
        gaussian_sigma_state=gaussian_sigma_state,
        gaussian_sigma_action=gaussian_sigma_action,
        ablation=args.ablation,
    )

    behaviorencoder.to(device)
    transformer_total_params = sum(p.numel() for p in behaviorencoder.parameters() if p.requires_grad)
    # print("\n=== RFF Initialization Check ===")
    # for name, param in behaviorencoder.named_parameters():
    #     if 'rff' in name.lower() and 'W' in name:
    #         print(f"{name}:")
    #         print(f"  Shape: {param.shape}")
    #         print(f"  Mean: {param.mean().item():.4f} (should be ~0)")
    #         print(f"  Std: {param.std().item():.4f} (should be ~sigma={gaussian_sigma_state})")
    #         print(f"  Requires grad: {param.requires_grad}")
    # print("\n=== Parameter Breakdown ===")
    # for name, param in behaviorencoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name:50s} {param.numel():>10,} params")
    # print("=" * 65)
    print(behaviorencoder.model_type,"parameters ->",transformer_total_params/1e6,"M")

    ########################################################
    #################### MODEL TRAINING ####################
    ########################################################


    print("\n--- BE Training ---")
    loss_type = "DCS" #DIM + Contrastive + Segment
    if args.alpha > 0.0 and args.beta > 0.0 and args.gamma > 0.0:
        loss_type = "DCS"
    elif args.alpha > 0.0 and args.beta > 0.0 and args.gamma == 0.0:
        loss_type = "DC" #dim + contrastive
    elif args.alpha > 0.0 and args.beta == 0.0 and args.gamma > 0.0:
        loss_type = "CS" #contrastive + segmentation
    elif args.alpha > 0.0 and args.beta == 0.0 and args.gamma == 0.0:
        loss_type = "C" #only contrastive
    else:
        loss_type = "Other"
    model_filename = f"/{behaviorencoder.model_type}_{loss_type}_{env_code}_{config_name}.pt"
    print("Model location:", model_folder + model_filename)

    if os.path.exists(model_folder + model_filename):
        print(f"Loading existing model: {model_filename}")
        behaviorencoder = th.load(model_folder + model_filename,map_location=th.device(device))
        # cluster_centroids = th.load(model_folder + model_filename.replace(".pt", "_centroids.pt"),map_location=th.device(device))
        # raw_alpha = th.load(model_folder + model_filename.replace(".pt", "_alpha.pt"))
    else:
        print(f"Model not found. Starting ENC-SA training for {model_filename}...")
        training_func = encoder_training
        behaviorencoder = training_func(
            env_id=env_id,
            encoder=behaviorencoder,
            dataloader=train_dataloader,
            device=device,
            epochs=epochs_pre,
            lr=tr_lr,
            alpha = alpha_training, #contrastive
            beta= beta_training,    #infomax
            gamma = gamma_training, #segmentation
        )
        # Save the trained model
        th.save(behaviorencoder, model_folder + model_filename)

    #########################################################
    #################### MODEL INFERENCE ####################
    #########################################################
    
    print("\n--- Running Inference on Dataloaders (ENC-SA) ---")
    behaviorencoder.eval()

    print("Inference on training dataloader...")
    inference_func = inference
    trajectory_manager, concatenations_train, indices_train = inference_func(behaviorencoder, train_dataloader, trajectory_manager, device)

    # print("Inference on test dataloader...")
    # trajectory_manager, concatenations_test, indices_test = inference_func(behaviorencoder, test_dataloader, trajectory_manager, device)

    if args.finetuning:
        print("Inference on online dataloader of unseen behaviors...")
        online_indices_offset = len(trajectories)
        trajectory_manager, concatenations_online, indices_online = inference_func(behaviorencoder, total_dataloader_online, trajectory_manager, device, index_offset=online_indices_offset)

    print("\n--- Processing Embeddings for Visualization ---")
    #extract true labels
    tj_embeddings_train_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_train]
    # tj_embeddings_test_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_test]
    if args.finetuning:
        tj_embeddings_online_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_online]

    #extract predicted labels - placeholders (-1) for now, will be filled after HDBSCAN
    tj_embeddings_train_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_train]
    # tj_embeddings_test_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_test]
    if args.finetuning:
        tj_embeddings_online_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_online]

    #convert embeddings to numpy arrays
    tj_concatenations_train = concatenations_train.squeeze(1).cpu().numpy()
    # tj_concatenations_test = concatenations_test.squeeze(1).cpu().numpy()
    if args.finetuning:
        tj_concatenations_online = concatenations_online.squeeze(1).cpu().numpy()

    # tj_concatenations_evaluation = np.vstack([tj_concatenations_test])
    # tj_embeddings_evaluation_true_labels = np.hstack([tj_embeddings_test_true_labels])
    # tj_embeddings_evaluations_pred_labels = np.hstack([tj_embeddings_test_pred_labels])
    if args.finetuning:
        tj_concatenations_online_evaluation = np.vstack([tj_concatenations_online])
        # tj_embeddings_evaluation_true_labels_with_online = np.hstack([tj_embeddings_evaluation_true_labels, tj_embeddings_online_true_labels])
        # tj_embeddings_evaluations_pred_labels_with_online = np.hstack([tj_embeddings_evaluations_pred_labels, tj_embeddings_online_pred_labels])

    #aggregate embeddings and labels
    tj_concatenations_seen = np.vstack([tj_concatenations_train])#, tj_concatenations_evaluation])
    tj_embeddings_seen_true_labels = np.hstack([tj_embeddings_train_true_labels])#, tj_embeddings_evaluation_true_labels])
    tj_embeddings_seen_pred_labels = np.hstack([tj_embeddings_train_pred_labels])#, tj_embeddings_evaluations_pred_labels])
    tj_concatenations_seen_true_labels = tj_embeddings_seen_true_labels
    if args.finetuning:
        tj_concatenations_seen_true_labels_with_online = np.hstack([tj_embeddings_seen_true_labels, tj_embeddings_online_true_labels])
        tj_concatenations_only_online = tj_concatenations_online_evaluation
        tj_concatenations_seen_with_online = np.vstack([tj_concatenations_seen, tj_concatenations_only_online])   
    
    ###################################################################
    #################### V I S U A L I Z A T I O N ####################
    ###################################################################

    #COLOR MAPPINGS
    if args.Reacherv4 or args.Pusherv4:
        color_label_mapping = {
                "tab:blue": "Mode 0",
                "tab:red": "Mode 1",
                "tab:green": "Mode 2",
                "tab:purple": "Mode 3",
                "tab:brown": "Mode 4",
                "tab:orange": "Mode 5",
                "y": "???"
            }
    if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5:
        color_label_mapping = {
            "tab:blue": "Mode 0",
            "tab:red": "Mode 1",
            "tab:green": "Mode 2",
        }
    print("\n--- Generating Visualizations and Metrics ---")

    if args.finetuning:
        granularity = 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.03 if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5 else  0.02
        if unseen_n == 1:
            quantile_ms = 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1 if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5 else 0.02
            granularity = 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.2 if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5 else 0.02
            granularity_adjustement = 0.5 if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5 else 0.4 if args.Reacherv4 else 0.5 if args.Pusherv4 else 0.5
        elif unseen_n == 2:
            quantile_ms = 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.15
            granularity = 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.1
        elif unseen_n == 3:
            quantile_ms = 0.1 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.15
            granularity = 0.075 if args.Reacherv4 else 0.05 if args.Pusherv4  else 0.1
    else:
        granularity = 0.05 if args.Reacherv4 else 0.1 if args.Pusherv4 else 0.1 if args.Walker2dv4 else 0.02
        quantile_ms = 0.095

    reducer = umap.UMAP(
            random_state=SEED,
            n_neighbors=100,
            min_dist=0.5,
            n_components=3 if args.threeD else 2,
            metric='cosine',
        )

    if not args.finetuning:
        umap_combined = reducer.fit_transform(tj_concatenations_seen)
    else:
        umap_combined = reducer.fit_transform(tj_concatenations_seen_with_online)
    Z_train = umap_combined[:len(tj_concatenations_train)]


    
    print("--- PHASE 0: Visualize space with true labels")
    if args.visualize_clusters:
        # --- Plot 0: Train + Eval data with TRUE labels ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d' if args.threeD else None)
        
        # Combine train and eval for a single "seen" plot
        Z_seen_plot = np.vstack([Z_train])
        labels_seen_plot = np.hstack([tj_embeddings_train_true_labels])
        
        unique_true_labels = np.unique(labels_seen_plot)
        
        for label_val in unique_true_labels:
            mask = labels_seen_plot == label_val
            
            # Determine color and label name
            color_idx = int(label_val - 10) % len(HIGH_CONTRAST_PREDICTED_COLORS)
            color = HIGH_CONTRAST_PREDICTED_COLORS[color_idx]
            
            label_name = f"Mode {int(label_val - 10)}"
            if color in color_label_mapping:
                label_name = color_label_mapping[color]

            if args.threeD:
                ax.scatter(Z_seen_plot[mask, 0], Z_seen_plot[mask, 1], Z_seen_plot[mask, 2],
                            c=color, s=20, alpha=0.8, label=label_name)
            else:
                ax.scatter(Z_seen_plot[mask, 0], Z_seen_plot[mask, 1],
                            c=color, s=20, alpha=0.8, label=label_name)

        ax.set_title(f"{env_id} Embedding Space (True Labels)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if args.threeD:
            ax.set_zlabel("UMAP-3")
        
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()
        plt.close()


    ###################################################################
    #################### C L U S T E R I N G PHASE ####################
    ###################################################################

    print("\n--- PHASE 1: Evaluating Baseline Model only on Seen ---")

    use_graph_clustering = True  # Set to False to use HDBSCAN
    
    X_seen = tj_concatenations_seen
    indices_seen = np.hstack([indices_train])

    # === EMPIRICAL TEST: Check if Jacobian features are redundant ===
    redundancy_result = compute_jacobian_redundancy(
        embeddings=X_seen,
        trajectory_manager=trajectory_manager,
        indices=indices_seen,
        k=15,
        verbose=True,
    )

    jacobian_redundancy_correlation = redundancy_result['pearson_correlation']
    jacobian_spearman_correlation = redundancy_result['spearman_correlation']
    use_behavioral_recommendation = redundancy_result['use_behavioral_recommendation']
    print(f"Jacobian Redundancy - Pearson Correlation: {jacobian_redundancy_correlation:.4f},\
         Spearman Correlation: {jacobian_spearman_correlation:.4f}")
    print(f"Using Behavioral Features in Clustering: {use_behavioral_recommendation}")



    if use_graph_clustering:
        # Graph-based clustering (Leiden)
        graph_k = 15
        behavioral_alpha = 0.3 if args.Walker2dv4 or args.Hopperv4 or args.HalfCheetahv5 else 0.3
        labels_seen, registry, core_ids, centers_seen, adaptive_info = fit_graph_clustering_joint(
            embeddings=X_seen,
            trajectory_manager=trajectory_manager,
            indices=indices_seen,
            k_range=(1, min(100, len(X_seen) // 4)),
            use_behavioral_features=use_behavioral_recommendation,
            behavioral_alpha=behavioral_alpha,
            min_cluster_size=max(5, int(0.02 * len(X_seen))), #0.1 before
            seed=SEED,
            verbose=True,
        )
        print(f"[Joint] Selected k={adaptive_info['k_selected']}")
        
        reg_ids = sorted(registry.keys())
        n_clusters_ = len(reg_ids)
        cluster_centers = centers_seen
        
        print(f"[Phase 1 - Leiden] Found {n_clusters_} clusters")

        if args.visualize_graphs and use_behavioral_recommendation and args.visualize_clusters:
            print("\n--- Visualizing Jacobian Reweighting Effect ---")
            fig_reweight, reweight_stats = visualize_jacobian_reweighting_effect(
                embeddings=X_seen,
                trajectory_manager=trajectory_manager,
                indices=indices_seen,
                k=adaptive_info['k_selected'],
                behavioral_alpha=behavioral_alpha,
                edge_sample_ratio=0.1,
                seed=SEED,
                save_path=f"./figures/{env_id}_jacobian_reweighting.pdf" if not args.visualize_clusters else None,
            )
            print(f"  Mean weight change: {reweight_stats['mean_change_pct']:.1f}%")
            print(f"  Edges strengthened: {reweight_stats['n_increased']}, weakened: {reweight_stats['n_decreased']}")

        # Visualization
        if args.visualize_clusters:
            graph_viz = TrajectoryGraph(
                embeddings=X_seen / (np.linalg.norm(X_seen, axis=1, keepdims=True) + 1e-8),
                k=adaptive_info['k_selected'],
                metric='cosine',
            )
            
            visualize_graph_clustering(
                embeddings=X_seen,
                labels=labels_seen,
                graph=graph_viz,
                reducer=reducer,
                title=f"{env_id} Graph Clustering (Leiden)",
                show_edges=True,
                edge_sample_ratio=0.05,
                seed=SEED,
            )
    else:
        # Original HDBSCAN clustering
        model_seen, labels_seen, core_ids, centers_seen = fit_hdbscan_seen(X_seen, granularity=granularity, seed=SEED)
        registry = build_registry(X_seen, labels_seen, core_ids, centers_seen)
        
        reg_ids = sorted(registry.keys())
        n_clusters_ = len(reg_ids)
        cluster_centers = np.stack([registry[cid]["center"] for cid in reg_ids], axis=0) if len(reg_ids) > 0 else np.zeros((0, X_seen.shape[1]))
        cluster_centers = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
        
        print(f"[Phase 1 - HDBSCAN] Found {n_clusters_} clusters")

    # Save baseline labels for stability report (used in finetuning phase)
    labels_seen_baseline = labels_seen.copy()

    # Initialize empty arrays for online data (will be populated in finetuning)
    labels_online = np.zeros((0,), dtype=int)
    novel_mask = np.zeros((0,), dtype=bool)
    novelty_scores = np.zeros((0,), dtype=float)

    # Final labels for Phase 1
    cluster_labels = labels_seen  # No online data yet in Phase 1
    
    # --- EVALUATE CLUSTERING QUALITY ---
    y_true_all = tj_embeddings_seen_true_labels
    
    mask_core_final = cluster_labels != -1
    if mask_core_final.sum() > 1 and len(np.unique(y_true_all[mask_core_final])) > 1 and len(np.unique(cluster_labels[mask_core_final])) > 1:
        nmi_final = normalized_mutual_info_score(y_true_all[mask_core_final], cluster_labels[mask_core_final])
        ari_final = adjusted_rand_score(y_true_all[mask_core_final], cluster_labels[mask_core_final])
        if len(np.unique(cluster_labels[mask_core_final])) >= 2:
            silhouette_final = silhouette_score(
                X_seen[mask_core_final],
                cluster_labels[mask_core_final],
                metric='cosine'
            )
        else:
            silhouette_final = np.nan
    else:
        nmi_final, ari_final = np.nan, np.nan
        silhouette_final = np.nan
    
    print(f"--- PHASE 1: Clustering Quality Metrics - NMI: {nmi_final:.4f} - ARI: {ari_final:.4f} - Silhouette: {silhouette_final:.4f} ---")

    if args.save_abl_results:
        save_ablation_results(
            env_id=env_id,
            loss_type=loss_type,
            seed=SEED,
            nmi=nmi_final,
            ari=ari_final,
            silhouette=silhouette_final,
            stage=baseline_stage_name
        )

    if args.visualize_clusters and not use_graph_clustering:
        Z_seen = reducer.transform(tj_concatenations_seen)
        Z_online_viz = np.zeros((0, Z_seen.shape[1]))
        visualize_controller_output(
            Z_seen=Z_seen,
            Z_online=Z_online_viz,
            labels_seen=labels_seen,
            labels_online=labels_online,
            novelty_scores=novelty_scores,
            registry=registry,
            reducer=reducer,
            title=f"HDBSCAN on {env_id} embeddings",
            is_3d=args.threeD,
            palette=HIGH_CONTRAST_PREDICTED_COLORS,
        )

    ########################################################################
    #################### Inverse Reinforcement Learning ####################
    ########################################################################

    if args.irl_training:
        print("=== Phase 1.5: First IRL Cycle ===")
        if args.parallel_irl:
            indices_seen = np.hstack([indices_train])
            
            # Prepare all IRL tasks
            task_configs = prepare_irl_tasks(
                cluster_ids=reg_ids,
                labels=labels_seen,
                indices=indices_seen,
                trajectory_manager=trajectory_manager,
                env_name=env_name,
                env_id=env_id,
                tr_name=tr_name,
                seed=SEED,
                stage=baseline_stage_name,
                rl_timesteps=RL_TS,
            )
            
            # Run IRL training (parallel on CPU, sequential on MPS)
            all_cluster_results = run_irl_training(
                task_configs=task_configs,
                device=device,
                max_workers=None,  # Auto-detect based on CPU count
                force_sequential=False,  # Set to True to disable parallelization
                verbose=False,
            )
            
            # Process results and save to CSV
            for cid, irl_results in all_cluster_results.items():
                cluster_mask = (labels_seen == cid)
                num_trajs = cluster_mask.sum()
                
                new_record = {
                    "seed": SEED,
                    "ratio": args.ratio,
                    "env_name": env_name,
                    "tr_name": tr_name,
                    "cluster_id": cid,
                    "num_trajs_in_cluster": num_trajs,
                    "state_scaler": state_scaler,
                    "granularity": granularity,
                    "NMI_final": nmi_final,
                    "ARI_final": ari_final,
                    "Silhouette_final": silhouette_final,
                    "stage": baseline_stage_name,
                    "jacobian_pearson_corr": jacobian_redundancy_correlation,
                    "jacobian_spearman_corr": jacobian_spearman_correlation,
                }
                new_record.update(irl_results)
                results.append(new_record)
            
            # Save results
            if results:
                df_partial = pd.DataFrame(results)
                df_partial.to_csv(csv_file_path, index=False)
                print(f"Results saved to {csv_file_path}")

            # Print summary
            print("\n--- Final Reward Comparison Summary ---")
            for cid, res in sorted(all_cluster_results.items()):
                print(f"Cluster {cid} (Mode {res.get('Expert Mode', '?')}): "
                    f"Expert Reward = {res.get('Expert Reward Mean', 0):.2f} | "
                    f"Learner Reward = {res.get('Learner Reward Mean', 0):.2f}")
        else:
            # Create a mapping from the index in tj_concatenations_seen back to the original trajectory_manager index
            indices_seen = np.hstack([indices_train])
            baseline_stage_name = "baseline" if args.finetuning else "ablation" if args.ablation else "complete"
            all_cluster_results = {}

            for cid in reg_ids:
                # 1. Extract trajectories for the current cluster
                cluster_mask = (labels_seen == cid)
                manager_indices = indices_seen[cluster_mask]
                
                cluster_trajectories = [trajectory_manager[i]['original_trajectory'] for i in manager_indices]
                cluster_trajectories_with_rew = [trajectory_manager[i]['original_trajectory_with_rew'] for i in manager_indices]
                cluster_true_labels = [trajectory_manager[i]['real_cluster_label'] for i in manager_indices]
                
                if not cluster_trajectories:
                    print(f"Cluster {cid} has no trajectories. Skipping.")
                    continue

                # 2. Call the generalized training function
                irl_results = train_and_evaluate_irl_agent(
                    cluster_id=cid,
                    trajectories=cluster_trajectories,
                    trajectories_with_rew=cluster_trajectories_with_rew,
                    true_labels=cluster_true_labels,
                    env_name=env_name,
                    env_id=env_id,
                    tr_name=tr_name,
                    seed=SEED,
                    base_learner_policy=MlpPolicy,
                    stage=baseline_stage_name,
                    rl_timesteps=RL_TS,
                )

                if irl_results:
                    all_cluster_results[cid] = irl_results
                    
                    # 3. Create and save a record for this cluster's results
                    # This replaces the repetitive record creation from the old code.
                    new_record = {
                        "seed": SEED,
                        "ratio": args.ratio,
                        "env_name": env_name,
                        "tr_name": tr_name,
                        "cluster_id": cid,
                        "num_trajs_in_cluster": len(cluster_trajectories),
                        # Add other relevant hyperparameters from `args` or config
                        "state_scaler": state_scaler,
                        "granularity": granularity,
                        # Add clustering metrics if you have them (calculate them once before this loop)
                        "NMI_final": nmi_final,
                        "ARI_final": ari_final,
                        "Silhouette_final": silhouette_final,
                        "stage": baseline_stage_name,
                    }
                    new_record.update(irl_results)
                    
                    results.append(new_record)
                    
                    # Save partial results to CSV after each cluster
                    df_partial = pd.DataFrame(results)
                    df_partial.to_csv(csv_file_path, index=False)
                    print(f"  Partial results for cluster {cid} saved to {csv_file_path}")

            # 4. Final summary and save
            print("\n--- Final Reward Comparison Summary ---")
            for cid, res in sorted(all_cluster_results.items()):
                print(f"Cluster {cid} (Mode {res['Expert Mode']}): "
                        f"Expert Reward = {res['Expert Reward Mean']:.2f} | "
                        f"Learner Reward = {res['Learner Reward Mean']:.2f} | "
                        # f"Normalized = {res['Normalized Learner Reward']:.2%}"
                    )

        final_df = pd.DataFrame(results)
        final_df.to_csv(csv_file_path, index=False)
        print(f"\nAll experiments done. Final results saved to {csv_file_path}")

    #####################################################################################
    #################### T R A N S F O R M E R   F I N E T U N I N G ####################
    #####################################################################################

    if args.finetuning:
        print("\n--- PHASE 2: Finetuning model on Seen + Online Data ---")
        if os.path.exists(model_folder + model_filename.replace(".pt", f"_FT{args.num_unseen_modes}.pt")):
            print(f"Loading existing model: {model_filename}")
            finetuned_encoder = th.load(model_folder + model_filename,map_location=th.device(device))
            finetuned_encoder.eval()
            # cluster_centroids = th.load(model_folder + model_filename.replace(".pt", "_centroids.pt"),map_location=th.device(device))
            # raw_alpha = th.load(model_folder + model_filename.replace(".pt", "_alpha.pt"))
        else:
            print(f"Model not found. Starting ENC-SA training for {model_filename}...")
            finetuned_encoder = encoder_finetuning(
                env_id=env_id,
                encoder=behaviorencoder,
                dataloader=finetuning_dataloader,
                K=K,
                device=device,
                lr=tr_lr,
                epochs=epochs_ft,
                alpha = alpha_training,
                beta = beta_training,
                gamma = gamma_training,
                delta = delta_training
            )
            finetuned_encoder.eval()
            # Save the finetuned model and centroids with _FT suffix
            finetuned_model_filename = model_filename.replace(".pt", f"_FT{args.num_unseen_modes}.pt")
            print(f"Saving finetuned model to: {finetuned_model_filename}")
            th.save(finetuned_encoder, model_folder + finetuned_model_filename)

        
        print("\n--- Running Inference on Dataloaders with Finetuned Encoder ---")
        inference_func = inference
        trajectory_manager, concatenations_train, indices_train = inference_func(finetuned_encoder,train_dataloader, trajectory_manager, device)

        # trajectory_manager, concatenations_test, indices_test = inference_func(finetuned_encoder, test_dataloader, trajectory_manager, device)

        online_indices_offset = len(trajectories)
        trajectory_manager, concatenations_online, indices_online =inference_func(finetuned_encoder, total_dataloader_online, trajectory_manager, device, index_offset=online_indices_offset)

        print("\n--- Processing Embeddings after Finetuning for Visualization ---")
        tj_embeddings_train_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_train]
        # tj_embeddings_test_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_test]
        tj_embeddings_online_true_labels = [trajectory_manager[i]['real_cluster_label'].item() for i in indices_online]

        tj_embeddings_train_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_train]
        # tj_embeddings_test_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_test]
        tj_embeddings_online_pred_labels = [trajectory_manager[i]['predicted_cluster_label'] for i in indices_online]

        tj_concatenations_train = concatenations_train.squeeze(1).cpu().numpy()
        # tj_concatenations_test = concatenations_test.squeeze(1).cpu().numpy()
        tj_concatenations_online = concatenations_online.squeeze(1).cpu().numpy()

        # tj_concatenations_evaluation = np.vstack([tj_concatenations_test])
        # tj_embeddings_evaluation_true_labels = np.hstack([tj_embeddings_test_true_labels])
        # tj_embeddings_evaluations_pred_labels = np.hstack([tj_embeddings_test_pred_labels])
        tj_concatenations_online_evaluation = np.vstack([tj_concatenations_online])
        # tj_embeddings_evaluation_true_labels_with_online = np.hstack([tj_embeddings_evaluation_true_labels, tj_embeddings_online_true_labels])
        # tj_embeddings_evaluations_pred_labels_with_online = np.hstack([tj_embeddings_evaluations_pred_labels, tj_embeddings_online_pred_labels])

        tj_concatenations_seen = np.vstack([tj_concatenations_train])#, tj_concatenations_evaluation])
        tj_embeddings_seen_true_labels = np.hstack([tj_embeddings_train_true_labels])#, tj_embeddings_evaluation_true_labels])
        tj_embeddings_seen_pred_labels = np.hstack([tj_embeddings_train_pred_labels])#, tj_embeddings_evaluations_pred_labels])
        tj_concatenations_seen_true_labels = tj_embeddings_seen_true_labels
        tj_concatenations_seen_true_labels_with_online = np.hstack([tj_embeddings_seen_true_labels, tj_embeddings_online_true_labels])
        tj_concatenations_only_online = tj_concatenations_online_evaluation
        tj_concatenations_seen_with_online = np.vstack([tj_concatenations_seen, tj_concatenations_only_online])   

        # === Post-finetune: cluster + visualize all trajectories ===
        print("\n=== PHASE 3: Clustering on finetuned embeddings (seen + online) ===")
        
        X_all_ft = tj_concatenations_seen_with_online
        y_true_all_ft = tj_concatenations_seen_true_labels_with_online
        indices_all_ft = np.hstack([indices_train, indices_online])
        n_seen_ft = len(tj_concatenations_seen)
        
        use_anchored = True #args.Walker2dv4  # Use two-stage only for Walker2d-v4



        if use_graph_clustering:

            if use_anchored:
                print("[Phase 3] Using TWO-STAGE clustering for continuous manifold...")
                
                # Number of baseline clusters from Phase 1
                n_baseline_clusters = len(reg_ids)
                
                labels_all_ft, registry_ft, core_ids_ft, centers_ft, adaptive_info_ft = fit_graph_clustering_two_stage(
                    embeddings_finetuned=X_all_ft,
                    n_seen=n_seen_ft,
                    n_baseline_clusters=n_baseline_clusters,
                    trajectory_manager=trajectory_manager,
                    indices=indices_all_ft,
                    k_range=(1, min(100, n_seen_ft // 4)), 
                    novelty_threshold=0.05 if env_id == "Walker2d-v4" else 0.1,
                    use_behavioral_features=use_behavioral_recommendation,
                    behavioral_alpha=behavioral_alpha,
                    min_cluster_size=max(5, int(0.02 * len(X_all_ft))),
                    seed=SEED,
                    verbose=True,
                )
                
                # Novel clusters are identified by the two-stage method
                novel_cluster_ids = adaptive_info_ft.get('novel_cids', set())

            else:
                # Graph-based clustering (Leiden) for finetuned embeddings
                print("[Phase 3] Using Leiden clustering on finetuned embeddings...")
                labels_all_ft, registry_ft, core_ids_ft, centers_ft, adaptive_info_ft = fit_graph_clustering_joint(
                    embeddings=X_all_ft,
                    trajectory_manager=trajectory_manager,
                    indices=indices_all_ft,
                    k_range=(1, min(100, len(X_all_ft) // 4)),
                    use_behavioral_features=use_behavioral_recommendation,
                    behavioral_alpha=behavioral_alpha,
                    min_cluster_size=max(5, int(0.02 * len(X_all_ft))),
                    seed=SEED,
                    verbose=True,
                )
                print(f"[Phase 3 - Joint] Selected k={adaptive_info_ft['k_selected']}")
                baseline_cluster_count = len(reg_ids)
                novel_cluster_ids = {cid for cid in core_ids_ft if cid >= baseline_cluster_count}
            
            reg_ids_ft = sorted(registry_ft.keys())
            n_clusters_ft = len(reg_ids_ft)
            cluster_centers_ft = centers_ft
            
            print(f"[Phase 3 - {'Two-Stage Leiden' if use_anchored else 'Leiden'}] Found {n_clusters_ft} clusters")

            # print("\n[DEBUG] Cluster-to-Mode Correspondence (Post-TSL):")
            # for cid in sorted(reg_ids_ft):
            #     cluster_mask = labels_all_ft == cid
            #     cluster_indices = indices_all_ft[cluster_mask]
                
            #     # Get true mode labels for trajectories in this cluster
            #     true_modes_in_cluster = []
            #     for idx in cluster_indices:
            #         true_label = trajectory_manager[idx]['real_cluster_label']
            #         true_mode = true_label.item() - 10 if hasattr(true_label, 'item') else true_label - 10
            #         true_modes_in_cluster.append(true_mode)
                
            #     mode_counts = Counter(true_modes_in_cluster)
            #     dominant_mode = mode_counts.most_common(1)[0][0] if mode_counts else -1
            #     purity = mode_counts.most_common(1)[0][1] / len(true_modes_in_cluster) if true_modes_in_cluster else 0
                
            #     is_novel = cid in novel_cluster_ids
            #     print(f"  Cluster {cid} {'[NOVEL]' if is_novel else '[BASELINE]'}:")
            #     print(f"    Mode distribution: {dict(mode_counts)}")
            #     print(f"    Dominant mode: {dominant_mode} (purity: {purity:.1%})")

            # Compute metrics
            mask_core_ft = labels_all_ft != -1
            if mask_core_ft.sum() > 1:
                nmi_ft = normalized_mutual_info_score(y_true_all_ft[mask_core_ft], labels_all_ft[mask_core_ft])
                ari_ft = adjusted_rand_score(y_true_all_ft[mask_core_ft], labels_all_ft[mask_core_ft])
                silhouette_ft = silhouette_score(X_all_ft[mask_core_ft], labels_all_ft[mask_core_ft], metric='cosine') \
                    if len(np.unique(labels_all_ft[mask_core_ft])) >= 2 else np.nan
            else:
                nmi_ft, ari_ft, silhouette_ft = np.nan, np.nan, np.nan
            
            print(f"[Phase 3] NMI={nmi_ft:.4f} | ARI={ari_ft:.4f} | Silhouette={silhouette_ft:.4f}")
            # Novel detection metrics
            novel_eval = evaluate_novel_detection(
                labels_pred=labels_all_ft,
                y_true=y_true_all_ft,
                unseen_modes_list=unseen_modes_list,
                n_seen=n_seen_ft,
                registry=registry_ft,
            )
            
            print(f"\n[Phase 3] Novel Detection: Precision={novel_eval['precision']:.4f} | "
                  f"Recall={novel_eval['recall']:.4f} | F1={novel_eval['f1']:.4f}")

            # Visualization
            if args.visualize_clusters:
                graph_viz_ft = TrajectoryGraph(
                    embeddings=X_all_ft / (np.linalg.norm(X_all_ft, axis=1, keepdims=True) + 1e-8),
                    k=adaptive_info_ft['k_selected'],
                    metric='cosine',
                )
                
                visualize_graph_clustering(
                    embeddings=X_all_ft,
                    labels=labels_all_ft,
                    graph=graph_viz_ft,
                    reducer=None,  # Will create new UMAP
                    title=f"{env_id} Graph Clustering (Leiden) - Post-Finetune",
                    show_edges=True,
                    edge_sample_ratio=0.05,
                    seed=SEED,
                )

        else:
            # Original HDBSCAN clustering for finetuned embeddings
            nX_ft = len(X_all_ft)
            min_cluster_size_ft = max(5, int(granularity * granularity_adjustement * nX_ft))
            min_samples_ft = max(1, int(math.sqrt(min_cluster_size_ft)))
            hdb_ft = HDBSCAN(
                min_cluster_size=min_cluster_size_ft,
                min_samples=min_samples_ft,
                metric='cosine',
                cluster_selection_epsilon=0.0001
            ).fit(X_all_ft)
            labels_all_ft = hdb_ft.labels_

            # Metrics
            mask_core_ft = labels_all_ft != -1
            if mask_core_ft.sum() > 1 and len(np.unique(y_true_all_ft[mask_core_ft])) > 1 and len(np.unique(labels_all_ft[mask_core_ft])) > 1:
                ari_ft = adjusted_rand_score(y_true_all_ft[mask_core_ft], labels_all_ft[mask_core_ft])
                nmi_ft = normalized_mutual_info_score(y_true_all_ft[mask_core_ft], labels_all_ft[mask_core_ft])
                if len(np.unique(labels_all_ft[mask_core_ft])) >= 2:
                    silhouette_ft = silhouette_score(X_all_ft[mask_core_ft], labels_all_ft[mask_core_ft], metric='cosine')
                else:
                    silhouette_ft = np.nan
            else:
                ari_ft, nmi_ft, silhouette_ft = np.nan, np.nan, np.nan
            
            uniq_core_ft = np.unique(labels_all_ft[labels_all_ft != -1])
            print(f"[Phase 3 - HDBSCAN] Found {len(uniq_core_ft)} clusters | NMI={nmi_ft:.4f} | ARI={ari_ft:.4f}")

        print(f"PHASE 3: Clustering Quality Metrics - NMI: {nmi_ft:.4f} - ARI: {ari_ft:.4f} - Silhouette: {silhouette_ft:.4f}")

        uniq_core_ft = np.unique(labels_all_ft[labels_all_ft != -1])
        print(f"=== [Finetuned] Core clusters: {len(uniq_core_ft)} | NMI(core)={nmi_ft:.4f} | ARI(core)={ari_ft:.4f}")

        if args.visualize_clusters:
            # UMAP visualization of all points (seen + online)
            reducer_ft = umap.UMAP(
                random_state=SEED,
                n_neighbors=100,
                min_dist=0.5,
                n_components=3 if args.threeD else 2,
                metric='cosine',
            )
            Z_all_ft = reducer_ft.fit_transform(X_all_ft)

            # Split seen vs online in the plot
            seen_len = len(tj_concatenations_seen)
            if args.threeD:
                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(8, 6))

            # Seen (circles)
            lbl_seen = labels_all_ft[:seen_len]
            for lab in np.unique(lbl_seen):
                m = lbl_seen == lab
                color = 'k' if lab == -1 else plt.cm.tab10(int(lab) % 10)
                label_txt = "Seen noise" if lab == -1 else f"Seen C{int(lab)}"
                if args.threeD:
                    ax.scatter(Z_all_ft[:seen_len][m, 0], Z_all_ft[:seen_len][m, 1], Z_all_ft[:seen_len][m, 2],
                                c=[color], s=18, alpha=0.7, marker='o', label=label_txt)
                else:
                    ax.scatter(Z_all_ft[:seen_len][m, 0], Z_all_ft[:seen_len][m, 1],
                                c=[color], s=18, alpha=0.7, marker='o', label=label_txt)

            # Online (triangles), if any
            if len(X_all_ft) > seen_len:
                lbl_online = labels_all_ft[seen_len:]
                for lab in np.unique(lbl_online):
                    m = lbl_online == lab
                    color = 'k' if lab == -1 else plt.cm.tab10(int(lab) % 10)
                    label_txt = "Online noise" if lab == -1 else f"Online C{int(lab)}"
                    if args.threeD:
                        ax.scatter(Z_all_ft[seen_len:][m, 0], Z_all_ft[seen_len:][m, 1], Z_all_ft[seen_len:][m, 2],
                                    c=[color], s=36, alpha=0.95, marker='^', label=label_txt)
                    else:
                        ax.scatter(Z_all_ft[seen_len:][m, 0], Z_all_ft[seen_len:][m, 1],
                                    c=[color], s=36, alpha=0.95, marker='^', label=label_txt)

            ax.set_title(f"UMAP of all embeddings (post-finetune) — ARI={ari_ft:.3f}, NMI={nmi_ft:.3f}")
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
            if args.threeD:
                ax.set_zlabel("UMAP-3")
            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='best')
            plt.tight_layout()
            plt.show()

            # --- EXTRA: Visualize the SAME post-finetune embedding colored by TRUE labels ---
            try:
                # y_true_all_ft was set earlier to the true labels for seen+online
                labels_true_ft = np.asarray(y_true_all_ft)
                Z_true_plot = Z_all_ft
                fig3 = plt.figure(figsize=(8, 6))
                ax3 = fig3.add_subplot(111, projection='3d' if args.threeD else None)
                unique_true = np.unique(labels_true_ft)
                for lab in unique_true:
                    mask = labels_true_ft == lab
                    color_idx = int(lab - 10) % len(HIGH_CONTRAST_PREDICTED_COLORS)
                    color = HIGH_CONTRAST_PREDICTED_COLORS[color_idx]
                    label_txt = f"Mode {int(lab - 10)}"
                    if args.threeD:
                        ax3.scatter(Z_true_plot[mask,0], Z_true_plot[mask,1], Z_true_plot[mask,2], c=color, s=20, alpha=0.85, label=label_txt)
                    else:
                        ax3.scatter(Z_true_plot[mask,0], Z_true_plot[mask,1], c=color, s=20, alpha=0.85, label=label_txt)
                ax3.set_title(f"{'3D' if args.threeD else '2D'} UMAP (Post-Finetune) - True labels")
                ax3.set_xlabel("UMAP-1"); ax3.set_ylabel("UMAP-2")
                if args.threeD: ax3.set_zlabel("UMAP-3")
                ax3.legend(loc='best', fontsize=8)
                plt.tight_layout()
                plt.show()
                plt.close(fig3)
            except Exception as e:
                print(f"[Viz after finetune] failed: {e}")

        print("\n=== Phase 4: Identifying Novel Clusters via Graph-Based Alignment ===")
        
        # Step 5: Analyze novel clusters (check if they're the unseen modes)
        print("\n--- Step 4.1: Novel Cluster Analysis ---")
        indices_all_ft = np.hstack([indices_train, indices_online])
        
        # Novel cluster analysis (for logging)
        if novel_cluster_ids:
            print("\n--- Novel Cluster Composition ---")
            for novel_cid in sorted(novel_cluster_ids):
                cluster_mask = (labels_all_ft == novel_cid)
                cluster_indices = indices_all_ft[cluster_mask]
                
                # Get true mode labels
                true_labels_in_cluster = [trajectory_manager[i]['real_cluster_label'] for i in cluster_indices]
                true_labels_in_cluster = [lab.item() if hasattr(lab, 'item') else lab for lab in true_labels_in_cluster]
                mode_counts = Counter([m - 10 for m in true_labels_in_cluster])
                
                dominant_mode = mode_counts.most_common(1)[0][0] if mode_counts else None
                purity = mode_counts.most_common(1)[0][1] / sum(mode_counts.values()) if mode_counts else 0
                
                is_expected_novel = dominant_mode in unseen_modes_list
                
                print(f"  Novel Cluster {novel_cid}:")
                print(f"    Size: {len(cluster_indices)} trajectories")
                print(f"    True mode distribution: {dict(mode_counts)}")
                print(f"    Dominant mode: {dominant_mode} (purity: {purity:.2%})")
                print(f"    Is expected novel mode: {'✓ YES' if is_expected_novel else '✗ NO'}")
        else:
            print("  No novel clusters detected.")

        if args.irl_training:

            print("\n--- Step 4.3: Training agents for novel clusters ---")

            if args.parallel_irl:
                novel_task_configs = prepare_irl_tasks(
                    cluster_ids=sorted(novel_cluster_ids),
                    labels=labels_all_ft,
                    indices=indices_all_ft,
                    trajectory_manager=trajectory_manager,
                    env_name=env_name,
                    env_id=env_id,
                    tr_name=tr_name,
                    seed=SEED,
                    stage="finetuned_novel",
                    rl_timesteps=RL_TS,
                )
                
                # Run parallel training for novel clusters
                novel_results = run_irl_training(
                    task_configs=novel_task_configs,
                    device=device,
                    max_workers=None,
                    force_sequential=False,
                    verbose=True,
                )
                
                # Process and save novel cluster results
                for novel_cid, irl_results_novel in novel_results.items():
                    cluster_mask_ft = (labels_all_ft == novel_cid)
                    num_trajs = cluster_mask_ft.sum()
                    
                    new_record = {
                        "seed": SEED,
                        "ratio": args.ratio,
                        "env_name": env_name,
                        "tr_name": tr_name,
                        "cluster_id": f"novel_{novel_cid}",
                        "num_trajs_in_cluster": num_trajs,
                        "state_scaler": state_scaler,
                        "granularity": granularity,
                        "NMI_final": nmi_ft,
                        "ARI_final": ari_ft,
                        "Silhouette_final": silhouette_ft,
                        "stage": "finetuned_novel",
                        "novel_precision": novel_eval['precision'],
                        "novel_recall": novel_eval['recall'],
                        "novel_f1": novel_eval['f1'],
                        "novel_TP": novel_eval['true_positives'],
                        "novel_FP": novel_eval['false_positives'],
                        "novel_FN": novel_eval['false_negatives'],
                        "novel_TN": novel_eval['true_negatives'],
                        "n_novel_clusters": len(novel_cluster_ids),
                    }
                    new_record.update(irl_results_novel)
                    results.append(new_record)
            
            else:
                if not novel_cluster_ids:
                    print("  No novel clusters detected. All finetuned clusters are covered by baseline agents.")
                else:
                    for novel_cid in sorted(novel_cluster_ids):
                        cluster_mask_ft = (labels_all_ft == novel_cid)
                        manager_indices_ft = indices_all_ft[cluster_mask_ft]
                        
                        novel_trajectories = [trajectory_manager[i]['original_trajectory'] for i in manager_indices_ft]
                        novel_trajectories_with_rew = [trajectory_manager[i]['original_trajectory_with_rew'] for i in manager_indices_ft]
                        novel_true_labels = [trajectory_manager[i]['real_cluster_label'] for i in manager_indices_ft]

                        if not novel_trajectories:
                            print(f"  Novel cluster {novel_cid} has no trajectories. Skipping.")
                            continue

                        print(f"  Training agent for novel cluster {novel_cid} ({len(novel_trajectories)} trajectories)...")
                        irl_results_novel = train_and_evaluate_irl_agent(
                            cluster_id=novel_cid,
                            trajectories=novel_trajectories,
                            trajectories_with_rew=novel_trajectories_with_rew,
                            true_labels=novel_true_labels,
                            env_name=env_name,
                            env_id=env_id,
                            tr_name=tr_name,
                            seed=SEED,
                            base_learner_policy=MlpPolicy,
                            stage="finetuned_novel",
                            rl_timesteps=RL_TS,
                        )
                        
                        if irl_results_novel:
                            new_record = {
                                "seed": SEED, "ratio": args.ratio, "env_name": env_name, "tr_name": tr_name,
                                "cluster_id": f"novel_{novel_cid}",
                                "num_trajs_in_cluster": len(novel_trajectories),
                                "state_scaler": state_scaler,
                                "granularity": granularity,
                                "NMI_final": nmi_ft,
                                "ARI_final": ari_ft,
                                "Silhouette_final": silhouette_ft,
                                "stage": "finetuned_novel",
                                "novel_precision": novel_eval['precision'],
                                "novel_recall": novel_eval['recall'],
                                "novel_f1": novel_eval['f1'],
                                "novel_TP": novel_eval['true_positives'],
                                "novel_FP": novel_eval['false_positives'],
                                "novel_FN": novel_eval['false_negatives'],
                                "novel_TN": novel_eval['true_negatives'],
                                "n_novel_clusters": len(novel_cluster_ids),
                            }
                            new_record.update(irl_results_novel)
                            results.append(new_record)
                            df_partial = pd.DataFrame(results)
                            df_partial.to_csv(csv_file_path, index=False)
                            print(f"  Results for novel cluster {novel_cid} saved.")
        if results:
            final_df = pd.DataFrame(results)
            final_df.to_csv(csv_file_path, index=False)
            print(f"\nAll experiments done. Final results saved to {csv_file_path}")

        
if __name__ == "__main__":
    main()