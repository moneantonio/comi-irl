# CoMI-IRL

Code for the paper A Behavior-first approach to Multi-Intention Inverse Reinforcement Learning (link: TODO), which introduces **CoMI-IRL** (Contrastive Multi-Intention Inverse Reinforcement Learning).

## Repository layout

| File | Purpose |
|---|---|
| `main.py` | Entry point for the core CoMI-IRL pipeline: loads expert trajectories, runs behavior encoding, graph clustering, and IRL training (AIRL/GAIL/SQIL) per discovered mode. |
| `be.py` | Behavior encoder (trajectory embedding) model. |
| `loss.py` | Loss terms used to train the behavior encoder (contrastive, infomax, segmentation, stability). |
| `training.py` | Training loops for the encoder and downstream IRL learners. |
| `inference.py` | Inference / evaluation utilities (post-training). |
| `graph_clustering.py` | Graph-based clustering of encoded trajectories, incl. Leiden partitioning (needs `leidenalg` + `python-igraph`), KMeans/HDBSCAN baselines, and clustering-quality metrics. |
| `irl_parallel.py` | Parallel execution of IRL training across clusters/seeds. |
| `khgail.py` | K-GAIL / G-GAIL / K-AIRL / G-AIRL baselines: cluster trajectories (KMeans/HDBSCAN/Leiden) then train one IRL learner per cluster. |
| `deep_dp_birl_stress_test.py` | Bayesian Nonparametric IRL baselines (Choi & Kim, BNIRL, BNIRL-subgoal). |
| `epic_reward_evaluation.py` | EPIC distance evaluation of learned reward functions against ground truth. |
| `graph_clustering.py` / `k_values.py` | Clustering utilities and sweep over number-of-modes `k`. |
| `ablation_results.py` | Aggregates and summarizes ablation run outputs. |
| `datasets.py`, `utils.py` | Shared dataset wrappers and utility functions. |
| `new_create_experts.py`, `new_deploy_expert.py` | Scripts to train/export expert policies and generate expert demonstration trajectories. |
| `run_*.sh` | Example launch scripts for the experiments in the paper (baselines, ablations, finetuning, full sweep). Read before running — several loop over seeds/environments and some blocks are commented out to select a subset. |
| `pdf/*.tex` | LaTeX sources for figures used in the paper. |
| `expert_trajectories_new/` | Pickled expert demonstration trajectories used as IRL input (`*_task_N.pkl`, `*_task_N_withrew.pkl`, `env_spec_*.pkl`). |
| `expert_policies_new/` | Pretrained Stable-Baselines3 expert policies (`.zip`) used to generate the trajectories above. |
| `epic_reward_eval_outputs/` | Precomputed EPIC evaluation outputs (example for Pusher-v4). |

**Not included in this repo** (excluded via `.gitignore`, too large for a code release): trained model checkpoints (`models/`), per-method learner outputs (`learners_*`), CSV result dumps (`csvs/`), and raw ablation dumps (`ablation_results/*.csv`). Re-run the scripts below to regenerate them.

## Dependencies

No pinned `requirements.txt` yet — install these into a Python 3.10+ environment:

```
torch
numpy
pandas
scipy
scikit-learn
networkx
matplotlib
seaborn
umap-learn
tqdm
gymnasium
mo-gymnasium
stable-baselines3
imitation
leidenalg
python-igraph
```

### `essinfogail` dependency

`main.py` and `khgail.py` import environment wrappers from `essinfogail.envs` (custom MuJoCo environment wrappers for Reacher/Pusher/Walker2d/Hopper/HalfCheetah with multi-intention experts, plus `essinfogail/expert_imitation_trajectories/`). This is a separate project by other developers, **not included** in this repository: [Ess-InfoGAIL](https://github.com/tRNAoO/Ess-InfoGAIL). Clone it and place it as an `essinfogail/` directory at the repo root before running `main.py` or `khgail.py`.

## Usage

Core CoMI-IRL pipeline, one environment/algorithm/seed at a time:

```bash
python main.py -Pv4 -seed 0 -airl --irl_training --parallel_irl
```

Key `main.py` flags:
- Environment (pick one): `-Rv4` (Reacher-v4), `-Pv4` (Pusher-v4), `-W2D` (Walker2d-v4), `-Ho4` (Hopper-v4), `-HC5` (HalfCheetah-v5)
- IRL algorithm (pick one): `-airl`, `-gail`, `-sqil`
- `--irl_training` — run the IRL training phase; `--parallel_irl` — parallelize it
- `--ablation` — use the ablation model variant (no RFF / TemporalConv)
- `--finetuning --num_unseen_modes N` — finetuning split with `N` held-out modes
- `--alpha/--beta/--gamma/--delta` — loss term weights (contrastive/infomax/segmentation/stability)
- `--save_abl_results` — dump ablation metrics
- `-vC/-vG/-vO/-3d/-r` — visualization flags (clusters, quality graphs, original trajectories, 3D UMAP, env rendering)

Baselines (K-GAIL/G-GAIL/K-AIRL/G-AIRL):

```bash
python khgail.py --env Pusher-v4 --seed 0 --clusterer leiden --irl airl --parallel
python khgail.py --env Pusher-v4 --seed 0 --clusterer kmeans --k 6 --irl gail --parallel
```

BNIRL / Choi-Kim baselines:

```bash
python deep_dp_birl_stress_test.py --env Pusher-v4 --seed 0 --iters 10 \
  --local-new-component-steps 0 --verbose --mode bnirl_subgoal --dp-alpha 0.05
```

Reward evaluation (EPIC distance against ground-truth reward):

```bash
python epic_reward_evaluation.py --env all
```

Example launch scripts (`run_*.sh`) reproduce the paper's experiment sweeps — inspect each for the exact commands/seeds/environments before running, several sections are commented out to select a subset.

## Citation

To be Added
