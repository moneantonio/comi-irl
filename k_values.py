import pandas as pd
import numpy as np
from pathlib import Path

# Base path to learners_K-AIRL
base_path = Path("/Users/antoniomone/Desktop/PhD/PHD Projects/CoMI-IRL/comi-irl/learners_K-AIRL")

# Dictionary to store results
results = {}

# Iterate through environments
for env_dir in sorted(base_path.iterdir()):
    if not env_dir.is_dir():
        continue
    
    env_name = env_dir.name
    results[env_name] = {}
    
    # Collect K values for this environment
    k_values = set()
    for seed_dir in env_dir.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            for k_dir in seed_dir.iterdir():
                if k_dir.is_dir() and k_dir.name.startswith("K_"):
                    k_values.add(k_dir.name)
    
    # For each K value
    for k_folder in sorted(k_values):
        k_num = k_folder.replace("K_", "")
        
        nmi_values = []
        ari_values = []
        silhouette_values = []
        
        # Iterate through seeds
        for seed_dir in sorted(env_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            
            summary_path = seed_dir / k_folder / "summary.csv"
            
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                
                # Take the first row (metrics are the same for all clusters within a seed/K)
                nmi_values.append(df['nmi'].iloc[0])
                ari_values.append(df['ari'].iloc[0])
                silhouette_values.append(df['silhouette'].iloc[0])
        
        if nmi_values:
            results[env_name][k_num] = {
                'nmi_mean': np.mean(nmi_values),
                'nmi_std': np.std(nmi_values),
                'ari_mean': np.mean(ari_values),
                'ari_std': np.std(ari_values),
                'silhouette_mean': np.mean(silhouette_values),
                'silhouette_std': np.std(silhouette_values),
            }

# Print results
print(f"{'Environment':<20} {'K':<5} {'NMI':<25} {'ARI':<25} {'Silhouette':<25}")
print("-" * 100)

for env, k_dict in sorted(results.items()):
    for k, metrics in sorted(k_dict.items(), key=lambda x: int(x[0])):
        nmi_str = f"{metrics['nmi_mean']:.2f} ± {metrics['nmi_std']:.2f}"
        ari_str = f"{metrics['ari_mean']:.2f} ± {metrics['ari_std']:.2f}"
        sil_str = f"{metrics['silhouette_mean']:.2f} ± {metrics['silhouette_std']:.2f}"
        print(f"{env:<20} {k:<5} {nmi_str:<25} {ari_str:<25} {sil_str:<25}")
    print()