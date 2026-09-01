"""
Aggregate ablation results and compute mean ± std for each loss type and environment.
"""

import os
import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import argparse


def aggregate_ablation_results(env_id: str = None, stage: str = None):
    """
    Aggregate ablation results from CSV files.
    
    Args:
        env_id: Specific environment to aggregate (None = all)
        stage: Specific stage to filter ('baseline', 'finetuned', None = all)
    """
    ablation_dir = "./ablation_results"
    
    if not os.path.exists(ablation_dir):
        print(f"No ablation results found in {ablation_dir}")
        return
    
    # Find all result files
    if env_id:
        csv_files = [f"ablation_results_{env_id}.csv"]
    else:
        csv_files = [f for f in os.listdir(ablation_dir) if f.startswith("ablation_results_") and f.endswith(".csv")]
    
    all_results = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(ablation_dir, csv_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_results.append(df)
    
    if not all_results:
        print("No results found.")
        return
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Filter by stage if specified
    if stage:
        df_all = df_all[df_all['stage'] == stage]
    
    print("=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Group by environment, loss_type, and stage
    for env in df_all['env_id'].unique():
        print(f"\n{'=' * 40}")
        print(f"Environment: {env}")
        print(f"{'=' * 40}")
        
        df_env = df_all[df_all['env_id'] == env]
        
        for stg in df_env['stage'].unique():
            print(f"\n--- Stage: {stg} ---")
            df_stage = df_env[df_env['stage'] == stg]
            
            # Aggregate by loss_type
            summary = df_stage.groupby('loss_type').agg({
                'NMI': ['mean', 'std', 'count'],
                'ARI': ['mean', 'std'],
                'Silhouette': ['mean', 'std'],
            }).round(4)
            
            # Flatten column names
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            summary = summary.rename(columns={'NMI_count': 'n_seeds'})
            
            # Create formatted output
            print(f"\n{'Loss Type':<10} | {'NMI':^20} | {'ARI':^20} | {'Silhouette':^20} | {'Seeds':^6}")
            print("-" * 85)
            
            for loss_type in summary.index:
                row = summary.loc[loss_type]
                nmi_str = f"{row['NMI_mean']:.4f} ± {row['NMI_std']:.4f}"
                ari_str = f"{row['ARI_mean']:.4f} ± {row['ARI_std']:.4f}"
                sil_str = f"{row['Silhouette_mean']:.4f} ± {row['Silhouette_std']:.4f}"
                n_seeds = int(row['n_seeds'])
                print(f"{loss_type:<10} | {nmi_str:^20} | {ari_str:^20} | {sil_str:^20} | {n_seeds:^6}")
    
    # Save aggregated summary to CSV
    summary_path = os.path.join(ablation_dir, f"ablation_summary_{env_id}.csv")
    
    # Create detailed summary DataFrame
    summary_records = []
    for env in df_all['env_id'].unique():
        df_env = df_all[df_all['env_id'] == env]
        for stg in df_env['stage'].unique():
            df_stage = df_env[df_env['stage'] == stg]
            for loss_type in df_stage['loss_type'].unique():
                df_loss = df_stage[df_stage['loss_type'] == loss_type]
                summary_records.append({
                    'env_id': env,
                    'stage': stg,
                    'loss_type': loss_type,
                    'n_seeds': len(df_loss),
                    'NMI_mean': df_loss['NMI'].mean(),
                    'NMI_std': df_loss['NMI'].std(),
                    'ARI_mean': df_loss['ARI'].mean(),
                    'ARI_std': df_loss['ARI'].std(),
                    'Silhouette_mean': df_loss['Silhouette'].mean(),
                    'Silhouette_std': df_loss['Silhouette'].std(),
                })
    
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n\nSummary saved to {summary_path}")
    
    return df_summary


def main():
    parser = argparse.ArgumentParser(description='Aggregate ablation results')
    parser.add_argument('--env', type=str, default=None, 
                        choices=['Reacher-v4', 'Pusher-v4', 'Walker2d-v4'],
                        help='Specific environment to aggregate (default: all)')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['complete'],
                        help='Specific stage to filter (default: all)')
    parser.add_argument('--latex', action='store_true',
                        help='Output results in LaTeX table format')
    args = parser.parse_args()
    
    df_summary = aggregate_ablation_results(env_id=args.env, stage=args.stage)
    
    if args.latex and df_summary is not None:
        print("\n" + "=" * 80)
        print("LATEX TABLE FORMAT")
        print("=" * 80)
        
        for env in df_summary['env_id'].unique():
            print(f"\n% {env}")
            print(r"\begin{tabular}{l|ccc}")
            print(r"\hline")
            print(r"Loss Type & NMI & ARI & Silhouette \\")
            print(r"\hline")
            
            df_env = df_summary[df_summary['env_id'] == env]
            for _, row in df_env.iterrows():
                nmi = f"${row['NMI_mean']:.3f} \\pm {row['NMI_std']:.3f}$"
                ari = f"${row['ARI_mean']:.3f} \\pm {row['ARI_std']:.3f}$"
                sil = f"${row['Silhouette_mean']:.3f} \\pm {row['Silhouette_std']:.3f}$"
                print(f"{row['loss_type']} & {nmi} & {ari} & {sil} \\\\")
            
            print(r"\hline")
            print(r"\end{tabular}")


if __name__ == "__main__":
    main()