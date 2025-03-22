#!/usr/bin/env python
"""
Generate publication-quality LaTeX tables from WandB sweep results.
This script fetches the results of a sweep and generates tables for:
1. Parameter comparison showing which configurations performed best
2. Performance comparison between methods (L2D vs baselines)
3. Summary statistics for quick reference
"""

import os
import sys
import argparse
import wandb
import pandas as pd
import numpy as np
import datetime
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from WandB results')
    parser.add_argument('--project', type=str, default='jssp-weighted-sum',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--sweep_id', type=str, required=True,
                        help='WandB sweep ID to analyze')
    parser.add_argument('--output', type=str, default='./data/results/latex_tables',
                        help='Output directory for LaTeX tables')
    parser.add_argument('--table_type', type=str, default='all',
                        help='Type of table to generate (env, model, feature, reward, all)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top configurations to include in tables')
    return parser.parse_args()

def fetch_sweep_runs(entity, project, sweep_id):
    """Fetch runs and config from a WandB sweep."""
    api = wandb.Api()
    sweep_path = f"{project}/{sweep_id}"
    if entity:
        sweep_path = f"{entity}/{sweep_path}"
        
    try:
        sweep = api.sweep(sweep_path)
        runs = sorted([run for run in sweep.runs if run.state == 'finished'], 
                      key=lambda r: r.summary.get('validation_win_vs_wspt', 0), 
                      reverse=True)
        return runs, sweep.config
    except Exception as e:
        print(f"Error fetching sweep data: {e}")
        return [], {}

def get_param_keys_by_type(sweep_type, sweep_config):
    """Determine which parameters to include in tables based on sweep type."""
    param_groups = {
        'env': ["n_j", "n_m", "low", "high", "weight_low", "weight_high", 
                "max_updates", "num_envs", "rewardscale"],
        'model': ["num_layers", "hidden_dim", "lr", "gamma", "k_epochs", "eps_clip", 
                 "neighbor_pooling_type", "graph_pool_type", "num_mlp_layers_feature_extract",
                 "num_mlp_layers_actor", "hidden_dim_actor", "num_mlp_layers_critic", 
                 "hidden_dim_critic", "ploss_coef", "vloss_coef", "entloss_coef"],
        'feature': ["feature_set"],
        'reward': ["reward_strategy"]
    }
    
    # If sweep_type is explicitly provided, return those params
    if sweep_type in param_groups:
        return param_groups[sweep_type]
    
    # Otherwise, try to infer from sweep config
    available_params = set(sweep_config.get('parameters', {}).keys())
    
    # Count how many parameters from each group are in the sweep config
    param_counts = {
        group: len(set(params) & available_params) 
        for group, params in param_groups.items()
    }
    
    # If no parameters are found, return all available parameters
    if max(param_counts.values(), default=0) == 0:
        return list(available_params)
    
    # Return the parameters for the group with the most matches
    best_group = max(param_counts.items(), key=lambda x: x[1])[0]
    return param_groups[best_group]

def create_parameter_table(runs, key_params, sort_by='validation_win_vs_wspt', top_k=5):
    """Create a parameter comparison table from WandB runs."""
    if not runs:
        return "No finished runs found.", pd.DataFrame()
    
    # Collect data from runs
    data = []
    for run in runs:
        run_data = {param: run.config.get(param, "N/A") for param in key_params}
        
        # Add key metrics
        for metric in ['validation_weighted_sum', 'validation_win_vs_wspt', 
                       'validation_win_vs_spt', 'validation_win_rate',
                       'validation_improvement_over_wspt']:
            run_data[metric] = run.summary.get(metric, 0)
        
        # Add run name and URL for reference
        run_data['run_name'] = run.name
        run_data['url'] = run.url
        
        data.append(run_data)
    
    df = pd.DataFrame(data)
    
    # Handle list parameters (convert to string representation)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Sort and select top k runs
    df = df.sort_values(sort_by, ascending=False).head(top_k)
    
    # Create a simplified version for the LaTeX table
    latex_df = df[key_params + ['validation_win_vs_wspt', 'validation_weighted_sum']]
    latex_df.columns = [col.replace('validation_', '') for col in latex_df.columns]
    
    # Generate LaTeX table with booktabs style
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Top-performing configurations sorted by win rate against WSPT}
\\begin{tabular}{%s}
\\toprule
%s \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\label{tab:top_configs}
\\end{table}
""" % ('l' + 'c' * (len(latex_df.columns) - 1),  # Column formats
       ' & '.join(latex_df.columns).replace('_', '\\_'),  # Header row
       '\\\\\n'.join([' & '.join([str(row[0])] + 
                                [f"{x:.2f}" if isinstance(x, float) else str(x) 
                                 for x in row[1:]]) 
                      for _, row in latex_df.iterrows()])  # Data rows
    )
    
    return latex_table, df

def create_performance_summary(runs, top_k=5):
    """Create a summary of performance statistics across the sweep."""
    if not runs:
        return "No finished runs found.", pd.DataFrame()
    
    # Extract the key performance metrics from all finished runs
    metrics = ['validation_weighted_sum', 'validation_win_vs_wspt', 
               'validation_win_vs_spt', 'validation_win_rate',
               'validation_improvement_over_wspt']
    
    metrics_data = {}
    for metric in metrics:
        values = [run.summary.get(metric, 0) for run in runs]
        metrics_data[metric] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame(metrics_data).T
    
    # Get best run for each metric
    best_runs = {}
    for metric in metrics:
        if metric == 'validation_weighted_sum':
            # Lower is better for weighted sum
            best_idx = np.argmin([run.summary.get(metric, float('inf')) for run in runs])
        else:
            # Higher is better for win rates
            best_idx = np.argmax([run.summary.get(metric, 0) for run in runs])
        
        best_runs[metric] = runs[best_idx].name
    
    # Create LaTeX table
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance summary across all sweep configurations}
\\begin{tabular}{lrrrrr}
\\toprule
Metric & Min & Max & Mean & Median & Std Dev \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\label{tab:perf_summary}
\\end{table}
""" % '\\\\\n'.join([f"{metric.replace('validation_', '').replace('_', '\\_')} & " + 
                    ' & '.join([f"{summary_df.loc[metric, col]:.2f}" 
                               for col in ['min', 'max', 'mean', 'median', 'std']])
                   for metric in metrics])
    
    # Create best runs table
    best_runs_latex = """\\begin{table}[htbp]
\\centering
\\caption{Best performing runs by metric}
\\begin{tabular}{ll}
\\toprule
Metric & Best Run \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\label{tab:best_runs}
\\end{table}
""" % '\\\\\n'.join([f"{metric.replace('validation_', '').replace('_', '\\_')} & {best_runs[metric]}"
                    for metric in metrics])
    
    return latex_table, best_runs_latex, summary_df, best_runs

def create_performance_profile_plot(runs, output_path, top_k=5):
    """Generate a performance profile plot comparing top runs with baselines."""
    if not runs or len(runs) < 2:
        return
    
    # Get the top k runs by win rate vs WSPT
    top_runs = sorted(runs, key=lambda r: r.summary.get('validation_win_vs_wspt', 0), 
                     reverse=True)[:top_k]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Add baseline performance profiles if available
    baselines = {'SPT': [], 'WSPT': []}
    l2d_values = []
    
    # Collect validation metrics for top runs
    for i, run in enumerate(top_runs):
        history = run.history()
        if 'validation_weighted_sum' in history.columns:
            l2d_values.append(history['validation_weighted_sum'].values)
    
    # If we don't have any validation data, return
    if not l2d_values:
        return
    
    # Plot performance profiles
    tau_values = np.linspace(1.0, 1.5, 100)  # Performance ratio values
    
    # For each top run, plot its profile
    for i, values in enumerate(l2d_values):
        run_name = top_runs[i].name
        
        # Sort the performance ratios
        sorted_ratios = np.sort(values / np.min(values))
        # Calculate the cumulative distribution
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        # Plot step function
        plt.step(sorted_ratios, y, where='post', label=f"Run {i+1}", linewidth=2)
    
    plt.title('Performance Profile for Top Configurations', fontsize=14)
    plt.xlabel('Performance Ratio (τ)', fontsize=12)
    plt.ylabel('Probability P(r_{p,s} ≤ τ)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance profile plot saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Add automatic timestamp to filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fetch runs and sweep config
    print(f"Fetching runs for sweep {args.sweep_id}...")
    runs, sweep_config = fetch_sweep_runs(args.entity, args.project, args.sweep_id)
    
    if not runs:
        print("No runs found or error fetching sweep data.")
        return
    
    print(f"Found {len(runs)} runs in sweep {args.sweep_id}")
    
    # Determine the sweep type and parameters to include
    sweep_type = args.table_type if args.table_type != 'all' else None
    key_params = get_param_keys_by_type(sweep_type, sweep_config)
    
    print(f"Using parameters: {key_params}")
    
    # Create parameter table
    print("Generating parameter comparison table...")
    param_table, param_df = create_parameter_table(runs, key_params, top_k=args.top_k)
    param_table_file = os.path.join(args.output, f"{args.table_type}_params_table_{timestamp}.tex")
    with open(param_table_file, "w") as f:
        f.write(param_table)
    
    # Create performance summary
    print("Generating performance summary tables...")
    perf_summary, best_runs_table, summary_df, best_runs = create_performance_summary(runs, top_k=args.top_k)
    
    perf_summary_file = os.path.join(args.output, f"{args.table_type}_perf_summary_{timestamp}.tex")
    with open(perf_summary_file, "w") as f:
        f.write(perf_summary)
    
    best_runs_file = os.path.join(args.output, f"{args.table_type}_best_runs_{timestamp}.tex")
    with open(best_runs_file, "w") as f:
        f.write(best_runs_table)
    
    # Create performance profile plot
    print("Generating performance profile plot...")
    profile_plot_file = os.path.join(args.output, f"{args.table_type}_perf_profile_{timestamp}.png")
    create_performance_profile_plot(runs, profile_plot_file, top_k=args.top_k)
    
    # Save summary data as CSV for easier analysis
    summary_csv_file = os.path.join(args.output, f"{args.table_type}_summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv_file)
    
    param_csv_file = os.path.join(args.output, f"{args.table_type}_params_{timestamp}.csv")
    param_df.to_csv(param_csv_file, index=False)
    
    # Save a YAML file with metadata about the sweep
    sweep_runs_info = {
        'sweep_id': args.sweep_id,
        'project': args.project,
        'entity': args.entity,
        'timestamp': timestamp,
        'total_runs': len(runs),
        'table_type': args.table_type,
        'generated_files': {
            'param_table': param_table_file,
            'perf_summary': perf_summary_file,
            'best_runs': best_runs_file,
            'profile_plot': profile_plot_file,
            'summary_csv': summary_csv_file,
            'param_csv': param_csv_file
        },
        'best_runs': best_runs
    }
    
    sweep_info_file = os.path.join(args.output, f"{args.table_type}_sweep_info_{timestamp}.yaml")
    with open(sweep_info_file, "w") as f:
        yaml.dump(sweep_runs_info, f)
    
    print(f"\nAll outputs generated successfully in {args.output}")
    print(f"Parameter table: {param_table_file}")
    print(f"Performance summary: {perf_summary_file}")
    print(f"Best runs table: {best_runs_file}")
    print(f"Performance profile plot: {profile_plot_file}")
    print(f"Summary data: {summary_csv_file}")

if __name__ == "__main__":
    main()