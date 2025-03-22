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
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate comprehensive tables and visualizations from WandB results')
    parser.add_argument('--project', type=str, default='jssp-weighted-sum',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--sweep_id', type=str, required=True,
                        help='WandB sweep ID to analyze')
    parser.add_argument('--output', type=str, default='./data/results/latex_tables',
                        help='Output directory for tables and visualizations')
    parser.add_argument('--table_type', type=str, default='all',
                        help='Type of sweep (env, model, feature, reward, all)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top configurations to include in tables')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed tables and visualizations')
    return parser.parse_args()

def fetch_sweep_runs(entity, project, sweep_id):
    """Fetch runs and config from a WandB sweep."""
    api = wandb.Api()
    sweep_path = f"{project}/{sweep_id}"
    if entity:
        sweep_path = f"{entity}/{sweep_path}"
        
    try:
        sweep = api.sweep(sweep_path)
        runs = [run for run in sweep.runs if run.state == 'finished']
        
        # Sort runs by validation win rate (if available)
        if runs and 'validation_win_vs_wspt' in runs[0].summary:
            runs = sorted(runs, key=lambda r: r.summary.get('validation_win_vs_wspt', 0), reverse=True)
        elif runs and 'validation_weighted_sum' in runs[0].summary:
            # Lower weighted sum is better
            runs = sorted(runs, key=lambda r: r.summary.get('validation_weighted_sum', float('inf')))
        
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
    if sweep_type.lower() in param_groups:
        return param_groups[sweep_type.lower()]
    
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
        metric_keys = [
            'validation_weighted_sum', 'validation_win_vs_wspt', 
            'validation_win_vs_spt', 'validation_win_rate',
            'validation_improvement_over_wspt', 'validation_improvement_over_spt'
        ]
        
        for metric in metric_keys:
            run_data[metric] = run.summary.get(metric, 0)
        
        # Add training metrics
        if hasattr(run, 'history') and run.history() is not None:
            history = run.history()
            if not history.empty:
                # Get last episode metrics
                last_epoch = history['episode'].max() if 'episode' in history.columns else None
                if last_epoch is not None:
                    last_row = history[history['episode'] == last_epoch].iloc[0]
                    for metric in ['weighted_sum', 'loss', 'reward']:
                        if metric in last_row:
                            run_data[f'final_{metric}'] = last_row[metric]
        
        # Add run name and URL for reference
        run_data['run_name'] = run.name
        run_data['url'] = run.url
        
        data.append(run_data)
    
    df = pd.DataFrame(data)
    
    # Handle list parameters (convert to string representation)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    
    # Sort and select top k runs
    if sort_by in df.columns:
        if sort_by == 'validation_weighted_sum':
            # Lower is better for weighted sum
            df = df.sort_values(sort_by).head(top_k)
        else:
            # Higher is better for win rates
            df = df.sort_values(sort_by, ascending=False).head(top_k)
    
    # Create a simplified version for the LaTeX table
    table_metrics = [
        'validation_win_vs_wspt', 'validation_win_vs_spt', 
        'validation_weighted_sum', 'validation_improvement_over_wspt'
    ]
    
    available_metrics = [m for m in table_metrics if m in df.columns]
    latex_df = df[key_params + available_metrics]
    
    # Clean up column names for LaTeX
    column_map = {
        'validation_win_vs_wspt': 'Win vs WSPT (\%)',
        'validation_win_vs_spt': 'Win vs SPT (\%)',
        'validation_weighted_sum': 'Weighted Sum',
        'validation_improvement_over_wspt': 'Impr. WSPT (\%)'
    }
    
    latex_df.columns = [column_map.get(col, col.replace('_', '\\_')) for col in latex_df.columns]
    
    # Generate LaTeX table with booktabs style
    column_format = 'l' + 'c' * (len(latex_df.columns) - 1)
    
    # Clean up data for LaTeX
    for col in latex_df.columns:
        if latex_df[col].dtype == 'float64':
            if 'Win' in col or 'Impr' in col:
                # Format percentages with 1 decimal place
                latex_df[col] = latex_df[col].apply(lambda x: f"{x:.1f}")
            else:
                # Format other floats with 2 decimal places
                latex_df[col] = latex_df[col].apply(lambda x: f"{x:.2f}")
    
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Top {top_k} configurations ranked by win rate against WSPT}}
\\begin{{tabular}}{{{column_format}}}
\\toprule
{' & '.join(latex_df.columns)} \\\\
\\midrule
{latex_df.to_string(index=False, header=False, formatters={i: lambda x: str(x) for i in range(len(latex_df.columns))}).replace('\n', ' \\\\\n')}
\\bottomrule
\\end{{tabular}}
\\label{{tab:top_configs}}
\\end{{table}}
"""
    
    return latex_table, df

def create_performance_summary(runs, sweep_type):
    """Create a summary of performance statistics across the sweep."""
    if not runs:
        return "No finished runs found.", pd.DataFrame()
    
    # Extract the key performance metrics from all finished runs
    metrics = [
        'validation_weighted_sum', 'validation_win_vs_wspt', 
        'validation_win_vs_spt', 'validation_win_rate',
        'validation_improvement_over_wspt', 'validation_improvement_over_spt'
    ]
    
    metrics_data = {}
    for metric in metrics:
        values = [run.summary.get(metric, 0) for run in runs if metric in run.summary]
        if not values:
            continue
        metrics_data[metric] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
    
    # If no metrics found, return empty
    if not metrics_data:
        return "No metrics found in runs.", None, None, None
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame(metrics_data).T
    
    # Get best run for each metric
    best_runs = {}
    for metric in metrics_data.keys():
        if metric == 'validation_weighted_sum':
            # Lower is better for weighted sum
            best_idx = np.argmin([run.summary.get(metric, float('inf')) for run in runs if metric in run.summary])
        else:
            # Higher is better for win rates
            best_idx = np.argmax([run.summary.get(metric, 0) for run in runs if metric in run.summary])
        
        best_runs[metric] = runs[best_idx].name
    
    # Format column names for LaTeX
    column_names = {
        'min': 'Min', 
        'max': 'Max', 
        'mean': 'Mean', 
        'median': 'Median', 
        'std': 'Std Dev'
    }
    
    # Format metric names for LaTeX
    metric_names = {
        'validation_weighted_sum': 'Weighted Sum',
        'validation_win_vs_wspt': 'Win vs WSPT (\%)',
        'validation_win_vs_spt': 'Win vs SPT (\%)',
        'validation_win_rate': 'Overall Win Rate (\%)',
        'validation_improvement_over_wspt': 'Impr. WSPT (\%)',
        'validation_improvement_over_spt': 'Impr. SPT (\%)'
    }
    
    # Create LaTeX table
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance summary across all {sweep_type} sweep configurations}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Metric & {' & '.join(column_names.values())} \\\\
\\midrule
{chr(10).join([f"{metric_names.get(metric, metric)} & " + 
              ' & '.join([f"{summary_df.loc[metric, col]:.2f}" 
                         for col in summary_df.columns])
             for metric in summary_df.index])}
\\bottomrule
\\end{{tabular}}
\\label{{tab:perf_summary_{sweep_type}}}
\\end{{table}}
"""
    
    # Create best runs table
    best_runs_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Best performing runs by metric in {sweep_type} sweep}}
\\begin{{tabular}}{{ll}}
\\toprule
Metric & Best Run \\\\
\\midrule
{chr(10).join([f"{metric_names.get(metric, metric)} & {best_runs[metric]}"
              for metric in best_runs])}
\\bottomrule
\\end{{tabular}}
\\label{{tab:best_runs_{sweep_type}}}
\\end{{table}}
"""
    
    return latex_table, best_runs_latex, summary_df, best_runs

def create_feature_importance_plot(runs, output_path):
    """Create feature importance visualization for feature sweeps."""
    if not runs:
        return
    
    # Extract feature set information
    feature_data = []
    for run in runs:
        if 'feature_set' not in run.config:
            continue
            
        features = run.config['feature_set']
        if not isinstance(features, list):
            continue
            
        win_rate = run.summary.get('validation_win_vs_wspt', 0)
        weighted_sum = run.summary.get('validation_weighted_sum', float('inf'))
        
        feature_data.append({
            'run': run.name,
            'features': features,
            'win_rate': win_rate,
            'weighted_sum': weighted_sum
        })
    
    if not feature_data:
        print("No feature data found in runs.")
        return
    
    # Create DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # Get all unique features
    all_features = set()
    for features in feature_df['features']:
        all_features.update(features)
    all_features = sorted(all_features)
    
    # Create feature presence matrix
    feature_presence = pd.DataFrame(index=feature_df.index, columns=all_features)
    for i, features in enumerate(feature_df['features']):
        for feature in all_features:
            feature_presence.iloc[i][feature] = 1 if feature in features else 0
    
    # Add performance metrics
    feature_presence['win_rate'] = feature_df['win_rate']
    feature_presence['weighted_sum'] = feature_df['weighted_sum']
    
    # Calculate correlation with performance
    corr_win_rate = [feature_presence[feature].corr(feature_presence['win_rate']) for feature in all_features]
    corr_weighted_sum = [feature_presence[feature].corr(feature_presence['weighted_sum']) for feature in all_features]
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Win Rate Correlation': corr_win_rate,
        'Weighted Sum Correlation': [-c for c in corr_weighted_sum]  # Invert since lower weighted sum is better
    })
    
    # Sort by absolute correlation with win rate
    importance_df = importance_df.iloc[np.argsort(np.abs(importance_df['Win Rate Correlation']))[::-1]]
    
    # Plot
    ax = importance_df.plot(x='Feature', y=['Win Rate Correlation', 'Weighted Sum Correlation'], 
                           kind='bar', figsize=(12, 8))
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Feature', fontsize=14)
    plt.ylabel('Correlation with Performance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")
    
    # Also create a heatmap of feature co-occurrence
    plt.figure(figsize=(10, 8))
    feature_correlation = feature_presence[all_features].corr()
    sns.heatmap(feature_correlation, annot=True, cmap='coolwarm', fmt='.2f',
               xticklabels=all_features, yticklabels=all_features)
    plt.title('Feature Co-occurrence Correlation', fontsize=16)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = output_path.replace('.png', '_correlation.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature correlation heatmap saved to {heatmap_path}")
    
    return importance_df

def create_model_architecture_comparison(runs, output_path):
    """Create visualization comparing different model architectures for model sweeps."""
    if not runs:
        return
    
    # Extract model architecture parameters
    model_params = [
        'num_layers', 'hidden_dim', 'num_mlp_layers_feature_extract',
        'num_mlp_layers_actor', 'hidden_dim_actor', 'num_mlp_layers_critic', 
        'hidden_dim_critic', 'neighbor_pooling_type', 'graph_pool_type'
    ]
    
    # Check which parameters are present in the runs
    available_params = set()
    for run in runs:
        for param in model_params:
            if param in run.config:
                available_params.add(param)
    
    # Extract parameter values and performance
    model_data = []
    for run in runs:
        model_info = {param: run.config.get(param, None) for param in available_params}
        model_info['run'] = run.name
        model_info['win_rate'] = run.summary.get('validation_win_vs_wspt', 0)
        model_info['weighted_sum'] = run.summary.get('validation_weighted_sum', float('inf'))
        model_data.append(model_info)
    
    if not model_data:
        print("No model architecture data found in runs.")
        return
    
    # Create DataFrame
    model_df = pd.DataFrame(model_data)
    
    # Select numerical parameters for correlation analysis
    numerical_params = [p for p in available_params if p in model_df.columns and 
                       model_df[p].dtype in ['int64', 'float64']]
    
    if not numerical_params:
        print("No numerical parameters found for correlation analysis.")
        return
    
    # Calculate correlation with performance
    corr_win_rate = [model_df[param].corr(model_df['win_rate']) for param in numerical_params]
    corr_weighted_sum = [model_df[param].corr(model_df['weighted_sum']) for param in numerical_params]
    
    # Create parameter importance plot
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Parameter': numerical_params,
        'Win Rate Correlation': corr_win_rate,
        'Weighted Sum Correlation': [-c for c in corr_weighted_sum]  # Invert since lower weighted sum is better
    })
    
    # Sort by absolute correlation with win rate
    importance_df = importance_df.iloc[np.argsort(np.abs(importance_df['Win Rate Correlation']))[::-1]]
    
    # Plot
    ax = importance_df.plot(x='Parameter', y=['Win Rate Correlation', 'Weighted Sum Correlation'], 
                           kind='bar', figsize=(12, 8))
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Model Parameter Importance', fontsize=16)
    plt.xlabel('Parameter', fontsize=14)
    plt.ylabel('Correlation with Performance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model parameter importance plot saved to {output_path}")
    
    # Create a scatter plot for most important parameter vs. performance
    if len(numerical_params) > 0:
        top_param = importance_df.iloc[0]['Parameter']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(model_df[top_param], model_df['win_rate'], alpha=0.7)
        plt.title(f'Effect of {top_param} on Performance', fontsize=16)
        plt.xlabel(top_param, fontsize=14)
        plt.ylabel('Win Rate vs WSPT (%)', fontsize=14)
        plt.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(model_df[top_param], model_df['win_rate'], 1)
        p = np.poly1d(z)
        plt.plot(model_df[top_param], p(model_df[top_param]), "r--", alpha=0.8)
        
        # Add R² value
        corr = model_df[top_param].corr(model_df['win_rate'])
        plt.annotate(f'R = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        scatter_path = output_path.replace('.png', f'_{top_param}_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter scatter plot saved to {scatter_path}")
    
    return importance_df

def create_reward_comparison_plot(runs, output_path):
    """Create visualization comparing different reward formulations for reward sweeps."""
    if not runs:
        return
    
    # Extract reward strategy information
    reward_data = []
    for run in runs:
        if 'reward_strategy' not in run.config:
            continue
            
        strategy = run.config['reward_strategy']
        win_rate = run.summary.get('validation_win_vs_wspt', 0)
        weighted_sum = run.summary.get('validation_weighted_sum', float('inf'))
        
        reward_data.append({
            'run': run.name,
            'strategy': strategy,
            'win_rate': win_rate,
            'weighted_sum': weighted_sum
        })
    
    if not reward_data:
        print("No reward strategy data found in runs.")
        return
    
    # Create DataFrame
    reward_df = pd.DataFrame(reward_data)
    
    # Group by strategy and calculate average performance
    strategy_performance = reward_df.groupby('strategy').agg({
        'win_rate': ['mean', 'std', 'count'],
        'weighted_sum': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-index columns
    strategy_performance.columns = ['_'.join(col).strip('_') for col in strategy_performance.columns.values]
    
    # Sort by win rate
    strategy_performance = strategy_performance.sort_values('win_rate_mean', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Create bars
    bars = plt.bar(strategy_performance['strategy'], strategy_performance['win_rate_mean'], 
                  yerr=strategy_performance['win_rate_std'], capsize=10, 
                  color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    # Add count information
    for i, count in enumerate(strategy_performance['win_rate_count']):
        plt.annotate(f'n={count}',
                    xy=(i, 0),
                    xytext=(0, -15),  # 15 points vertical offset
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=9, color='gray')
    
    plt.title('Reward Strategy Comparison', fontsize=16)
    plt.xlabel('Reward Strategy', fontsize=14)
    plt.ylabel('Win Rate vs WSPT (%)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, max(strategy_performance['win_rate_mean']) * 1.2)  # Add some headroom
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reward strategy comparison plot saved to {output_path}")
    
    # Also create a weighted sum comparison plot
    plt.figure(figsize=(12, 8))
    
    # Create bars
    bars = plt.bar(strategy_performance['strategy'], strategy_performance['weighted_sum_mean'], 
                  yerr=strategy_performance['weighted_sum_std'], capsize=10, 
                  color='lightgreen', edgecolor='black', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    plt.title('Reward Strategy Effect on Weighted Sum', fontsize=16)
    plt.xlabel('Reward Strategy', fontsize=14)
    plt.ylabel('Weighted Sum (lower is better)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    ws_path = output_path.replace('.png', '_weighted_sum.png')
    plt.savefig(ws_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weighted sum comparison plot saved to {ws_path}")
    
    return strategy_performance

def create_environment_parameter_analysis(runs, output_path):
    """Create visualization analyzing effect of environment parameters for environment sweeps."""
    if not runs:
        return
    
    # Extract environment parameters
    env_params = [
        'n_j', 'n_m', 'low', 'high', 'weight_low', 'weight_high',
        'max_updates', 'num_envs'
    ]
    
    # Check which parameters are present in the runs
    available_params = set()
    for run in runs:
        for param in env_params:
            if param in run.config:
                available_params.add(param)
    
    # Extract parameter values and performance
    env_data = []
    for run in runs:
        env_info = {param: run.config.get(param, None) for param in available_params}
        env_info['run'] = run.name
        env_info['win_rate'] = run.summary.get('validation_win_vs_wspt', 0)
        env_info['weighted_sum'] = run.summary.get('validation_weighted_sum', float('inf'))
        env_data.append(env_info)
    
    if not env_data:
        print("No environment parameter data found in runs.")
        return
    
    # Create DataFrame
    env_df = pd.DataFrame(env_data)
    
    # Select numerical parameters for correlation analysis
    numerical_params = [p for p in available_params if p in env_df.columns and 
                       env_df[p].dtype in ['int64', 'float64']]
    
    if not numerical_params:
        print("No numerical parameters found for correlation analysis.")
        return
    
    # Calculate correlation with performance
    corr_win_rate = [env_df[param].corr(env_df['win_rate']) for param in numerical_params]
    corr_weighted_sum = [env_df[param].corr(env_df['weighted_sum']) for param in numerical_params]
    
    # Create parameter importance plot
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Parameter': numerical_params,
        'Win Rate Correlation': corr_win_rate,
        'Weighted Sum Correlation': [-c for c in corr_weighted_sum]  # Invert since lower weighted sum is better
    })
    
    # Sort by absolute correlation with win rate
    importance_df = importance_df.iloc[np.argsort(np.abs(importance_df['Win Rate Correlation']))[::-1]]
    
    # Plot
    ax = importance_df.plot(x='Parameter', y=['Win Rate Correlation', 'Weighted Sum Correlation'], 
                           kind='bar', figsize=(12, 8))
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Environment Parameter Importance', fontsize=16)
    plt.xlabel('Parameter', fontsize=14)
    plt.ylabel('Correlation with Performance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Environment parameter importance plot saved to {output_path}")
    
    # Check if we have problem size parameters and create a visualization
    if 'n_j' in env_df.columns and 'n_m' in env_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Create a problem size column
        env_df['problem_size'] = env_df['n_j'].astype(str) + 'x' + env_df['n_m'].astype(str)
        
        # Group by problem size and calculate average performance
        size_performance = env_df.groupby('problem_size').agg({
            'win_rate': ['mean', 'std', 'count'],
            'weighted_sum': ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-index columns
        size_performance.columns = ['_'.join(col).strip('_') for col in size_performance.columns.values]
        
        # Sort by problem size (parsing NxM format)
        def size_key(size_str):
            parts = size_str.split('x')
            return int(parts[0]) * 100 + int(parts[1])
        
        size_performance = size_performance.iloc[sorted(
            range(len(size_performance)), 
            key=lambda i: size_key(size_performance.iloc[i]['problem_size'])
        )]
        
        # Create bars
        bars = plt.bar(size_performance['problem_size'], size_performance['win_rate_mean'], 
                      yerr=size_performance['win_rate_std'], capsize=10, 
                      color='salmon', edgecolor='black', alpha=0.7)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
        
        plt.title('Effect of Problem Size on Performance', fontsize=16)
        plt.xlabel('Problem Size (NxM)', fontsize=14)
        plt.ylabel('Win Rate vs WSPT (%)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        size_path = output_path.replace('.png', '_problem_size.png')
        plt.savefig(size_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Problem size analysis plot saved to {size_path}")
    
    # Check if we have weight range parameters and create a visualization
    if 'weight_low' in env_df.columns and 'weight_high' in env_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Create a weight range column
        env_df['weight_range'] = env_df['weight_low'].astype(str) + '-' + env_df['weight_high'].astype(str)
        
        # Group by weight range and calculate average performance
        weight_performance = env_df.groupby('weight_range').agg({
            'win_rate': ['mean', 'std', 'count'],
            'weighted_sum': ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-index columns
        weight_performance.columns = ['_'.join(col).strip('_') for col in weight_performance.columns.values]
        
        # Sort by weight range
        def weight_key(range_str):
            parts = range_str.split('-')
            return int(parts[0]) * 100 + int(parts[1])
        
        weight_performance = weight_performance.iloc[sorted(
            range(len(weight_performance)), 
            key=lambda i: weight_key(weight_performance.iloc[i]['weight_range'])
        )]
        
        # Create bars
        bars = plt.bar(weight_performance['weight_range'], weight_performance['win_rate_mean'], 
                      yerr=weight_performance['win_rate_std'], capsize=10, 
                      color='lightblue', edgecolor='black', alpha=0.7)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
        
        plt.title('Effect of Weight Range on Performance', fontsize=16)
        plt.xlabel('Weight Range (min-max)', fontsize=14)
        plt.ylabel('Win Rate vs WSPT (%)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        weight_path = output_path.replace('.png', '_weight_range.png')
        plt.savefig(weight_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Weight range analysis plot saved to {weight_path}")
    
    return importance_df

def create_performance_profile(runs, output_path, metric='validation_weighted_sum'):
    """Generate a performance profile plot comparing run performance distributions."""
    if not runs:
        return
    
    # Check if we have the required metric
    has_metric = [metric in run.summary for run in runs]
    if not all(has_metric):
        print(f"Not all runs have the metric {metric}.")
        return
    
    # Get values
    values = np.array([run.summary[metric] for run in runs])
    
    # For weighted sum, lower is better, so we invert the values
    if 'sum' in metric.lower():
        best_value = np.min(values)
        ratios = values / best_value
    else:
        # For win rates, higher is better
        best_value = np.max(values)
        # If best value is 0, we can't compute ratios
        if best_value == 0:
            print(f"Best value for {metric} is 0, cannot create performance profile.")
            return
        ratios = best_value / values
    
    # Create performance profile
    plt.figure(figsize=(10, 6))
    
    # Define range of tau values
    tau_values = np.linspace(1.0, 2.0, 1000)
    
    # Calculate profiles
    profiles = []
    
    # Get top 5 runs (or fewer if there are fewer runs)
    top_k = min(5, len(runs))
    
    # For each run, calculate percentage of problems with ratio <= tau
    for i in range(top_k):
        ratio = ratios[i]
        profile = [np.mean(ratio <= tau) for tau in tau_values]
        profiles.append((runs[i], profile))
    
    # Plot profiles
    for i, (run, profile) in enumerate(profiles):
        # Generate a distinct color
        color = plt.cm.tab10(i)
        
        # Get run config value to use as label
        if 'feature_set' in run.config:
            label = f"Features: {run.config['feature_set']}"
        elif 'reward_strategy' in run.config:
            label = f"Reward: {run.config['reward_strategy']}"
        elif 'num_layers' in run.config:
            label = f"Layers: {run.config['num_layers']}"
        else:
            label = f"Run {i+1}"
        
        plt.plot(tau_values, profile, label=label, color=color, linewidth=2)
    
    plt.title('Performance Profile', fontsize=16)
    plt.xlabel('Performance Ratio (τ)', fontsize=14)
    plt.ylabel('Probability P(r_{p,s} ≤ τ)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance profile plot saved to {output_path}")

def create_learning_curves(runs, output_path):
    """Create learning curves showing training progress over episodes."""
    if not runs:
        return
    
    # Get top 5 runs (or fewer if there are fewer runs)
    top_k = min(5, len(runs))
    top_runs = runs[:top_k]
    
    # Metrics to plot
    metrics = ['weighted_sum', 'reward', 'loss']
    
    # Create plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for i, run in enumerate(top_runs):
            # Skip if run doesn't have history
            if not hasattr(run, 'history') or run.history() is None:
                continue
                
            history = run.history()
            if history.empty or 'episode' not in history.columns or metric not in history.columns:
                continue
            
            # Get training data
            episodes = history['episode'].values
            values = history[metric].values
            
            # Skip if we don't have enough data points
            if len(episodes) < 2:
                continue
            
            # Generate a distinct color
            color = plt.cm.tab10(i)
            
            # Get run config value to use as label
            if 'feature_set' in run.config:
                label = f"Features: {run.config['feature_set']}"
            elif 'reward_strategy' in run.config:
                label = f"Reward: {run.config['reward_strategy']}"
            elif 'num_layers' in run.config:
                label = f"Layers: {run.config['num_layers']}"
            else:
                label = f"Run {i+1}"
            
            # Plot raw data with low alpha
            plt.plot(episodes, values, color=color, alpha=0.2, label='_nolegend_')
            
            # Add smoothed line
            window_size = max(2, min(len(episodes) // 10, 20))
            if len(episodes) >= window_size:
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                smooth_episodes = episodes[window_size-1:]
                plt.plot(smooth_episodes, smoothed, color=color, linewidth=2, label=label)
        
        metric_name = metric.replace('_', ' ').title()
        plt.title(f'{metric_name} Learning Curves', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        metric_path = output_path.replace('.png', f'_{metric}.png')
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{metric_name} learning curve saved to {metric_path}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Add automatic timestamp to directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch runs and sweep config
    print(f"Fetching runs for sweep {args.sweep_id}...")
    runs, sweep_config = fetch_sweep_runs(args.entity, args.project, args.sweep_id)
    
    if not runs:
        print("No runs found or error fetching sweep data.")
        return
    
    print(f"Found {len(runs)} runs in sweep {args.sweep_id}")
    
    # Determine the sweep type and parameters to include
    sweep_type = args.table_type.lower() if args.table_type.lower() != 'all' else None
    key_params = get_param_keys_by_type(args.table_type, sweep_config)
    
    print(f"Using parameters: {key_params}")
    
    # Create parameter table
    print("Generating parameter comparison table...")
    param_table, param_df = create_parameter_table(runs, key_params, top_k=args.top_k)
    param_table_file = os.path.join(output_dir, f"{args.table_type}_params_table.tex")
    with open(param_table_file, "w") as f:
        f.write(param_table)
    
    # Save parameter data as CSV
    param_csv_file = os.path.join(output_dir, f"{args.table_type}_params_table.csv")
    param_df.to_csv(param_csv_file, index=False)
    
    # Create performance summary
    print("Generating performance summary tables...")
    perf_summary, best_runs_table, summary_df, best_runs = create_performance_summary(runs, args.table_type)
    
    perf_summary_file = os.path.join(output_dir, f"{args.table_type}_perf_summary.tex")
    with open(perf_summary_file, "w") as f:
        f.write(perf_summary)
    
    best_runs_file = os.path.join(output_dir, f"{args.table_type}_best_runs.tex")
    with open(best_runs_file, "w") as f:
        f.write(best_runs_table)
    
    # Save summary data as CSV if available
    if summary_df is not None:
        summary_csv_file = os.path.join(output_dir, f"{args.table_type}_summary.csv")
        summary_df.to_csv(summary_csv_file)
    
    # Create performance profile plot
    print("Generating performance profile plot...")
    profile_plot_file = os.path.join(output_dir, f"{args.table_type}_perf_profile.png")
    create_performance_profile(runs, profile_plot_file)
    
    # Create learning curves
    print("Generating learning curves...")
    learning_curves_file = os.path.join(output_dir, f"{args.table_type}_learning_curves.png")
    create_learning_curves(runs, learning_curves_file)
    
    # Generate specialized visualizations based on sweep type
    if sweep_type == 'feature':
        print("Generating feature importance visualization...")
        feature_plot_file = os.path.join(output_dir, f"{args.table_type}_feature_importance.png")
        create_feature_importance_plot(runs, feature_plot_file)
    
    elif sweep_type == 'model':
        print("Generating model architecture comparison...")
        model_plot_file = os.path.join(output_dir, f"{args.table_type}_model_comparison.png")
        create_model_architecture_comparison(runs, model_plot_file)
    
    elif sweep_type == 'reward':
        print("Generating reward strategy comparison...")
        reward_plot_file = os.path.join(output_dir, f"{args.table_type}_reward_comparison.png")
        create_reward_comparison_plot(runs, reward_plot_file)
    
    elif sweep_type == 'env':
        print("Generating environment parameter analysis...")
        env_plot_file = os.path.join(output_dir, f"{args.table_type}_env_parameter_analysis.png")
        create_environment_parameter_analysis(runs, env_plot_file)
    
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
            'param_csv': param_csv_file,
            'perf_summary': perf_summary_file,
            'best_runs': best_runs_file,
            'perf_profile': profile_plot_file,
            'learning_curves': learning_curves_file
        }
    }
    
    if best_runs:
        sweep_runs_info['best_runs'] = best_runs
    
    sweep_info_file = os.path.join(output_dir, f"{args.table_type}_sweep_info.yaml")
    with open(sweep_info_file, "w") as f:
        yaml.dump(sweep_runs_info, f, default_flow_style=False)
    
    # Create an index.html file for easy navigation
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{args.table_type.capitalize()} Sweep Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
        .plot {{ margin-bottom: 20px; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{args.table_type.capitalize()} Sweep Results</h1>
    <p>Sweep ID: {args.sweep_id}</p>
    <p>Project: {args.project}</p>
    <p>Generated on: {timestamp}</p>
    
    <h2>Best Runs</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Best Run</th>
            <th>Value</th>
        </tr>
"""
    
    # Add rows for best runs if available
    if best_runs:
        for metric, run_name in best_runs.items():
            value = next((run.summary.get(metric, 'N/A') for run in runs if run.name == run_name), 'N/A')
            html_content += f"""
        <tr>
            <td>{metric.replace('validation_', '')}</td>
            <td>{run_name}</td>
            <td>{value:.2f if isinstance(value, (int, float)) else value}</td>
        </tr>"""
    
    html_content += """
    </table>
    
    <h2>Visualizations</h2>
    <div class="plot-container">
"""
    
    # Add plots based on what was generated
    plot_files = [
        ('Performance Profile', f"{args.table_type}_perf_profile.png"),
        ('Learning Curves (Weighted Sum)', f"{args.table_type}_learning_curves_weighted_sum.png"),
        ('Learning Curves (Reward)', f"{args.table_type}_learning_curves_reward.png"),
        ('Learning Curves (Loss)', f"{args.table_type}_learning_curves_loss.png")
    ]
    
    # Add sweep-specific plots
    if sweep_type == 'feature':
        plot_files.extend([
            ('Feature Importance', f"{args.table_type}_feature_importance.png"),
            ('Feature Correlation', f"{args.table_type}_feature_importance_correlation.png")
        ])
    elif sweep_type == 'model':
        plot_files.extend([
            ('Model Parameter Importance', f"{args.table_type}_model_comparison.png")
        ])
    elif sweep_type == 'reward':
        plot_files.extend([
            ('Reward Strategy Comparison', f"{args.table_type}_reward_comparison.png"),
            ('Reward Effect on Weighted Sum', f"{args.table_type}_reward_comparison_weighted_sum.png")
        ])
    elif sweep_type == 'env':
        plot_files.extend([
            ('Environment Parameter Analysis', f"{args.table_type}_env_parameter_analysis.png"),
            ('Problem Size Analysis', f"{args.table_type}_env_parameter_analysis_problem_size.png"),
            ('Weight Range Analysis', f"{args.table_type}_env_parameter_analysis_weight_range.png")
        ])
    
    # Add plot divs to HTML
    for title, filename in plot_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            rel_path = os.path.basename(filepath)
            html_content += f"""
        <div class="plot">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>"""
    
    html_content += """
    </div>
    
    <h2>Generated Files</h2>
    <ul>
"""
    
    # Add links to all generated files
    for file in os.listdir(output_dir):
        if file.endswith('.tex') or file.endswith('.csv') or file.endswith('.yaml'):
            html_content += f"""
        <li><a href="{file}">{file}</a></li>"""
    
    html_content += """
    </ul>
    
    <p>Generated by the L2D-weighted project's table generator.</p>
</body>
</html>
"""
    
    # Save the HTML file
    html_file = os.path.join(output_dir, "index.html")
    with open(html_file, "w") as f:
        f.write(html_content)
    
    print(f"\nAll outputs generated successfully in {output_dir}")
    print(f"Parameter table: {param_table_file}")
    print(f"Performance summary: {perf_summary_file}")
    print(f"Best runs table: {best_runs_file}")
    print(f"Performance profile plot: {profile_plot_file}")
    print(f"HTML overview: {html_file}")

if __name__ == "__main__":
    main()