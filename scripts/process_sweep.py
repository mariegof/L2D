#!/usr/bin/env python
"""
Process W&B sweep results and generate reports, tables, and visualizations.

This script fetches data from a completed WandB sweep, processes it, and
generates LaTeX tables and plots for analysis.
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import reporting modules
try:
    from src.reporting import (
        WandBFetcher, 
        SweepDataProcessor, 
        TableGenerator, 
        PyLatexFormatter,
        SweepVisualizer,
        ReportWriter
    )
except ImportError:
    # Add parent directory to path if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.reporting import (
        WandBFetcher, 
        SweepDataProcessor, 
        TableGenerator, 
        PyLatexFormatter,
        SweepVisualizer,
        ReportWriter
    )

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process WandB sweep data and generate reports')
    parser.add_argument('--project', type=str, required=True, 
                      help='WandB project name')
    parser.add_argument('--sweep_id', type=str, required=True, 
                      help='WandB sweep ID')
    parser.add_argument('--entity', type=str, default=None, 
                      help='WandB entity (username/organization)')
    parser.add_argument('--output', type=str, required=True, 
                      help='Root output directory for reports')
    parser.add_argument('--sweep_type', type=str, default=None, 
                      choices=['environment', 'env', 'feature', 'reward', 'model'],
                      help='Type of sweep (environment, feature, reward, model)')
    parser.add_argument('--top_k', type=int, default=5, 
                      help='Number of top runs to include in summary')
    parser.add_argument('--primary_metric', type=str, default='validation_win_vs_wspt', 
                      help='Primary metric for determining best run')
    parser.add_argument('--weight_type', type=str, default='uniform', 
                      choices=['uniform', 'variable'],
                      help='Type of job weight setting used (uniform or variable)')
    return parser.parse_args()

def main():
    """Main function to process sweep data and generate reports."""
    args = parse_args()
    
    # Set up writer for file output - create only artifacts directories
    writer = ReportWriter(args.output, create_all=False)
    
    # Fetch data from WandB
    fetcher = WandBFetcher()
    runs, sweep_config = fetcher.fetch_sweep_runs(
        project=args.project,
        sweep_id=args.sweep_id,
        entity=args.entity
    )
    
    # Check if we got valid data
    if not runs:
        print("Error: No valid run data fetched.")
        return 1
    
    # Process data
    processor = SweepDataProcessor()
    param_keys = processor.get_param_keys_by_type(args.sweep_type, sweep_config)
    print(f"Using parameters: {param_keys}")
    
    # Get best runs and performance metrics
    best_runs = processor.get_best_runs(runs)
    metrics_data = processor.get_performance_metrics(runs)
    
    # Create metrics summary table
    if metrics_data:
        metrics_caption = 'Summary Statistics of Performance Metrics'
        metrics_label = 'tab:metrics'
        formatter = PyLatexFormatter()
        metrics_table = formatter.format_summary_table(metrics_data, metrics_caption, metrics_label)
        writer.write_latex_table(metrics_table, "metrics_table")
        print(f"Created metrics summary table")
    
    # Generate tables
    generator = TableGenerator()
    formatter = PyLatexFormatter()
    
    # Create visualizer (used across all sweep types)
    visualizer = SweepVisualizer(writer.get_directory('plots'))
    
    # Generate appropriate table based on sweep type
    if args.sweep_type and args.sweep_type.lower() in ['environment', 'env']:
        df = generator.create_environment_table(runs)
        caption = 'Comparison of Environment Configurations'
        label = 'tab:environment'
        env_table = formatter.format_dataframe_to_latex(df, caption, label)
        writer.write_latex_table(env_table, "environment_table")
        
        # Create performance plot with explicit sweep type
        visualizer.create_performance_plot(
            runs, 
            writer.get_path("performance_plot.png", "plots"),
            sweep_type=args.sweep_type
        )
        
        # Create parameter impact plots
        for param in ['rewardscale']:
            if param in df.columns:
                visualizer.create_parameter_impact_plot(
                    df, param, args.primary_metric, 
                    writer.get_path(f"{param}_impact.png", "plots")
                )
        
    elif args.sweep_type and args.sweep_type.lower() in ['feature', 'features']:
        df = generator.create_feature_table(runs)
        caption = 'Comparison of Feature Sets'
        label = 'tab:features'
        feature_table = formatter.format_dataframe_to_latex(df, caption, label)
        writer.write_latex_table(feature_table, "feature_table")
        
        # Create performance plot with explicit sweep type
        visualizer.create_performance_plot(
            runs, 
            writer.get_path("performance_plot.png", "plots"),
            sweep_type=args.sweep_type
        )
        
    elif args.sweep_type and args.sweep_type.lower() in ['reward', 'rewards']:
        df = generator.create_reward_table(runs)
        caption = 'Comparison of Reward Strategies'
        label = 'tab:rewards'
        reward_table = formatter.format_dataframe_to_latex(df, caption, label)
        writer.write_latex_table(reward_table, "reward_table")
        
        # Create performance plot with explicit sweep type
        visualizer.create_performance_plot(
            runs, 
            writer.get_path("performance_plot.png", "plots"),
            sweep_type=args.sweep_type
        )
        
    elif args.sweep_type and args.sweep_type.lower() in ['model', 'models']:
        df = generator.create_model_table(runs)
        caption = 'Comparison of Model Configurations'
        label = 'tab:models'
        model_table = formatter.format_dataframe_to_latex(df, caption, label)
        writer.write_latex_table(model_table, "model_table")
        
        # Create performance plot with explicit sweep type
        visualizer.create_performance_plot(
            runs, 
            writer.get_path("performance_plot.png", "plots"),
            sweep_type=args.sweep_type
        )
        
    # Create summary table
    summary_df = generator.create_summary_table(runs, args.sweep_type or 'unknown', args.top_k)
    caption = f'Top {args.top_k} Configurations'
    label = 'tab:summary'
    summary_table = formatter.format_dataframe_to_latex(summary_df, caption, label)
    writer.write_latex_table(summary_table, "summary_table")
    
    print(f"Sweep processing complete. Results available in {args.output}/artifacts")
    print(f"LaTeX tables: {writer.get_directory('latex')}")
    print(f"Plots: {writer.get_directory('plots')}")
    
    # List specific tables and plots created
    latex_files = writer.get_files_by_extension('tex', 'latex')
    plot_files = writer.get_files_by_extension('png', 'plots')
    
    print(f"\nGenerated {len(latex_files)} LaTeX tables:")
    for file in latex_files:
        print(f"  - {file}")
        
    print(f"\nGenerated {len(plot_files)} plots:")
    for file in plot_files:
        print(f"  - {file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())