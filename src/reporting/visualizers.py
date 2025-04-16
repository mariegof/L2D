"""
Visualization module for creating plots and charts from WandB data.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import traceback

class SweepVisualizer:
    """Create visualizations of sweep results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_run_label(self, run: Dict[str, Any], sweep_type: Optional[str] = None) -> str:
        """
        Create an informative label for a run based on its configuration and the sweep type.
        
        Args:
            run: Run dictionary with config and summary
            sweep_type: Type of sweep (environment, feature, reward, model) if known
            
        Returns:
            String label for the run
        """
        config = run.get('config', {})
        
        # If we don't know the sweep type, try to infer it
        if sweep_type is None:
            if 'reward_strategy' in config and len(set(k for k in config if k != 'reward_strategy')) <= 3:
                sweep_type = 'reward'
            elif 'feature_set' in config and len(set(k for k in config if k != 'feature_set')) <= 3:
                sweep_type = 'feature'
            elif 'hidden_dim' in config and 'lr' in config and 'num_layers' in config:
                sweep_type = 'model'
            elif 'num_envs' in config and 'rewardscale' in config:
                sweep_type = 'environment'
            else:
                sweep_type = 'unknown'
        
        # Create tailored labels based on sweep type
        if sweep_type and sweep_type.lower() in ['reward', 'rewards']:
            # For reward sweeps, just show the strategy name
            if 'reward_strategy' in config:
                return str(config['reward_strategy'])
        
        elif sweep_type and sweep_type.lower() in ['feature', 'features']:
            # For feature sweeps, use a compact representation of features
            if 'feature_set' in config:
                features = config['feature_set']
                if isinstance(features, list):
                    if len(features) <= 3:
                        # For small feature sets, show short names for each
                        feature_names = [f.split('_')[0] for f in features]
                        return ','.join(feature_names)
                    else:
                        # For larger feature sets, just show the count
                        return f"{len(features)} features"
                else:
                    return str(features)
        
        elif sweep_type and sweep_type.lower() in ['model', 'models']:
            # For model sweeps, combine the most important hyperparameters
            parts = []
            
            # Always include hidden_dim if available
            if 'hidden_dim' in config:
                parts.append(f"h={config['hidden_dim']}")
            
            # Include learning rate with scientific notation
            if 'lr' in config:
                lr = config['lr']
                parts.append(f"lr={float(lr):.1e}")
                
            # Include layers if available (often critical)
            if 'num_layers' in config:
                parts.append(f"L={config['num_layers']}")
                
            # For PPO-specific parameters, include k_epochs if it varies
            if 'k_epochs' in config:
                parts.append(f"k={config['k_epochs']}")
                
            # Join with commas for a compact representation
            if parts:
                return ", ".join(parts)
        
        elif sweep_type and sweep_type.lower() in ['environment', 'env']:
            # For environment sweeps, focus on key parameters
            parts = []
            
            # Problem dimensions if they vary
            if 'n_j' in config and 'n_m' in config:
                parts.append(f"{config['n_j']}×{config['n_m']}")
            
            # Number of environments is often important
            if 'num_envs' in config:
                parts.append(f"envs={config['num_envs']}")
            
            # Reward scale often has significant impact
            if 'rewardscale' in config:
                rs = config['rewardscale']
                # Round to 2 decimal places if it's small
                if rs < 0.1:
                    parts.append(f"rs={rs:.2f}")
                else:
                    parts.append(f"rs={rs:.1f}")
                    
            # Join with commas for a compact representation
            if parts:
                return ", ".join(parts)
        
        # Fall back to run name or index for unknown sweep types
        return run.get('name', f"Run {id(run) % 1000}")  # Use part of object id as a unique identifier
    
    def create_performance_plot(self, runs: List[Dict[str, Any]], output_path: str, sweep_type: Optional[str] = None) -> bool:
        """
        Create a performance plot visualization comparing run results using best validation metrics.
        
        Args:
            runs: List of run dictionaries with config and summary data
            output_path: Path to save the visualization
            sweep_type: Type of sweep (environment, feature, reward, model)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have runs with valid data
            if not runs:
                print("No runs to visualize")
                return False
            
            # Define metrics to compare - using best_ prefix
            metrics = [
                ('best_validation_win_rate', 'Win (%)'),
                ('best_validation_win_vs_spt', 'Win vs SPT (%)'),
                ('best_validation_win_vs_wspt', 'Win vs WSPT (%)'),
                ('best_validation_win_vs_srpt', 'Win vs SRPT (%)'),
                ('best_validation_weighted_sum', 'Weighted Sum')
            ]
            
            # Verify at least one metric exists in the runs
            valid_metrics = []
            for metric_key, metric_label in metrics:
                # Check if metric exists in any run's summary
                has_metric = [metric_key in run.get('summary', {}) for run in runs]
                if any(has_metric):
                    valid_metrics.append((metric_key, metric_label))
            
            if not valid_metrics:
                print("No valid best validation metrics found in run data")
                return False
                    
            # We'll add an extra subplot for win rate comparison
            num_plots = len(valid_metrics) + 2  # +1 for weighted sum comparison, +1 for win rate comparison
            
            # Setup the figure with all subplots
            fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6*num_plots))
            if num_plots == 1:
                axs = [axs]  # Make iterable for single subplot
            
            # For each metric, we'll sort runs differently
            for i, (metric_key, metric_label) in enumerate(valid_metrics):
                ax = axs[i]
                
                # Sort runs based on this specific metric
                if 'best_validation_weighted_sum' in metric_key:
                    # Lower is better for weighted sum
                    sorted_runs = sorted(
                        runs, 
                        key=lambda r: r.get('summary', {}).get(metric_key, float('inf'))
                    )
                else:
                    # Higher is better for win rates
                    sorted_runs = sorted(
                        runs, 
                        key=lambda r: r.get('summary', {}).get(metric_key, 0),
                        reverse=True
                    )
                
                # Generate colors for each run
                cmap = cm.get_cmap('viridis', min(20, len(sorted_runs)))
                colors = [cmap(i/len(sorted_runs)) for i in range(len(sorted_runs))]
                
                # Extract values, handling missing data
                values = []
                labels = []
                for j, run in enumerate(sorted_runs):
                    val = run.get('summary', {}).get(metric_key)
                    if val is not None:
                        values.append(val)
                        
                        # Create label using our adaptive function with explicit sweep type
                        label = self.create_run_label(run, sweep_type)
                        labels.append(label)
                
                # Skip if no valid values
                if not values:
                    ax.text(0.5, 0.5, f"No data for {metric_label}", 
                        ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot
                x = range(len(values))
                bars = ax.bar(x, values, color=colors[:len(values)])
                
                # Add labels
                ax.set_xticks(x)
                if len(labels) <= 10:
                    # Show all labels if not too many
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                else:
                    # Show every Nth label for large datasets
                    n = max(1, len(labels) // 10)
                    visible_indices = range(0, len(labels), n)
                    visible_labels = [labels[i] if i in visible_indices else '' for i in range(len(labels))]
                    ax.set_xticklabels(visible_labels, rotation=45, ha='right')
                
                # Set titles and add grid
                ax.set_title(f"{metric_label} by Configuration (Sorted)")
                ax.set_ylabel(metric_label)
                ax.grid(axis='y', alpha=0.3)
                
                # Find and highlight best performing run
                if values:
                    # Since we've sorted the runs for this metric, the best is at index 0
                    best_idx = 0
                    best_val = values[best_idx]
                    
                    # Highlight the best bar
                    bars[best_idx].set_color('green')
                    bars[best_idx].set_alpha(0.7)
                    
                    # Add text indicating the best value
                    if 'best_validation_weighted_sum' in metric_key:
                        # Format as integer for weighted sum
                        ax.text(best_idx, best_val, f"Best: {int(best_val)}", 
                            ha='center', va='bottom', fontweight='bold')
                    else:
                        # Format with 1 decimal place for percentages
                        ax.text(best_idx, best_val, f"Best: {best_val:.1f}%", 
                            ha='center', va='bottom', fontweight='bold')
                    
                    # Add value labels to all bars
                    for j, v in enumerate(values):
                        if j != best_idx:  # Skip the best one since we already labeled it
                            if 'best_validation_weighted_sum' in metric_key:
                                ax.text(j, v, f"{int(v)}", ha='center', va='bottom')
                            else:
                                ax.text(j, v, f"{v:.1f}%", ha='center', va='bottom')
            
            # Create the weighted sum comparison plot (second to last subplot)
            ax_weighted_sum = axs[-2]  # Second to last subplot
            
            # Find the best L2D run and baseline data from all available runs
            methods = {
                'L2D': None,
                'SPT': None,
                'WSPT': None,
                'SRPT': None
            }
            
            # Also track win rates from the best run
            win_rates = {
                'L2D': None,
                'SPT': None,
                'WSPT': None,
                'SRPT': None
            }
            
            # First pass: Find best L2D run based on best_validation_weighted_sum
            best_l2d_run = None
            best_weighted_sum = float('inf')
            
            for run in runs:
                summary = run.get('summary', {})
                if 'best_validation_weighted_sum' in summary:
                    l2d_ws = summary['best_validation_weighted_sum']
                    if l2d_ws < best_weighted_sum:
                        best_weighted_sum = l2d_ws
                        best_l2d_run = run
            
            if best_l2d_run:
                methods['L2D'] = best_weighted_sum
                
                # Check if this run contains baseline data
                summary = best_l2d_run.get('summary', {})
                for baseline in ['spt', 'wspt', 'srpt']:
                    baseline_key = f"validation_baseline_{baseline}_weighted_sum"
                    if baseline_key in summary and summary[baseline_key] is not None:
                        methods[baseline.upper()] = summary[baseline_key]
                    
                # Use the best_validation_ prefixed metrics that are already being logged
                win_rates['L2D'] = summary.get('best_win_rate', 0)
                win_rates['SPT'] = summary.get('best_validation_win_rate_spt', 0)
                win_rates['WSPT'] = summary.get('best_validation_win_rate_wspt', 0)
                win_rates['SRPT'] = summary.get('best_validation_win_rate_srpt', 0)
            
            # Second pass: If we're missing baseline data, check all runs
            if best_l2d_run and (methods['SPT'] is None or methods['WSPT'] is None or methods['SRPT'] is None):
                for run in runs:
                    summary = run.get('summary', {})
                    for baseline in ['spt', 'wspt', 'srpt']:
                        baseline_key = f"validation_baseline_{baseline}_weighted_sum"
                        baseline_upper = baseline.upper()
                        
                        # Only fill in missing values
                        if methods[baseline_upper] is None and baseline_key in summary and summary[baseline_key] is not None:
                            methods[baseline_upper] = summary[baseline_key]
                            
                        # Also check for comparison metrics which might contain this info
                        if methods[baseline_upper] is None:
                            metric_name = f"best_validation_improvement_over_{baseline}"
                            if metric_name in summary and 'best_validation_weighted_sum' in summary:
                                # Calculate baseline value from improvement percentage
                                improvement = summary[metric_name] / 100  # Convert from percentage
                                l2d_ws = summary['best_validation_weighted_sum']
                                # Only if the improvement is valid (between 0 and 100%)
                                if 0 <= improvement < 1:
                                    methods[baseline_upper] = l2d_ws / (1 - improvement)
            
            # Filter out methods with no weighted sum values
            valid_methods = {k: v for k, v in methods.items() if v is not None}
            
            if len(valid_methods) > 1:  # Need at least L2D plus one baseline
                # Create bar chart comparing methods
                method_names = list(valid_methods.keys())
                method_values = list(valid_methods.values())
                
                # Create color map for methods - highlight L2D
                method_colors = ['green' if m == 'L2D' else 'skyblue' for m in method_names]
                
                # Plot
                x = range(len(method_names))
                bars = ax_weighted_sum.bar(x, method_values, color=method_colors)
                
                # Add labels
                ax_weighted_sum.set_xticks(x)
                ax_weighted_sum.set_xticklabels(method_names, rotation=0)
                
                # Add value labels
                for i, v in enumerate(method_values):
                    ax_weighted_sum.text(i, v, f"{int(v)}", ha='center', va='bottom', 
                                        fontweight='bold' if method_names[i] == 'L2D' else 'normal')
                
                # Set titles and add grid
                ax_weighted_sum.set_title("Absolute Weighted Sum Comparison Between Methods")
                ax_weighted_sum.set_ylabel("Weighted Sum")
                ax_weighted_sum.grid(axis='y', alpha=0.3)
                
                # Add improvement percentages relative to L2D
                if 'L2D' in valid_methods:
                    l2d_value = valid_methods['L2D']
                    for i, method in enumerate(method_names):
                        if method != 'L2D':
                            method_value = valid_methods[method]
                            improvement = (method_value - l2d_value) / method_value * 100
                            ax_weighted_sum.text(i, method_value/2, f"↓{improvement:.1f}%", 
                                            ha='center', va='center', color='white', fontweight='bold')
            else:
                # Create a more informative error message
                available_data = []
                for method, value in methods.items():
                    available_data.append(f"{method}: {'Available' if value is not None else 'Missing'}")
                
                error_msg = "Insufficient data for method comparison\n" + "\n".join(available_data)
                
                # Display detailed message in the plot
                ax_weighted_sum.text(0.5, 0.5, error_msg, 
                                ha='center', va='center', transform=ax_weighted_sum.transAxes,
                                fontsize=10, color='red')
            
            # Create win rate comparison chart (last subplot)
            ax_win_rates = axs[-1]  # Last subplot
            
            # Filter out methods with no win rate values or zero values
            valid_win_rates = {k: v for k, v in win_rates.items() if v is not None and v > 0}
            
            if len(valid_win_rates) > 1:  # Need at least two methods
                method_names = list(valid_win_rates.keys())
                win_rate_values = list(valid_win_rates.values())
                
                # Create the win rate chart
                self.create_win_rate_chart(ax_win_rates, method_names, win_rate_values)
            else:
                # Display a message if no win rate data is available
                ax_win_rates.text(0.5, 0.5, "Insufficient data for win rate comparison", 
                                ha='center', va='center', transform=ax_win_rates.transAxes,
                                fontsize=10, color='red')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Created enhanced performance plot at {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating performance plot: {e}")
            traceback.print_exc()
            return False
                
    def create_win_rate_chart(self, ax, methods, win_rates):
        """
        Create a bar chart comparing win rates of different methods.
        
        Args:
            ax: Matplotlib axis to plot on
            methods: List of method names
            win_rates: List of win rate percentages
            
        Returns:
            The matplotlib bars object for further customization
        """
        # Create color map - highlight L2D
        method_colors = ['green' if m == 'L2D' else 'skyblue' for m in methods]
        
        # Plot win rates
        x = range(len(methods))
        bars = ax.bar(x, win_rates, color=method_colors)
        
        # Add labels
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0)
        
        # Add value labels
        for i, v in enumerate(win_rates):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', 
                fontweight='bold' if methods[i] == 'L2D' else 'normal')
        
        # Set titles and add grid
        ax.set_title("Win Rate Comparison Between Methods")
        ax.set_ylabel("Win Rate (%)")
        ax.grid(axis='y', alpha=0.3)
        
        # Add a note explaining win rate
        ax.text(0.5, -0.15, 
            "Win rate: Percentage of instances where each method achieved the lowest weighted sum", 
            ha='center', va='center', transform=ax.transAxes, fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        return bars            
                
    def create_parameter_impact_plot(self, run_data: pd.DataFrame, parameter: str, 
                                    metric: str, output_path: str) -> bool:
        """
        Create visualization showing impact of a parameter on performance.
        
        Args:
            run_data: DataFrame of processed run data
            parameter: Parameter to analyze (e.g., 'hidden_dim')
            metric: Metric to measure (e.g., 'validation_win_vs_wspt')
            output_path: Path to save the visualization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have data
            if run_data.empty or parameter not in run_data.columns or metric not in run_data.columns:
                print(f"Missing data for parameter {parameter} or metric {metric}")
                return False
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get parameter values and metrics
            param_values = run_data[parameter].values
            metric_values = run_data[metric].values
            
            # Check if parameter is numeric
            is_numeric = pd.api.types.is_numeric_dtype(run_data[parameter])
            
            if is_numeric:
                # Scatter plot for numeric parameters
                scatter = ax.scatter(param_values, metric_values, alpha=0.7, s=80, c=metric_values, 
                                    cmap='viridis')
                
                # Add trend line if enough points
                if len(param_values) > 2:
                    try:
                        z = np.polyfit(param_values, metric_values, 1)
                        p = np.poly1d(z)
                        ax.plot(param_values, p(param_values), "r--", alpha=0.7, 
                               label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
                        ax.legend()
                    except:
                        pass  # Skip trend line if it fails
            else:
                # Bar plot for categorical parameters
                # Convert to strings for cleaner display
                param_str_values = [str(val) for val in param_values]
                
                # Calculate average metric for each parameter value
                param_df = pd.DataFrame({
                    'param': param_str_values,
                    'metric': metric_values
                })
                avg_metrics = param_df.groupby('param')['metric'].mean()
                
                # Sort by metric value
                sorted_indices = avg_metrics.argsort()
                if 'win' in metric.lower():
                    # Higher is better for win rates
                    sorted_indices = sorted_indices[::-1]
                
                # Plot in order
                categories = avg_metrics.index[sorted_indices]
                values = avg_metrics.values[sorted_indices]
                
                # Use bar plot
                bars = ax.bar(range(len(categories)), values, color=cm.viridis(np.linspace(0, 1, len(categories))))
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # Format axis labels
            ax.set_xlabel(parameter.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').replace('validation ', '').title())
            
            # Set title
            ax.set_title(f"Impact of {parameter.replace('_', ' ').title()} on {metric.replace('_', ' ').replace('validation ', '').title()}")
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Add data points
            for i, txt in enumerate(run_data.index):
                if is_numeric:
                    # For numeric params, annotate with index
                    ax.annotate(str(i), (param_values[i], metric_values[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
            # Highlight best point
            best_idx = metric_values.argmax() if 'win' in metric.lower() else metric_values.argmin()
            if is_numeric:
                ax.scatter([param_values[best_idx]], [metric_values[best_idx]], color='red', 
                          s=100, edgecolor='black', zorder=10, label='Best')
                ax.legend()
            else:
                # For categorical, highlight the best bar
                best_category = param_str_values[best_idx]
                for i, category in enumerate(categories):
                    if category == best_category:
                        bars[i].set_color('green')
                        ax.text(i, values[i], f"Best: {values[i]:.2f}", 
                               ha='center', va='bottom', fontsize=10)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Created parameter impact plot at {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating parameter impact plot: {e}")
            traceback.print_exc()
            return False
