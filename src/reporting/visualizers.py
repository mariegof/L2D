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
    
    def create_performance_profile(self, runs: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Create a performance profile visualization comparing run results.
        
        Args:
            runs: List of run dictionaries with config and summary data
            output_path: Path to save the visualization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have runs with valid data
            if not runs:
                print("No runs to visualize")
                return False
            
            # Define metrics to compare
            metrics = [
                ('validation_win_vs_wspt', 'Win vs WSPT (%)'),
                ('validation_win_vs_spt', 'Win vs SPT (%)'),
                ('validation_weighted_sum', 'Weighted Sum')
            ]
            
            # Verify at least one metric exists in the runs
            valid_metrics = []
            for metric_key, _ in metrics:
                # Check if metric exists in any run's summary
                has_metric = [metric_key in run.get('summary', {}) for run in runs]
                if any(has_metric):
                    valid_metrics.append((metric_key, _))
            
            if not valid_metrics:
                print("No valid metrics found in run data")
                return False
                
            # Setup the figure (one for each valid metric)
            fig, axs = plt.subplots(len(valid_metrics), 1, figsize=(10, 6*len(valid_metrics)))
            if len(valid_metrics) == 1:
                axs = [axs]  # Make iterable for single subplot
                
            # Sort runs by best performance on first metric
            sorted_runs = sorted(
                runs, 
                key=lambda r: r.get('summary', {}).get(valid_metrics[0][0], 0), 
                reverse=True if 'win' in valid_metrics[0][0].lower() else False
            )
            
            # Generate colors for each run
            cmap = cm.get_cmap('viridis', min(20, len(sorted_runs)))
            colors = [cmap(i/len(sorted_runs)) for i in range(len(sorted_runs))]
            
            # Plot each metric
            for i, (metric_key, metric_label) in enumerate(valid_metrics):
                ax = axs[i]
                
                # Extract values, handling missing data
                values = []
                labels = []
                for j, run in enumerate(sorted_runs):
                    val = run.get('summary', {}).get(metric_key)
                    if val is not None:
                        values.append(val)
                        
                        # Create label from config if available
                        config = run.get('config', {})
                        if 'n_j' in config and 'n_m' in config:
                            label = f"{config['n_j']}×{config['n_m']}"
                        elif 'feature_set' in config:
                            # Join feature set if it's a list
                            features = config['feature_set']
                            if isinstance(features, list):
                                label = ', '.join(features)
                            else:
                                label = str(features)
                        elif 'reward_strategy' in config:
                            label = str(config['reward_strategy'])
                        elif 'hidden_dim' in config and 'lr' in config:
                            label = f"h={config['hidden_dim']}, lr={config['lr']:.1e}"
                        else:
                            # Default to run name
                            label = run.get('name', f"Run {j}")
                            
                        labels.append(label)
                
                # Skip if no valid values
                if not values:
                    ax.text(0.5, 0.5, f"No data for {metric_label}", 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot
                x = range(len(values))
                ax.bar(x, values, color=colors[:len(values)])
                
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
                ax.set_title(f"{metric_label} by Configuration")
                ax.set_ylabel(metric_label)
                ax.grid(axis='y', alpha=0.3)
                
                # Highlight best performing run
                if values:
                    best_idx = 0  # Already sorted
                    ax.bar(best_idx, values[best_idx], color='green', alpha=0.7)
                    ax.text(best_idx, values[best_idx], f"Best: {values[best_idx]:.2f}", 
                           ha='center', va='bottom')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Created performance profile at {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating performance profile: {e}")
            traceback.print_exc()
            return False
    
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
    
    def create_learning_curve(self, runs: List[Dict[str, Any]], metric: str, 
                             output_path: str) -> bool:
        """
        Create learning curve plots from run histories.
        
        Args:
            runs: List of run dictionaries with history data
            metric: Metric to plot from history
            output_path: Path to save the visualization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify we have runs with history data
            valid_runs = [run for run in runs if run.get('history') is not None]
            if not valid_runs:
                print("No runs with history data")
                return False
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define color map for runs
            cmap = cm.get_cmap('viridis', min(20, len(valid_runs)))
            
            # Plot each run's learning curve
            for i, run in enumerate(valid_runs):
                history = run.get('history', [])
                if not history or metric not in history[0]:
                    continue
                    
                # Extract metric values and episode numbers
                episodes = []
                values = []
                for entry in history:
                    if metric in entry and 'episode' in entry:
                        episodes.append(entry['episode'])
                        values.append(entry[metric])
                
                if not episodes:
                    continue
                    
                # Plot learning curve
                color = cmap(i/len(valid_runs))
                ax.plot(episodes, values, color=color, alpha=0.7, 
                       label=run.get('name', f"Run {i}"))
                
                # Smooth curve for clearer trend
                if len(episodes) > 10:
                    window_size = max(3, len(episodes) // 20)
                    smoothed = pd.Series(values).rolling(window=window_size).mean().values
                    ax.plot(episodes, smoothed, color=color, linewidth=2)
            
            # Format axis labels
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Set title
            ax.set_title(f"Learning Curves for {metric.replace('_', ' ').title()}")
            
            # Add legend if not too many runs
            if len(valid_runs) <= 10:
                ax.legend(loc='best')
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Created learning curve at {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating learning curve: {e}")
            traceback.print_exc()
            return False