"""
Data processing module for transforming WandB run data consistently.
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
import traceback

class SweepDataProcessor:
    """Process data from WandB sweeps for reporting with consistent dictionary access."""
    
    def get_param_keys_by_type(self, sweep_type: str, sweep_config: Dict[str, Any]) -> List[str]:
        """
        Determine which parameters to include in tables based on sweep type.
        
        Args:
            sweep_type: Type of sweep ('environment', 'feature', 'reward', 'model')
            sweep_config: Sweep configuration dictionary
            
        Returns:
            List of parameter keys to include in reports
        """
        # Parameter groups by sweep type
        param_groups = {
            'env': ["n_j", "n_m", "low", "high", "weight_low", "weight_high", 
                    "max_updates", "num_envs", "rewardscale"],
            'environment': ["n_j", "n_m", "low", "high", "weight_low", "weight_high", 
                    "max_updates", "num_envs", "rewardscale"],
            'model': ["num_layers", "hidden_dim", "lr", "gamma", "k_epochs", "eps_clip", 
                    "neighbor_pooling_type", "graph_pool_type", "num_mlp_layers_feature_extract",
                    "num_mlp_layers_actor", "hidden_dim_actor", "num_mlp_layers_critic", 
                    "hidden_dim_critic", "ploss_coef", "vloss_coef", "entloss_coef"],
            'feature': ["feature_set"],
            'features': ["feature_set"],
            'reward': ["reward_strategy"],
            'rewards': ["reward_strategy"]
        }
        
        # If sweep_type is explicitly provided, return those params
        sweep_type_lower = sweep_type.lower()
        if sweep_type_lower in param_groups:
            return param_groups[sweep_type_lower]
        
        # Otherwise, try to infer from sweep config
        if not sweep_config:
            print("Warning: Empty sweep config, returning empty parameter list")
            return []
            
        # Get available parameters from config
        available_params: Set[str] = set()
        
        # Extract from 'parameters' section if available
        if 'parameters' in sweep_config:
            params_section = sweep_config.get('parameters', {})
            if isinstance(params_section, dict):
                available_params = set(params_section.keys())
        else:
            # Try to get parameters directly from config
            available_params = set(sweep_config.keys())
        
        # Count how many parameters from each group are in the sweep config
        param_counts = {
            group: len(set(params) & available_params) 
            for group, params in param_groups.items()
        }
        
        # If no parameters are found, return all available parameters
        if max(param_counts.values(), default=0) == 0:
            print("Warning: Could not determine parameter group, using all parameters")
            return list(available_params)
        
        # Return the parameters for the group with the most matches
        best_group = max(param_counts.items(), key=lambda x: x[1])[0]
        return param_groups[best_group]
    
    def extract_run_data(self, runs: List[Dict[str, Any]], key_params: List[str]) -> pd.DataFrame:
        """
        Extract parameter and metric data from WandB runs into a DataFrame.
        
        Args:
            runs: List of run dictionaries with config and summary
            key_params: List of parameters to extract from config
            
        Returns:
            DataFrame with standardized run data
        """
        if not runs:
            print("Warning: No runs provided to extract_run_data")
            return pd.DataFrame()
        
        # Collect data from runs
        data = []
        for run in runs:
            try:
                # Get config and summary as dictionaries
                config = run.get('config', {})
                summary = run.get('summary', {})
                
                # Skip runs with missing data
                if not config or not summary:
                    print(f"Skipping run {run.get('name', 'unknown')} - missing config or summary")
                    continue
                    
                # Extract run data
                run_data = {
                    'run_name': run.get('name', 'unknown'),
                    'run_id': run.get('id', 'unknown'),
                    'url': run.get('url', '')
                }
                
                # Extract parameters from config
                for param in key_params:
                    if param in config:
                        # Handle special formatting for feature_set
                        if param == 'feature_set' and isinstance(config[param], list):
                            run_data[param] = ', '.join(config[param])
                        else:
                            run_data[param] = config[param]
                    else:
                        run_data[param] = "N/A"
                
                # Extract key metrics
                metric_keys = [
                    'validation_weighted_sum', 'validation_win_vs_wspt', 
                    'validation_win_vs_spt', 'validation_win_rate',
                    'validation_improvement_over_wspt', 'validation_improvement_over_spt'
                ]
                
                for metric in metric_keys:
                    if metric in summary:
                        value = summary[metric]
                        # Convert string representations of numbers to actual numbers if needed
                        if isinstance(value, str):
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except:
                                pass  # Keep as string if conversion fails
                        run_data[metric] = value
                    else:
                        # Use appropriate default values based on metric
                        if 'weighted_sum' in metric:
                            run_data[metric] = float('inf')  # Lower is better
                        else:
                            run_data[metric] = 0  # Higher is better
                
                data.append(run_data)
            except Exception as e:
                print(f"Error processing run data: {e}")
                print(f"Run: {run.get('name', 'unknown')}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        # Handle list parameters consistently
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
        
        return df
    
    def get_best_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Find the best run for each metric.
        
        Args:
            runs: List of run dictionaries with summary metrics
            
        Returns:
            Dictionary mapping metrics to best run names
        """
        metrics = [
            'validation_weighted_sum', 'validation_win_vs_wspt', 
            'validation_win_vs_spt', 'validation_win_rate',
            'validation_improvement_over_wspt', 'validation_improvement_over_spt'
        ]
        
        best_runs = {}
        for metric in metrics:
            try:
                # Get valid runs for this metric
                valid_runs = []
                for run in runs:
                    summary = run.get('summary', {})
                    name = run.get('name', "Unknown")
                    
                    # Check if metric exists and add to valid runs
                    value = summary.get(metric)
                    if value is not None:
                        # Convert strings to numbers if needed
                        if isinstance(value, str):
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except:
                                continue  # Skip invalid values
                        valid_runs.append((name, value))
                
                if not valid_runs:
                    continue
                    
                # Find best run based on metric
                if metric == 'validation_weighted_sum':
                    # Lower is better for weighted sum
                    best_run = min(valid_runs, key=lambda x: x[1])
                else:
                    # Higher is better for win rates
                    best_run = max(valid_runs, key=lambda x: x[1])
                
                best_runs[metric] = best_run[0]
            except Exception as e:
                print(f"Error finding best run for {metric}: {e}")
                continue
        
        return best_runs
    
    def get_performance_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for performance metrics.
        
        Args:
            runs: List of run dictionaries with summary metrics
            
        Returns:
            Dictionary mapping metrics to statistics
        """
        metrics = [
            'validation_weighted_sum', 'validation_win_vs_wspt', 
            'validation_win_vs_spt', 'validation_win_rate',
            'validation_improvement_over_wspt', 'validation_improvement_over_spt'
        ]
        
        metrics_data = {}
        for metric in metrics:
            try:
                values = []
                for run in runs:
                    summary = run.get('summary', {})
                    
                    # Get metric value
                    value = summary.get(metric)
                    if value is not None:
                        # Convert strings to numbers if needed
                        if isinstance(value, str):
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                                values.append(value)
                            except:
                                pass  # Skip invalid values
                        else:
                            values.append(value)
                
                if not values:
                    continue
                    
                metrics_data[metric] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values))
                }
            except Exception as e:
                print(f"Error calculating metrics for {metric}: {e}")
                traceback.print_exc()
                continue
        
        return metrics_data
    
    def get_best_config(self, runs: List[Dict[str, Any]], 
                      primary_metric: str = 'validation_win_vs_wspt') -> Dict[str, Any]:
        """
        Extract the configuration from the best performing run.
        
        Args:
            runs: List of run dictionaries
            primary_metric: Metric to use for determining the best run
            
        Returns:
            Configuration dictionary from best run
        """
        if not runs:
            return {}
            
        try:
            # Find runs with the primary metric
            valid_runs = []
            for run in runs:
                summary = run.get('summary', {})
                value = summary.get(primary_metric)
                
                if value is not None:
                    # Convert to number if needed
                    if isinstance(value, str):
                        try:
                            value = float(value) if '.' in value else int(value)
                        except:
                            continue
                    valid_runs.append((run, value))
            
            if not valid_runs:
                print(f"No runs found with metric {primary_metric}")
                return {}
                
            # Sort runs by metric (higher is better, except for weighted_sum)
            if 'weighted_sum' in primary_metric:
                sorted_runs = sorted(valid_runs, key=lambda x: x[1])  # Lower is better
            else:
                sorted_runs = sorted(valid_runs, key=lambda x: x[1], reverse=True)  # Higher is better
                
            # Get best run
            best_run = sorted_runs[0][0]
            best_config = {}
            
            # Copy all config values directly (don't nest them under 'config')
            if 'config' in best_run and isinstance(best_run['config'], dict):
                best_config.update(best_run['config'])
            
            # Add metadata
            best_config['__best_run_name__'] = best_run.get('name', 'unknown')
            best_config['__best_run_id__'] = best_run.get('id', 'unknown')
            best_config['__best_metric__'] = primary_metric
            best_config['__best_value__'] = sorted_runs[0][1]
            
            return best_config
            
        except Exception as e:
            print(f"Error getting best config: {e}")
            traceback.print_exc()
            return {}