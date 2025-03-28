"""
Table generation module for creating various tables from WandB data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import traceback

class TableGenerator:
    """Generate tables from WandB run data with consistent dictionary access."""
    
    def create_environment_table(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a table for environment sweep comparisons including SRPT metrics."""
        try:
            # Create a DataFrame for the results
            data = []
            for run in runs:
                try:
                    # Extract config and summary
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Skip runs with missing data
                    if not config or not summary:
                        continue
                    
                    run_data = {
                        'Reward Scale': config.get('rewardscale', 'N/A'),
                        'Win Rate (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'vs SPT (%)': summary.get('validation_win_vs_spt', 0),
                        'vs SRPT (%)': summary.get('validation_win_vs_srpt', 0),  # Added SRPT comparison
                        'Weighted Sum': summary.get('validation_weighted_sum', 0)
                    }
                    data.append(run_data)
                except Exception as e:
                    print(f"Error processing run data: {e}")
                    continue
            
            # Convert to DataFrame and sort by win rate
            if data:
                df = pd.DataFrame(data)
                return df.sort_values('Win Rate (%)', ascending=False)
            else:
                print("Warning: No valid run data found for environment table")
                return pd.DataFrame(columns=[
                    'Reward Scale', 'Win Rate (%)', 'vs WSPT (%)', 
                    'vs SPT (%)', 'vs SRPT (%)', 'Weighted Sum'
                ])
        except Exception as e:
            print(f"Error creating environment table: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        
    def create_feature_table(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a table for feature sweep comparisons.
        
        Args:
            runs: List of run dictionaries with config and summary
            
        Returns:
            DataFrame with feature configuration data
        """
        try:
            # Create a DataFrame for the results
            data = []
            for run in runs:
                try:
                    # Extract config and summary
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Skip runs with missing data
                    if not config or not summary:
                        continue
                    
                    # Format the feature set nicely
                    feature_set = config.get('feature_set', [])
                    if isinstance(feature_set, list):
                        feature_set_str = ", ".join(feature_set)
                    else:
                        feature_set_str = str(feature_set)
                        
                    run_data = {
                        'Features': feature_set_str,
                        'Count': len(feature_set) if isinstance(feature_set, list) else 1,
                        'Win Rate (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'vs SPT (%)': summary.get('validation_win_vs_spt', 0),
                        'vs SRPT (%)': summary.get('validation_win_vs_srpt', 0), 
                        'Weighted Sum': summary.get('validation_weighted_sum', 0),
                        'Impr. SRPT (%)': summary.get('validation_improvement_over_srpt', 0)
                    }
                    data.append(run_data)
                except Exception as e:
                    print(f"Error processing run for feature table: {e}")
                    continue
            
            # Convert to DataFrame and sort by win rate
            if data:
                df = pd.DataFrame(data)
                return df.sort_values('Win Rate (%)', ascending=False)
            else:
                print("Warning: No valid run data found for feature table")
                return pd.DataFrame(columns=[
                    'Features', 'Count', 'Win Rate (%)', 'vs WSPT (%)', 'vs SPT (%)', 'vs SRPT (%),''Weighted Sum'
                ])
        except Exception as e:
            print(f"Error creating feature table: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_reward_table(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a table for reward function comparisons.
        
        Args:
            runs: List of run dictionaries with config and summary
            
        Returns:
            DataFrame with reward strategy data including SRPT comparison
        """
        try:
            # Create a DataFrame for the results
            data = []
            for run in runs:
                try:
                    # Extract config and summary
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Skip runs with missing data
                    if not config or not summary:
                        continue
                    
                    run_data = {
                        'Reward Strategy': config.get('reward_strategy', 'default'),
                        'Win Rate (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'vs SPT (%)': summary.get('validation_win_vs_spt', 0),
                        'vs SRPT (%)': summary.get('validation_win_vs_srpt', 0), 
                        'Weighted Sum': summary.get('validation_weighted_sum', 0),
                        'Improv. WSPT (%)': summary.get('validation_improvement_over_wspt', 0),
                    }
                    data.append(run_data)
                except Exception as e:
                    print(f"Error processing run for reward table: {e}")
                    continue
            
            # Convert to DataFrame and sort by win rate against WSPT (our main metric)
            if data:
                df = pd.DataFrame(data)
                return df.sort_values('vs WSPT (%)', ascending=False)
            else:
                print("Warning: No valid run data found for reward table")
                return pd.DataFrame(columns=[
                    'Reward Strategy', 'Win Rate (%)', 'vs WSPT (%)', 
                    'vs SPT (%)', 'vs SRPT (%)', 'Weighted Sum', 'Improv. WSPT (%)'
                ])
        except Exception as e:
            print(f"Error creating reward table: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_model_table(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a table for model/hyperparameter comparisons.
        
        Args:
            runs: List of run dictionaries with config and summary
            
        Returns:
            DataFrame with model configuration data
        """
        try:
            # Create a DataFrame for the results
            data = []
            for run in runs:
                try:
                    # Extract config and summary
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Skip runs with missing data
                    if not config or not summary:
                        continue
                    
                    # Simplify the configuration description to focus on key hyperparameters
                    config_parts = []
                    
                    # Format learning rate in scientific notation
                    lr = config.get('lr')
                    lr_str = f"{float(lr):.1e}" if lr is not None else "N/A"
                    
                    if 'hidden_dim' in config:
                        config_parts.append(f"h={config['hidden_dim']}")
                    if 'num_layers' in config:
                        config_parts.append(f"L={config['num_layers']}")
                    if 'lr' in config:
                        config_parts.append(f"lr={lr_str}")
                        
                    config_str = ", ".join(config_parts)
                    
                    run_data = {
                        'Config': config_str,
                        'Hidden Dim': config.get('hidden_dim', 'N/A'),
                        'Layers': config.get('num_layers', 'N/A'),
                        'Learning Rate': lr_str,
                        'Win Rate (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'Weighted Sum': summary.get('validation_weighted_sum', 0)
                    }
                    data.append(run_data)
                except Exception as e:
                    print(f"Error processing run for model table: {e}")
                    continue
            
            # Convert to DataFrame and sort by win rate
            if data:
                df = pd.DataFrame(data)
                return df.sort_values('Win Rate (%)', ascending=False)
            else:
                print("Warning: No valid run data found for model table")
                return pd.DataFrame(columns=[
                    'Config', 'Hidden Dim', 'Layers', 'Learning Rate',
                    'Win Rate (%)', 'vs WSPT (%)', 'Weighted Sum'
                ])
        except Exception as e:
            print(f"Error creating model table: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_summary_table(self, runs: List[Dict[str, Any]], sweep_type: str, top_k: int = 5) -> pd.DataFrame:
        """
        Create a summary table of top-performing runs.
        
        Args:
            runs: List of run dictionaries with config and summary
            sweep_type: Type of sweep ('environment', 'feature', 'reward', 'model')
            top_k: Number of top runs to include
            
        Returns:
            DataFrame with summary data
        """
        try:
            # Skip if no runs
            if not runs:
                print("No runs provided for summary table")
                return pd.DataFrame()
                
            # Get runs with validation metrics
            runs_with_metrics = [
                run for run in runs 
                if 'summary' in run and 'validation_win_rate' in run['summary']
            ]
            
            if not runs_with_metrics:
                print("No runs with validation metrics found for summary table")
                return pd.DataFrame()
                
            # Sort by win rate (descending)
            top_runs = sorted(
                runs_with_metrics, 
                key=lambda x: x['summary'].get('validation_win_rate', 0), 
                reverse=True
            )[:top_k]
            
            # Create table data based on sweep type
            sweep_type_lower = sweep_type.lower()
            
            if sweep_type_lower in ['environment', 'env']:
                data = []
                for run in top_runs:
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    n_j = config.get('n_j', 'N/A')
                    n_m = config.get('n_m', 'N/A')
                    weight_low = config.get('weight_low', 'N/A')
                    weight_high = config.get('weight_high', 'N/A')
                    
                    data.append({
                        'Config': f"{n_j}×{n_m}",
                        'Weights': f"{weight_low}-{weight_high}",
                        'Win (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'vs SPT (%)': summary.get('validation_win_vs_spt', 0),
                        'vs SRPT (%)': summary.get('validation_win_vs_srpt', 0),
                        'W. Sum': summary.get('validation_weighted_sum', 0)
                    })
                
            elif sweep_type_lower in ['feature', 'features']:
                data = []
                for run in top_runs:
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Format feature set
                    feature_set = config.get('feature_set', [])
                    if isinstance(feature_set, list):
                        feature_str = ", ".join(feature_set)
                    else:
                        feature_str = str(feature_set)
                    
                    data.append({
                        'Features': feature_str,
                        'Win (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'W. Sum': summary.get('validation_weighted_sum', 0)
                    })
                
            elif sweep_type_lower in ['reward', 'rewards']:
                data = []
                for run in top_runs:
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    data.append({
                        'Reward': config.get('reward_strategy', 'default'),
                        'Win (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'W. Sum': summary.get('validation_weighted_sum', 0)
                    })
                
            elif sweep_type_lower in ['model', 'models']:
                data = []
                for run in top_runs:
                    config = run.get('config', {})
                    summary = run.get('summary', {})
                    
                    # Format learning rate
                    lr = config.get('lr')
                    lr_str = f"{float(lr):.1e}" if lr is not None else "N/A"
                    
                    data.append({
                        'Hidden': config.get('hidden_dim', 'N/A'),
                        'Layers': config.get('num_layers', 'N/A'),
                        'LR': lr_str,
                        'Win (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'W. Sum': summary.get('validation_weighted_sum', 0)
                    })
                
            else:
                # Default format if sweep type is unknown
                data = []
                for run in top_runs:
                    summary = run.get('summary', {})
                    
                    data.append({
                        'Run': run.get('name', 'Unknown'),
                        'Win (%)': summary.get('validation_win_rate', 0),
                        'vs WSPT (%)': summary.get('validation_win_vs_wspt', 0),
                        'W. Sum': summary.get('validation_weighted_sum', 0)
                    })
            
            # Create DataFrame
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error creating summary table: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_comparison_table(self, runs: List[Dict[str, Any]], 
                              primary_param: str, 
                              metrics: List[str] = None) -> pd.DataFrame:
        """
        Create a table comparing runs based on a primary parameter.
        
        Args:
            runs: List of run dictionaries with config and summary
            primary_param: Parameter to use for grouping (e.g., 'feature_set', 'reward_strategy')
            metrics: List of metrics to include (defaults to standard set)
            
        Returns:
            DataFrame with comparison data
        """
        try:
            if not runs:
                return pd.DataFrame()
                
            # Default metrics if not provided
            if metrics is None:
                metrics = [
                    'validation_win_rate', 
                    'validation_win_vs_wspt', 
                    'validation_win_vs_spt',
                    'validation_win_vs_srpt',
                    'validation_weighted_sum'
                ]
                
            # Display names for metrics
            metric_display = {
                'validation_win_rate': 'Win Rate (%)',
                'validation_win_vs_wspt': 'vs WSPT (%)',
                'validation_win_vs_spt': 'vs SPT (%)',
                'validation_win_vs_srpt': 'vs SRPT (%)',
                'validation_weighted_sum': 'Weighted Sum',
                'validation_improvement_over_wspt': 'Improv. WSPT (%)',
                'validation_improvement_over_spt': 'Improv. SPT (%)',
                'validation_improvement_over_srpt': 'Improv. SRPT (%)'
            }
            
            # Group runs by parameter value
            param_groups = {}
            for run in runs:
                config = run.get('config', {})
                if primary_param not in config:
                    continue
                
                param_value = config[primary_param]
                
                # Convert to string for complex types
                if isinstance(param_value, list):
                    param_value = ', '.join(map(str, param_value))
                elif not isinstance(param_value, (str, int, float, bool)):
                    param_value = str(param_value)
                
                if param_value not in param_groups:
                    param_groups[param_value] = []
                    
                param_groups[param_value].append(run)
            
            # Calculate average metrics for each group
            data = []
            for param_value, group_runs in param_groups.items():
                row = {primary_param: param_value}
                
                # Add number of runs in group
                row['Runs'] = len(group_runs)
                
                # Calculate metrics
                for metric in metrics:
                    values = [
                        run.get('summary', {}).get(metric, None) 
                        for run in group_runs
                    ]
                    
                    # Filter out None values
                    values = [v for v in values if v is not None]
                    
                    # Add metrics
                    if values:
                        row[metric_display.get(metric, metric)] = np.mean(values)
                    else:
                        row[metric_display.get(metric, metric)] = 'N/A'
                        
                data.append(row)
            
            # Convert to DataFrame
            if data:
                df = pd.DataFrame(data)
                
                # Sort by win rate if it exists, otherwise by first metric
                sort_col = 'Win Rate (%)' if 'Win Rate (%)' in df.columns else df.columns[1]
                return df.sort_values(sort_col, ascending=False)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error creating comparison table: {e}")
            traceback.print_exc()
            return pd.DataFrame()