"""
Data retrieval module for fetching data from Weights & Biases API with clean, minimal implementation.
"""
import wandb
import json
import os
from typing import List, Dict, Tuple, Any, Optional
import traceback
import pandas as pd

class WandBFetcher:
    """Fetch run data from WandB API in a standardized dictionary format."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WandB API client.
        
        Args:
            api_key: Optional WandB API key for authentication
        """
        self.api = None
        try:
            self.api = wandb.Api()
            if api_key:
                wandb.login(key=api_key)
            print("WandB API initialized successfully")
        except Exception as e:
            print(f"Error initializing WandB API: {e}")
    
    def fetch_sweep_runs(self, project: str, sweep_id: str, entity: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fetch run data from a WandB sweep with minimal processing.
        
        Args:
            project: WandB project name
            sweep_id: WandB sweep ID
            entity: Optional WandB entity (username/organization)
            
        Returns:
            Tuple of (run_list, sweep_config)
        """
        if not self.api:
            print("WandB API not initialized")
            return [], {}
            
        try:
            # Build sweep path
            sweep_path = f"{project}/{sweep_id}"
            if entity:
                sweep_path = f"{entity}/{sweep_path}"
            
            # Get sweep
            sweep = self.api.sweep(sweep_path)
            
            # Extract basic sweep config
            sweep_config = {}
            if hasattr(sweep, 'config') and sweep.config:
                sweep_config = dict(sweep.config)
            
            # Count runs
            all_runs = list(sweep.runs)
            finished_runs = [run for run in all_runs if getattr(run, 'state', None) == "finished"]
            print(f"Found {len(all_runs)} runs, {len(finished_runs)} are finished")
            
            # Process only finished runs
            processed_runs = []
            for run in finished_runs:
                # Create standardized run dictionary
                run_dict = {
                    "name": getattr(run, 'name', getattr(run, 'id', 'Unknown')),
                    "id": getattr(run, 'id', 'Unknown'),
                    "config": {},
                    "summary": {},
                    "url": getattr(run, 'url', '')
                }
                
                # Extract config (skip internal WandB keys)
                if hasattr(run, 'config'):
                    for key, value in dict(run.config).items():
                        if not key.startswith('_'):
                            run_dict["config"][key] = value
                
                # Extract summary metrics
                if hasattr(run, 'summary'):
                    for key, value in dict(run.summary).items():
                        run_dict["summary"][key] = value
                
                processed_runs.append(run_dict)
            
            print(f"Successfully processed {len(processed_runs)} runs")
            return processed_runs, sweep_config
            
        except Exception as e:
            print(f"Error fetching sweep data: {e}")
            traceback.print_exc()
            return [], {}
    
    def load_run_data_from_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load run data from a JSON or CSV file when offline processing is needed.
        
        Args:
            file_path: Path to JSON or CSV file
            
        Returns:
            Tuple of (run_list, sweep_config)
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return [], {}
                
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                runs = data.get('runs', [])
                sweep_config = data.get('sweep_config', {})
                
                return runs, sweep_config
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                # Simple conversion to standard format
                runs = []
                for _, row in df.iterrows():
                    run_dict = {'name': 'unknown', 'config': {}, 'summary': {}}
                    
                    for col, value in row.items():
                        if col.startswith('validation_'):
                            run_dict['summary'][col] = value
                        elif col == 'run_name':
                            run_dict['name'] = value
                        else:
                            run_dict['config'][col] = value
                    
                    runs.append(run_dict)
                
                return runs, {}
            
            else:
                print(f"Unsupported file format: {file_path}")
                return [], {}
                
        except Exception as e:
            print(f"Error loading run data from file: {e}")
            return [], {}