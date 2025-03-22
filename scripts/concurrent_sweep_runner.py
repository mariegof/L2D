#!/usr/bin/env python
"""
Concurrent sweep runner for L2D-weighted project.
Launches a WandB sweep with multiple agents running in parallel.
"""

import os
import sys
import argparse
import wandb
import yaml
import time
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path

# Configuration dictionary - edit these values directly
SWEEP_CONFIG = {
    "type": "feature",           # Sweep type: "environment", "feature", "model", or "reward"
    "project": "jssp-weighted-sum",  # WandB project name
    "entity": None,              # WandB entity (username) - set to None if not using
    "count": 10,                 # Total number of runs to execute
    "workers": None,             # Number of concurrent workers (None = CPU count - 1)
    "config_file": None,         # Path to custom sweep config file (None = use default)
    "name": None                 # Custom name for the sweep (None = auto-generate)
}

def create_directory_structure():
    """Create the standard directory structure if it doesn't exist."""
    directories = [
        'configs',
        'data/instances/uniform_weights',
        'data/instances/variable_weights',
        'data/results/environment_sweeps',
        'data/results/feature_sweeps',
        'data/results/model_sweeps',
        'data/results/reward_sweeps',
        'data/results/latex_tables',
        'models/checkpoints',
        'models/best',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("Directory structure verified.")

def get_sweep_config(sweep_type, custom_config=None):
    """Get the sweep configuration, either from default or custom file."""
    if custom_config and os.path.exists(custom_config):
        config_path = custom_config
    else:
        config_path = f'configs/{sweep_type}_sweeps.yaml'
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Sweep configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
        
    # Add standard output directory to configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'parameters' not in sweep_config:
        sweep_config['parameters'] = {}
        
    # Add a parameter to specify where results should be saved
    output_dir = f'data/results/{sweep_type}_sweeps/{timestamp}'
    sweep_config['parameters']['output_dir'] = {
        'value': output_dir
    }
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    return sweep_config, timestamp, output_dir

def run_sweep_agent(project, sweep_id, entity=None, count=1, agent_id=0):
    """Run a single sweep agent."""
    try:
        print(f"Agent {agent_id} starting to process {count} runs...")
        
        # Build the wandb agent command
        cmd = ["wandb", "agent"]
        if entity:
            cmd.append(f"{entity}/{project}/{sweep_id}")
        else:
            cmd.append(f"{project}/{sweep_id}")
        
        cmd.extend(["--count", str(count)])
        
        # Run the agent
        subprocess.run(cmd, check=True)
        print(f"Agent {agent_id} completed {count} runs successfully")
        return True
    except Exception as e:
        print(f"Error in agent {agent_id}: {str(e)}")
        return False

def main(config=SWEEP_CONFIG):    
    # Create directory structure
    create_directory_structure()
    
    # Get sweep configuration
    # Get sweep configuration
    sweep_config, timestamp, output_dir = get_sweep_config(
        config["type"], 
        config["config_file"]
    )
    
    # Generate sweep name if not provided
    sweep_name = config["name"] if config["name"] else f"{config['type']}_sweep_{timestamp}"
    
    # Initialize WandB and create sweep
    try:
        wandb.login()
        
        # Create the sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=config["project"],
            entity=config["entity"],
            name=sweep_name
        )
        
        print(f"Created sweep '{sweep_name}' with ID: {sweep_id}")
        print(f"Results will be saved to: {output_dir}")
        
        # Save sweep information
        sweep_info = {
            'sweep_id': sweep_id,
            'sweep_type': config["type"],
            'timestamp': timestamp,
            'name': sweep_name,
            'total_runs': config["count"],
            'config_path': config["config"] if config["config"] else f'configs/{config["type"]}_sweeps.yaml',
            'output_dir': output_dir
        }
        
        with open(f'{output_dir}/sweep_info.yaml', 'w') as f:
            yaml.dump(sweep_info, f)
            
        # Determine the number of parallel workers
        workers = config["workers"] if config["workers"] is not None else max(1, multiprocessing.cpu_count() - 1)
        workers = min(workers, config["count"])  # Don't use more workers than runs
        
        print(f"Running sweep with {workers} concurrent workers")
        
        # Calculate runs per worker
        runs_per_worker = [config["count"] // workers] * workers
        # Distribute any remainder
        for i in range(config["count"] % workers):
            runs_per_worker[i] += 1
            
        # Start worker processes
        processes = []
        for i in range(workers):
            if runs_per_worker[i] > 0:
                p = multiprocessing.Process(
                    target=run_sweep_agent,
                    args=(config["project"], sweep_id, config["entity"], runs_per_worker[i], i)
                )
                processes.append(p)
                p.start()
                
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
        print(f"All {config["count"]} runs completed for sweep {sweep_id}")
        
        # Generate tables after sweep completion
        print("Generating result tables...")
        try:
            table_cmd = ["python", "scripts/generate_tables.py", 
                         "--project", config["project"], 
                         "--sweep_id", sweep_id,
                         "--output", f"{output_dir}/tables",
                         "--table_type", config["type"]]
            
            if config["entity"]:
                table_cmd.extend(["--entity", config["entity"]])
                
            subprocess.run(table_cmd, check=True)
            print(f"Tables saved to: {output_dir}/tables/")
        except Exception as e:
            print(f"Error generating tables: {str(e)}")
        
        # Create a README with sweep information
        readme_content = f"""# {sweep_name.replace('_', ' ').title()}

Sweep ID: {sweep_id}
Timestamp: {timestamp}
Type: {config["type"]}
Total Runs: {config["count"]}
Concurrent Workers: {workers}
Project: {config["project"]}

## Results
- WandB URL: https://wandb.ai/{config["entity"] + '/' if config["entity"] else ''}{config["project"]}/sweeps/{sweep_id}
- LaTeX tables: ./tables/
- Performance graphs: ./tables/

## How to use the best model
Check the performance table to identify the best run, then find the corresponding
model file in the WandB artifacts or in the models/best/ directory.
"""
        
        with open(f'{output_dir}/README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"Sweep documentation saved to: {output_dir}/README.md")
        print(f"View your sweep results at: https://wandb.ai/{config["entity"] + '/' if config["entity"] else ''}{config["project"]}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error running sweep: {str(e)}")
        return None

if __name__ == "__main__":
    main()