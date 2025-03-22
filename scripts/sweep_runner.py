#!/usr/bin/env python
"""
Unified sweep runner script for L2D-weighted project.
This script provides a standardized way to run different types of parameter sweeps
while ensuring consistent directory structure and result organization.
"""

import os
import sys
import argparse
import wandb
import yaml
import subprocess
import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run WandB sweep with organized outputs')
    parser.add_argument('--type', type=str, required=True, choices=['environment', 'feature', 'model', 'reward'],
                        help='Type of sweep to run')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom sweep configuration file (optional)')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of runs to execute in the sweep')
    parser.add_argument('--project', type=str, default='jssp-weighted-sum',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for this sweep (default: auto-generated based on type and timestamp)')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes to attach to this sweep')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Tags to attach to this sweep')
    return parser.parse_args()

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    if 'parameters' not in sweep_config:
        sweep_config['parameters'] = {}
        
    # Add a parameter to specify where results should be saved
    sweep_config['parameters']['output_dir'] = {
        'value': f'data/results/{sweep_type}_sweeps/{timestamp}'
    }
    
    # Create the output directory
    os.makedirs(f'data/results/{sweep_type}_sweeps/{timestamp}', exist_ok=True)
    
    return sweep_config, timestamp

def run_sweep(args):
    """Initialize and run the sweep."""
    # Create the directory structure
    create_directory_structure()
    
    # Get sweep configuration
    sweep_config, timestamp = get_sweep_config(args.type, args.config)
    
    # Generate sweep name if not provided
    sweep_name = args.name if args.name else f"{args.type}_sweep_{timestamp}"
    
    # Initialize WandB
    wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=args.project,
        entity=args.entity,
        name=sweep_name
    )
    
    # Save sweep information
    sweep_info = {
        'sweep_id': sweep_id,
        'sweep_type': args.type,
        'timestamp': timestamp,
        'name': sweep_name,
        'notes': args.notes,
        'tags': args.tags,
        'run_count': args.count,
        'config_path': args.config if args.config else f'configs/{args.type}_sweeps.yaml'
    }
    
    with open(f'data/results/{args.type}_sweeps/{timestamp}/sweep_info.yaml', 'w') as f:
        yaml.dump(sweep_info, f)
    
    print(f"Created sweep '{sweep_name}' with ID: {sweep_id}")
    print(f"Results will be saved to: data/results/{args.type}_sweeps/{timestamp}/")
    
    # Run the sweep agent
    agent_command = ["wandb", "agent"]
    if args.entity:
        agent_command.append(f"{args.entity}/{args.project}/{sweep_id}")
    else:
        agent_command.append(f"{args.project}/{sweep_id}")
        
    agent_command.extend(["--count", str(args.count)])
    
    print(f"Running sweep agent with command: {' '.join(agent_command)}")
    subprocess.run(agent_command)
    
    print(f"Completed {args.count} runs for sweep {sweep_id}")
    
    # Generate tables after sweep completion
    table_command = ["python", "scripts/generate_tables.py", 
                     "--project", args.project, 
                     "--sweep_id", sweep_id,
                     "--output", f"data/results/{args.type}_sweeps/{timestamp}/tables",
                     "--table_type", args.type]
    
    print("Generating result tables...")
    subprocess.run(table_command)
    print(f"Tables saved to: data/results/{args.type}_sweeps/{timestamp}/tables/")
    
    # Create a README with sweep information
    readme_content = f"""# {sweep_name.replace('_', ' ').title()}

Sweep ID: {sweep_id}
Timestamp: {timestamp}
Type: {args.type}
Run Count: {args.count}
Project: {args.project}

## Description
{args.notes if args.notes else "No description provided."}

## Tags
{', '.join(args.tags) if args.tags else "No tags."}

## Results
- Raw data in WandB: https://wandb.ai/{args.entity + '/' if args.entity else ''}{args.project}/sweeps/{sweep_id}
- LaTeX tables: ./tables/
- Best parameters: See tables/{args.type}_params_table_*.tex
- Performance comparison: See tables/{args.type}_performance_table_*.tex

## How to use the best model
Check the performance table to identify the best run, then find the corresponding
model file in the models/best/ directory.
"""
    
    with open(f'data/results/{args.type}_sweeps/{timestamp}/README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"Sweep documentation saved to: data/results/{args.type}_sweeps/{timestamp}/README.md")
    return sweep_id

if __name__ == "__main__":
    args = parse_args()
    sweep_id = run_sweep(args)