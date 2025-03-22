import os
import sys
import argparse
import wandb
import yaml
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Run WandB sweep for feature combinations')
    parser.add_argument('--config', type=str, default='configs/feature_sweeps.yaml',
                        help='Path to sweep configuration file')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of runs to execute in the sweep')
    parser.add_argument('--project', type=str, default='jssp-weighted-sum',
                        help='WandB project name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize WandB
    wandb.login()
    
    # Load sweep configuration
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Created sweep with ID: {sweep_id}")
    
    # Run the sweep
    subprocess.run(["wandb", "agent", f"{args.project}/{sweep_id}", "--count", str(args.count)])
    
    print(f"Completed {args.count} runs for sweep {sweep_id}")

if __name__ == "__main__":
    main()