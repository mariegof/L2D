# mini_test_sweep.py

import os
import wandb
import yaml
import time
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path

# Mini test sweep configuration - uses a smaller configuration for quick testing
TEST_SWEEP_CONFIG = {
    "type": "feature",            # Testing feature combinations
    "project": "jssp-test",       # Use a test project to avoid cluttering main project
    "entity": None,               # Your WandB username if needed
    "count": 2,                   # Just 2 runs to verify everything works
    "workers": 1,                 # Single worker for simplicity
    "config_file": None,          # Use default config
    "display_name": "server_test_sweep",   # Easily identifiable name
    "max_updates": 5
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

def get_sweep_config(sweep_type, custom_config=None, config=None):
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
        
    # Add max_updates parameter if specified in the config
    if config and "max_updates" in config:
        sweep_config['parameters']['max_updates'] = {
            'value': config["max_updates"]
        }
        
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
        
        # Run the agent and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output in real-time
        stdout_lines = []
        for line in process.stdout:
            stdout_lines.append(line.strip())
            print(f"Agent {agent_id}: {line.strip()}")
            
        process.wait()
        
        if process.returncode != 0:
            if "Finished run successfully" in "\n".join(stdout_lines):
                # Sometimes WandB returns non-zero codes even when successful
                print(f"Agent {agent_id} completed runs successfully despite error code")
                return True
            else:
                print(f"Agent {agent_id} exited with error code {process.returncode}")
                print("Last few log lines:")
                for line in stdout_lines[-5:]:
                    print(f"  {line}")
                return False
            
        print(f"Agent {agent_id} completed {count} runs successfully")
        return True
    except Exception as e:
        print(f"Error in agent {agent_id}: {str(e)}")
        return False

def main(config=None):
    """Run a sweep with the given configuration."""
    if config is None:
        config = TEST_SWEEP_CONFIG
    
    # Create directory structure
    create_directory_structure()
    
    # Get sweep configuration - pass the config to get_sweep_config
    sweep_config, timestamp, output_dir = get_sweep_config(config["type"], config["config_file"], config)
    
    # Add display name to sweep config (instead of passing as parameter)
    if config.get("display_name"):
        if "name" not in sweep_config:
            sweep_config["name"] = config["display_name"]
    
    # Initialize WandB and create sweep
    try:
        wandb.login()
        
        # Create the sweep - don't pass the name as a separate parameter
        sweep_id = wandb.sweep(
            sweep_config, 
            project=config["project"],
            entity=config["entity"]
        )
        
        print(f"Created sweep with ID: {sweep_id}")
        print(f"Results will be saved to: {output_dir}")
        
        # Save sweep information
        sweep_info = {
            'sweep_id': sweep_id,
            'sweep_type': config["type"],
            'timestamp': timestamp,
            'display_name': config.get("display_name", ""),
            'total_runs': config["count"],
            'config_path': config["config_file"] if config["config_file"] else f'configs/{config["type"]}_sweeps.yaml',
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
            
        print(f"All {config['count']} runs completed for sweep {sweep_id}")
        
        # Generate README with sweep information
        readme_content = f"""# Test Sweep

Sweep ID: {sweep_id}
Timestamp: {timestamp}
Type: {config["type"]}
Total Runs: {config["count"]}
Concurrent Workers: {workers}
Project: {config["project"]}

## Results
- WandB URL: https://wandb.ai/{config["entity"] + '/' if config["entity"] else ''}{config["project"]}/sweeps/{sweep_id}
"""
        
        with open(f'{output_dir}/README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"Sweep documentation saved to: {output_dir}/README.md")
        print(f"View your sweep results at: https://wandb.ai/{config['entity'] + '/' if config['entity'] else ''}{config['project']}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error running sweep: {str(e)}")
        return None

if __name__ == "__main__":
    main(TEST_SWEEP_CONFIG)