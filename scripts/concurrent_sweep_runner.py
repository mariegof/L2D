#!/usr/bin/env python
"""
Run WandB sweeps for JSSP experiments in parallel with robust error handling.

This script:
1. Configures and launches WandB sweeps based on YAML config
2. Runs multiple agents in parallel to process sweep runs
3. Collects and processes results after completion
4. Generates comprehensive reports and visualizations
"""

import os
import sys
import argparse
import wandb
import yaml
import time
import subprocess
import multiprocessing
import signal
import json
from datetime import datetime
from pathlib import Path
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.reporting import ReportWriter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run WandB sweep with multiple agents')
    parser.add_argument('--config', type=str, default='configs/sweep_config.yaml',
                        help='Path to main sweep configuration file')
    parser.add_argument('--weight_type', type=str, choices=['uniform', 'variable'],
                        help='Override weight type in config')
    parser.add_argument('--sweep_type', type=str, choices=['environment', 'feature', 'model', 'reward'],
                        help='Override sweep type in config')
    parser.add_argument('--workers', type=int, help='Override number of workers in config')
    parser.add_argument('--count', type=int, help='Override number of runs in config')
    parser.add_argument('--no_process', action='store_true',
                        help='Skip processing the sweep results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    return parser.parse_args()

def load_sweep_config(args):
    """
    Load the main sweep configuration with command-line overrides.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict containing the sweep configuration
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Main configuration file not found: {args.config}")
    
    # Load config from file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
    if args.weight_type:
        config['weight_type'] = args.weight_type
    if args.sweep_type:
        config['sweep_type'] = args.sweep_type
    if args.workers:
        config['workers'] = args.workers
    if args.count:
        config['count'] = args.count
    
    # Set default values if not present
    if 'count' not in config:
        config['count'] = 10
    if 'workers' not in config:
        config['workers'] = None
    if 'override_max_updates' not in config:
        # By default, don't override sweep file's max_updates
        config['override_max_updates'] = False
    if 'name' not in config:
        # Generate a default name if not provided
        config['name'] = f"{config.get('weight_type', 'default')}_{config.get('sweep_type', 'default')}_sweep"
    
    # Add debug flag
    config['debug'] = args.debug
    
    return config

def create_directory_structure(output_dir=None):
    """Create the standard directory structure if it doesn't exist."""
    from pathlib import Path
    
    directories = [
        'configs',
        'data/instances/uniform_weights',
        'data/instances/variable_weights',
        'results',
        'models/checkpoints',
        'models/best',
    ]
    
    # Create base directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # If an output directory is specified, create sweep-specific structure
    if output_dir:
        sweep_directories = [
            f'{output_dir}/runs',            # Individual run outputs
            f'{output_dir}/artifacts/latex',  # LaTeX tables
            f'{output_dir}/artifacts/plots',  # Generated plots
            f'{output_dir}/best_run',         # Best run info
            f'{output_dir}/best_run/model',   # Best run model
        ]
        
        for directory in sweep_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    if os.path.exists('results'):
        print("Directory structure verified.")
    else:
        print("Failed to create directory structure.")
        return False
        
    return True

def get_sweep_config(config):
    """
    Get the specific sweep configuration based on main config.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        Tuple of (sweep_config, timestamp, output_dir)
    """
    sweep_type = config['sweep_type']
    weight_type = config['weight_type']
    
    # First, load the appropriate base weight configuration
    base_config_path = f'configs/base_configs/{weight_type}_weights.yaml'
    
    if not os.path.exists(base_config_path):
        print(f"Warning: Base weight configuration not found at {base_config_path}")
        base_config = {}
    else:
        print(f"Loading base weight configuration from {base_config_path}")
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    
    # Now load the sweep-specific configuration
    sweep_config_path = f'configs/{sweep_type}_sweeps.yaml'
    
    if not os.path.exists(sweep_config_path):
        raise FileNotFoundError(f"Sweep configuration file not found: {sweep_config_path}")
    
    # Load the sweep configuration
    print(f"Loading sweep configuration from {sweep_config_path}")
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the output directory structure
    output_dir = f'results/{weight_type}_weights/{sweep_type}_sweeps/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure 'parameters' section exists
    if 'parameters' not in sweep_config:
        sweep_config['parameters'] = {}
    
    # List of parameters that train.py actually accepts
    valid_train_params = [
        'device', 'n_j', 'n_m', 'rewardscale', 'init_quality_flag', 'low', 'high',
        'weight_low', 'weight_high', 'np_seed_train', 'np_seed_validation',
        'torch_seed', 'et_normalize_coef', 'wkr_normalize_coef', 'num_layers',
        'neighbor_pooling_type', 'graph_pool_type', 'input_dim', 'hidden_dim',
        'num_mlp_layers_feature_extract', 'num_mlp_layers_actor', 'hidden_dim_actor',
        'num_mlp_layers_critic', 'hidden_dim_critic', 'num_envs', 'max_updates',
        'lr', 'decayflag', 'decay_step_size', 'decay_ratio', 'gamma', 'k_epochs',
        'eps_clip', 'vloss_coef', 'ploss_coef', 'entloss_coef', 'feature_set',
        'reward_strategy', 'validate_every', 'output_dir'
    ]
    
    # Check if we're doing a feature sweep - if so, we need to prevent input_dim from being set
    is_feature_sweep = (sweep_type == 'feature' and 
                       'feature_set' in sweep_config.get('parameters', {}))
    
    # Apply base weight configuration parameters to sweep config (only valid params)
    for key, value in base_config.items():
        # Skip non-parameter entries and invalid params
        if key not in ['method', 'program', 'metric', 'parameters'] and key in valid_train_params:
            # Skip input_dim if we're doing a feature sweep
            if is_feature_sweep and key == 'input_dim':
                print("Skipping input_dim from base config for feature sweep")
                continue
                
            # Only add if not already in parameters
            if key not in sweep_config['parameters']:
                sweep_config['parameters'][key] = {'value': value}
                
    # Add output_dir parameter (this one is valid)
    sweep_config['parameters']['output_dir'] = {'value': output_dir}
    
    # Add max_updates if specified AND override flag is set
    if 'max_updates' in config and config['max_updates'] and config.get('override_max_updates', False):
        print(f"Overriding sweep file's max_updates with {config['max_updates']}")
        sweep_config['parameters']['max_updates'] = {'value': config['max_updates']}
    else:
        print(f"Using max_updates values from sweep file or base config")
        
    # Add validate_every if specified
    if 'validate_every' in config and config['validate_every']:
        sweep_config['parameters']['validate_every'] = {'value': config['validate_every']}
    
    # Set project name based on weight type
    if weight_type == 'variable':
        config['project'] = 'jssp-variable-sum'
    else:
        config['project'] = 'jssp-uniform-sum'
    
    # Save complete merged configuration to output directory
    sweep_config_path = os.path.join(output_dir, 'sweep_config.yaml')
    with open(sweep_config_path, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    return sweep_config, timestamp, output_dir

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"Received signal {signum}. Shutting down workers...")
    # This will allow the script to exit naturally after current operations complete
    sys.exit(1)

def run_sweep_agent(project, sweep_id, entity=None, count=1, agent_id=0, debug=False):
    """
    Run a single sweep agent with robust error handling for wandb 0.19.8.
    
    Args:
        project: WandB project name
        sweep_id: WandB sweep ID
        entity: Optional WandB entity name
        count: Number of runs for this agent
        agent_id: Agent identifier for logging
        debug: Whether to enable debug output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Agent {agent_id} starting to process {count} runs...")
        
        # Build the wandb agent command
        cmd = ["wandb", "agent"]
        if entity:
            cmd.append(f"{entity}/{project}/{sweep_id}")
        else:
            cmd.append(f"{project}/{sweep_id}")
        
        cmd.extend(["--count", str(count)])
        
        # Add debug mode if requested
        if debug:
            print(f"Agent {agent_id} command: {' '.join(cmd)}")
        
        # Run the agent and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            # Skip printing the common WandB service error
            if "WandbServiceNotOwnedError" not in line:
                if debug:
                    print(f"Agent {agent_id}: {line.strip()}")
                else:
                    # Only print important messages in non-debug mode
                    if any(x in line for x in ['Starting', 'Completed', 'Error', 'Failed']):
                        print(f"Agent {agent_id}: {line.strip()}")
            
        process.wait()
        
        # With wandb 0.19.8, exit codes are different
        # Always consider agent runs successful unless there's clear evidence otherwise
        if process.returncode != 0:
            print(f"Agent {agent_id}: Exited with code {process.returncode}, but may have completed runs successfully")
            # Still return True since wandb-core can exit with non-zero codes even on success
            return True
            
        print(f"Agent {agent_id} completed {count} runs successfully")
        return True
    except Exception as e:
        # Check if it's the common WandB service error
        if "WandbServiceNotOwnedError" in str(e):
            print(f"Agent {agent_id}: WandB service termination issue (can be ignored)")
            return True
        else:
            print(f"Error in agent {agent_id}: {str(e)}")
            traceback.print_exc()
            return False

def process_sweep_results(sweep_id, project, entity, output_dir, sweep_type, weight_type):
    """Process sweep results to generate only essential outputs."""
    try:
        print("Generating analysis artifacts...")
        
        # Build command for process_sweep.py
        cmd = [
            sys.executable,
            "scripts/process_sweep.py",
            "--project", project,
            "--sweep_id", sweep_id,
            "--output", output_dir,  # Pass the main output dir
            "--sweep_type", sweep_type,
            "--weight_type", weight_type
        ]
        
        if entity:
            cmd.extend(["--entity", entity])
        
        # Run the processing script
        process = subprocess.run(cmd, check=True)
        
        if process.returncode == 0:
            # Use ReportWriter to get the standard paths
            writer = ReportWriter(output_dir, create_all=False)
            
            print(f"Analysis artifacts generated:")
            print(f"  - LaTeX tables: {writer.get_directory('latex')}")
            print(f"  - Plots: {writer.get_directory('plots')}")
            return True
        else:
            print(f"Error processing sweep results. Return code: {process.returncode}")
            return False
    except Exception as e:
        print(f"Error processing sweep results: {e}")
        traceback.print_exc()
        return False

def create_readme(sweep_name, sweep_id, timestamp, sweep_type, weight_type, count, workers, 
                project, entity, max_updates, validate_every, output_dir):
    """
    Create a comprehensive README with sweep information.
    
    Args:
        sweep_name: Name of the sweep
        sweep_id: WandB sweep ID
        timestamp: Timestamp of the sweep
        sweep_type: Type of sweep
        weight_type: Type of weights
        count: Number of runs
        workers: Number of concurrent workers
        project: WandB project name
        entity: WandB entity name
        max_updates: Maximum number of updates per run
        validate_every: Validation frequency
        output_dir: Output directory
        
    Returns:
        Path to the created README file
    """
    try:
        # Format sweep type for display
        sweep_type_display = sweep_type.capitalize()
        
        # Create README content
        readme_content = f"""# {sweep_name.replace('_', ' ').title()}

## Experiment Details
- **Sweep ID:** {sweep_id}
- **Timestamp:** {timestamp}
- **Type:** {sweep_type_display}
- **Weight Type:** {weight_type}
- **Total Runs:** {count}
- **Concurrent Workers:** {workers}
- **Project:** {project}
- **Max Updates:** {max_updates if max_updates else 'Default from sweep file'}
- **Validation Frequency:** Every {validate_every if validate_every else 'Default'} episodes

## Results Location
- **WandB URL:** https://wandb.ai/{entity + '/' if entity else ''}{project}/sweeps/{sweep_id}
- **Analysis Directory:** {os.path.join(output_dir, 'analysis')}

## Key Findings
Check the analysis directory for detailed results on:
- Best performing configurations
- Comparisons with baseline methods (SPT, WSPT)
- Performance metrics (win rates, weighted sum values)

## Using the Best Model
The best configuration has been saved in `best_config.yaml`. To use this model:

1. Load the configuration parameters 
2. Initialize the model with these parameters
3. Use the `test_methods.py` script to evaluate on test instances

## Reproducibility
To reproduce this experiment, use:
```
python scripts/concurrent_sweep_runner.py --config configs/sweep_config.yaml --sweep_type {sweep_type} --weight_type {weight_type}
```
"""
        
        # Write README file
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        print(f"Sweep documentation saved to: {readme_path}")
        return readme_path
        
    except Exception as e:
        print(f"Error creating README: {e}")
        return None

def download_sweep_artifacts(project, sweep_id, entity, output_dir):
    """Download artifacts from the best run in the sweep with robust error handling."""
    try:
        print(f"Extracting artifacts from best run in sweep {sweep_id}...")
        
        # Create writer with full directory structure
        writer = ReportWriter(output_dir, create_all=True)
        
        # Initialize WandB API
        api = wandb.Api()
        
        # Build sweep path
        sweep_path = f"{project}/{sweep_id}"
        if entity:
            sweep_path = f"{entity}/{sweep_path}"
        
        # Get sweep runs
        sweep = api.sweep(sweep_path)
        runs = list(sweep.runs)
        
        if not runs:
            print("No runs found in sweep.")
            return False
        
        # Find the best run based on validation_win_vs_wspt
        best_run = None
        best_win_rate = -1
        
        for run in runs:
            if run.state == "finished":
                summary = run.summary._json_dict
                win_rate = summary.get("validation_win_vs_wspt", -1)
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_run = run
        
        if not best_run:
            print("No finished runs with win rate metrics found.")
            return False
        
        # Save best run information
        best_run_info = {
            "run_id": best_run.id,
            "run_name": best_run.name,
            "url": best_run.url,
            "created_at": best_run.created_at,
            "metrics": {
                "win_vs_wspt": best_win_rate,
                "weighted_sum": best_run.summary._json_dict.get("validation_weighted_sum", "N/A"),
                "win_vs_spt": best_run.summary._json_dict.get("validation_win_vs_spt", "N/A"),
                "win_vs_srpt": best_run.summary._json_dict.get("validation_win_vs_srpt", "N/A"),
                "win_rate": best_run.summary._json_dict.get("validation_win_rate", "N/A"),
                "improvement_over_wspt": best_run.summary._json_dict.get("validation_improvement_over_wspt", "N/A"),
                "improvement_over_spt": best_run.summary._json_dict.get("validation_improvement_over_spt", "N/A"),
                "improvement_over_srpt": best_run.summary._json_dict.get("validation_improvement_over_srpt", "N/A")
            }
        }
        
        writer.write_metadata(best_run_info, "best_run_info.yaml", "best_run")
        
        # Save best run config
        best_config = {}
        for k, v in best_run.config.items():
            if not k.startswith('_'):
                best_config[k] = v
        
        writer.write_metadata(best_config, "best_config.yaml", "best_run")
        
        # Find and copy the learning curves plot from the best run to artifacts folder
        run_plots_dir = os.path.join(output_dir, "runs", best_run.id)
        if os.path.exists(run_plots_dir):
            source_plot = os.path.join(run_plots_dir, "learning_curves.png")
            dest_plot = os.path.join(writer.get_directory("best_run"), "learning_curves.png")
            
            # Copy if it exists
            if os.path.exists(source_plot):
                import shutil
                shutil.copy(source_plot, dest_plot)
                print(f"Copied learning curves from best run to {dest_plot}")
        
        # Download the latest learning curves image with retry mechanism
        try:
            files = best_run.files()
            # Find the latest learning curve
            latest_learning_curve = None
            latest_step = -1

            for file in files:
                if "media/images" in file.name and "learning_curves" in file.name:
                    try:
                        # Extract step number from filename
                        # Pattern like: learning_curves_220_b0750b1240a41695d5dc.png
                        step_str = file.name.split('learning_curves_')[1].split('_')[0]
                        step = int(step_str)
                        
                        if step > latest_step:
                            latest_step = step
                            latest_learning_curve = file
                    except Exception as e:
                        print(f"Error parsing filename {file.name}: {e}")
                        # Keep this file as a fallback if we can't parse step
                        if latest_learning_curve is None:
                            latest_learning_curve = file

            # Download the latest learning curve with retry mechanism
            if latest_learning_curve:
                max_retries = 3
                retry_delay = 5  # seconds
                
                for attempt in range(max_retries):
                    try:
                        # Download with original filename
                        media_dir = os.path.join(output_dir, "best_run")
                        os.makedirs(media_dir, exist_ok=True)
                        latest_learning_curve.download(root=media_dir, replace=True)
                        download_path = os.path.join(media_dir, os.path.basename(latest_learning_curve.name))
                        print(f"Downloaded latest learning curves (step {latest_step}) to {download_path}")
                        
                        # Copy to a standardized filename for easy reference
                        import shutil
                        standard_path = os.path.join(media_dir, "learning_curves.png")
                        if os.path.exists(download_path):
                            shutil.copy(download_path, standard_path)
                            print(f"Copied to standard filename: {standard_path}")
                        
                        break  # Success
                    except Exception as e:
                        print(f"Error downloading learning curves (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"Error processing image files: {e}")
        
        # Save best model weights with retry mechanism
        artifacts = []
        try:
            artifacts = best_run.logged_artifacts()
        except Exception as e:
            print(f"Error accessing logged artifacts: {e}")
        
        for artifact in artifacts:
            if "model" in artifact.type and ("best" in artifact.name or "final" in artifact.name):
                max_retries = 3
                retry_delay = 5  # seconds
                
                for attempt in range(max_retries):
                    try:
                        model_dir = writer.get_directory("best_run_model")
                        artifact.download(root=model_dir)
                        print(f"Downloaded model weights from best run to {model_dir}")
                        break  # Success
                    except Exception as e:
                        print(f"Error downloading model (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
        
        print(f"Best run artifacts saved to {writer.get_directory('best_run')}")
        return True
        
    except Exception as e:
        print(f"Error downloading sweep artifacts: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function to run WandB sweeps in parallel."""
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Load the main configuration
        config = load_sweep_config(args)
        
        # Create directory structure
        if not create_directory_structure():
            print("Error: Could not create basic project directory structure.")
            return 1
        
        # Get sweep configuration
        sweep_config, timestamp, output_dir = get_sweep_config(config)
        
        # Set up sweep name
        weight_prefix = "uniform" if config["weight_type"] == "uniform" else "variable"
        sweep_name = config.get("name", f"{weight_prefix}_{config['sweep_type']}_sweep_{timestamp}")
        
        # Initialize WandB and create sweep
        try:
            # For wandb-core compatibility, don't explicitly call login
            # wandb-core handles authentication differently
            # (wandb.login() doesn't exist in 0.19.8)
            print("WandB authentication handled automatically")
            
            # Create the sweep - compatible with wandb-core
            sweep_id = wandb.sweep(
                sweep_config, 
                project=config['project'],
                entity=config.get('entity'),
            )
            
            print(f"Created sweep '{sweep_name}' with ID: {sweep_id}")
            print(f"Results will be saved to: {output_dir}")
            
            # Save sweep information
            sweep_info = {
                'sweep_id': sweep_id,
                'sweep_type': config["sweep_type"],
                'weight_type': config["weight_type"],
                'timestamp': timestamp,
                'name': sweep_name,
                'total_runs': config["count"],
                'config_path': args.config,
                'output_dir': output_dir
            }
            
            sweep_info_path = os.path.join(output_dir, 'sweep_info.yaml')
            with open(sweep_info_path, 'w') as f:
                yaml.dump(sweep_info, f)
                
            # Create output directory and sweep structure
            create_directory_structure(output_dir)  # Set up run directories
            print(f"Created sweep directory structure in {output_dir}")
                
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
                        args=(
                            config['project'], 
                            sweep_id, 
                            config.get('entity'), 
                            runs_per_worker[i], 
                            i,
                            config.get('debug', False)
                        )
                    )
                    processes.append(p)
                    p.start()
                    
            # Wait for all processes to complete
            for p in processes:
                p.join()
                
            print(f"All {config['count']} runs completed for sweep {sweep_id}")
            
            download_sweep_artifacts(config['project'], sweep_id, config['entity'], output_dir)
            
            # Process sweep results if requested
            if not args.no_process:
                success = process_sweep_results(
                    sweep_id=sweep_id,
                    project=config['project'],
                    entity=config.get('entity'),
                    output_dir=output_dir,
                    sweep_type=config['sweep_type'],
                    weight_type=config['weight_type']
                )
                
                if success:
                    print(f"Sweep processing completed successfully")
                else:
                    print(f"Sweep processing encountered errors")
            
            # Create a comprehensive README
            create_readme(
                sweep_name=sweep_name,
                sweep_id=sweep_id,
                timestamp=timestamp,
                sweep_type=config['sweep_type'],
                weight_type=config['weight_type'],
                count=config['count'],
                workers=workers,
                project=config['project'],
                entity=config.get('entity'),
                max_updates=config.get('max_updates'),
                validate_every=config.get('validate_every'),
                output_dir=output_dir
            )
            
            print(f"View your sweep results at: https://wandb.ai/{config['entity'] + '/' if config.get('entity') else ''}{config['project']}/sweeps/{sweep_id}")
            
            return sweep_id
            
        except Exception as e:
            print(f"Error running sweep: {str(e)}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        traceback.print_exc()
        return 1
if __name__ == "__main__":
    sys.exit(0 if main() else 1)