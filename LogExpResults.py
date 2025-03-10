import json
import os
import time
import datetime
import hashlib
import numpy as np

def log_experiment_results(configs, results, feature_set, metadata=None):
    """
    Log experiment results to a JSON file for tracking hyperparameter tuning.
    Creates a comprehensive record of experiment parameters and results.
    
    Args:
        configs: The configuration object containing all hyperparameters
        results: Dictionary containing performance metrics
        feature_set: List of features used in the experiment
        metadata: Optional dictionary with additional information
    """
    # Prepare JSON file path
    json_dir = './experiment_logs'
    os.makedirs(json_dir, exist_ok=True)
    
    # Create a unique experiment ID based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"
    
    # Prepare experiment record with metadata
    experiment = {
        'id': experiment_id,
        'timestamp': timestamp,
        'duration_minutes': metadata.get('duration_minutes', 0) if metadata else 0,
        'description': metadata.get('description', '') if metadata else '',
        
        # Environment parameters
        'environment': {
            'n_j': metadata.get('n_j', configs.n_j if hasattr(configs, 'n_j') else None),
            'n_m': metadata.get('n_m', configs.n_m if hasattr(configs, 'n_m') else None),
            'device': metadata.get('device', configs.device if hasattr(configs, 'device') else None),
            'num_envs': metadata.get('num_envs', configs.num_envs if hasattr(configs, 'num_envs') else None),
            'max_updates': metadata.get('max_updates', configs.max_updates if hasattr(configs, 'max_updates') else None),
            'feature_set': feature_set,
            'wspt_guidance_enabled': metadata.get('wspt_guidance_enabled', False),
            'wspt_guidance_duration': metadata.get('wspt_guidance_duration', 0.0),
        },
        
        # Model architecture
        'architecture': {
            'num_layers': configs.num_layers if hasattr(configs, 'num_layers') else None,
            'input_dim': configs.input_dim if hasattr(configs, 'input_dim') else None,
            'hidden_dim': configs.hidden_dim if hasattr(configs, 'hidden_dim') else None,
            'num_mlp_layers_feature_extract': configs.num_mlp_layers_feature_extract if hasattr(configs, 'num_mlp_layers_feature_extract') else None,
            'num_mlp_layers_actor': configs.num_mlp_layers_actor if hasattr(configs, 'num_mlp_layers_actor') else None,
            'hidden_dim_actor': configs.hidden_dim_actor if hasattr(configs, 'hidden_dim_actor') else None,
            'num_mlp_layers_critic': configs.num_mlp_layers_critic if hasattr(configs, 'num_mlp_layers_critic') else None,
            'hidden_dim_critic': configs.hidden_dim_critic if hasattr(configs, 'hidden_dim_critic') else None,
        },
        
        # Training parameters
        'training_params': {
            'lr': configs.lr if hasattr(configs, 'lr') else None,
            'decayflag': configs.decayflag if hasattr(configs, 'decayflag') else None,
            'decay_step_size': configs.decay_step_size if hasattr(configs, 'decay_step_size') else None,
            'decay_ratio': configs.decay_ratio if hasattr(configs, 'decay_ratio') else None,
            'gamma': configs.gamma if hasattr(configs, 'gamma') else None,
            'k_epochs': configs.k_epochs if hasattr(configs, 'k_epochs') else None,
            'eps_clip': configs.eps_clip if hasattr(configs, 'eps_clip') else None,
            'vloss_coef': configs.vloss_coef if hasattr(configs, 'vloss_coef') else None,
            'ploss_coef': configs.ploss_coef if hasattr(configs, 'ploss_coef') else None,
            'entloss_coef': configs.entloss_coef if hasattr(configs, 'entloss_coef') else None,
            'rewardscale': configs.rewardscale if hasattr(configs, 'rewardscale') else None,
            'init_quality_flag': configs.init_quality_flag if hasattr(configs, 'init_quality_flag') else None,
        },
        
        # Results (full metrics)
        'results': {}
    }
    
    # Convert numpy arrays and other complex types for JSON serialization
    def prepare_for_json(obj):
        if isinstance(obj, np.ndarray):
            # For small arrays, we might want to keep them
            if obj.size <= 100:  # Example threshold
                return obj.tolist()
            # For larger arrays, store just summary statistics
            return {
                'type': 'numpy.ndarray',
                'shape': obj.shape,
                'mean': float(np.mean(obj)),
                'min': float(np.min(obj)),
                'max': float(np.max(obj)),
                'std': float(np.std(obj))
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)) and len(obj) > 100:
            # Summarize long lists too
            return {
                'type': 'list',
                'length': len(obj),
                'sample': obj[:5],  # First 5 elements
                'summary': {
                    'mean': np.mean(obj),
                    'min': np.min(obj),
                    'max': np.max(obj)
                } if all(isinstance(x, (int, float)) for x in obj[:10]) else None
            }
        else:
            return obj
    
    # Add all results, converting as needed
    for key, value in results.items():
        experiment['results'][key] = prepare_for_json(value)
    
    # Calculate a configuration hash for duplicate checking
    config_dict = {
        **experiment['environment'],
        **experiment['architecture'],
        **experiment['training_params']
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    experiment['config_hash'] = config_hash
    
    # Check for duplicates in previous runs
    index_file = os.path.join(json_dir, 'experiment_index.json')
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r') as f:
                experiment_index = json.load(f)
            
            # Check if this config hash already exists
            if config_hash in experiment_index:
                print(f"Configuration {config_hash} already exists in experiments:")
                for exp_id in experiment_index[config_hash]:
                    print(f"  - {exp_id}")
                
                # We could still save it, but with a note about the duplicate
                experiment['is_duplicate'] = True
                experiment['duplicate_of'] = experiment_index[config_hash]
            else:
                experiment['is_duplicate'] = False
                # Add to index
                experiment_index[config_hash] = experiment_index.get(config_hash, []) + [experiment_id]
        except Exception as e:
            print(f"Error checking experiment index: {e}")
            experiment_index = {config_hash: [experiment_id]}
    else:
        # Create new index
        experiment_index = {config_hash: [experiment_id]}
    
    # Save the experiment data
    experiment_file = os.path.join(json_dir, f"{experiment_id}.json")
    with open(experiment_file, 'w') as f:
        json.dump(experiment, f, indent=2, default=str)
    
    # Update the index
    with open(index_file, 'w') as f:
        json.dump(experiment_index, f, indent=2)
    
    print(f"Experiment results logged to {experiment_file}")
    print(f"Configuration hash: {config_hash}")
    
    # Also maintain a CSV summary for quick reference (optional)
    summary_file = os.path.join(json_dir, 'experiments_summary.csv')
    summary_exists = os.path.exists(summary_file)
    
    import csv
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(['ID', 'Timestamp', 'Config Hash', 'N_J', 'N_M', 
                            'Training Complete', 'Testing Complete', 
                            'Best Validation', 'Test Win Rate'])
        
        writer.writerow([
            experiment_id,
            timestamp,
            config_hash,
            experiment['environment']['n_j'],
            experiment['environment']['n_m'],
            results.get('training_completed', False),
            results.get('testing_completed', False),
            results.get('validation_weighted_sum', None),
            results.get('test_win_rate', None)
        ])
    
    return experiment_id