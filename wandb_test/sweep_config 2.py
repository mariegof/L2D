import wandb
from run_exp import train_l2d_multi_env
'''
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for efficient search
    'metric': {
        'name': 'validation_weighted_sum',  # We want to minimize this metric
        'goal': 'minimize'
    },
    'parameters': {
        # Feature combinations (comma-separated strings)
        'feature_set': {
            'values': [
                'LBs,finished_mark,weighted_priorities',  # Basic features
                'LBs,finished_mark,weighted_priorities,normalized_weights',  # Add weights
                'LBs,finished_mark,weighted_priorities,remaining_weighted_work',  # Add remaining work
                'LBs,finished_mark,weighted_priorities,normalized_weights,remaining_weighted_work',  # Combined
                'LBs,finished_mark,weighted_priorities,normalized_weights,machine_contention',  # Add machine contention
                'LBs,finished_mark,weighted_priorities,normalized_weights,remaining_weighted_work,machine_contention',  # Full set
                'LBs,finished_mark,weighted_priorities,normalized_weights,remaining_weighted_work,machine_contention,time_elapsed'  # Everything
            ]
        },
        # Network architecture
        'num_layers': {
            'values': [2, 3, 4]
        },
        'hidden_dim': {
            'values': [32, 64, 128]
        },
        'num_mlp_layers_feature_extract': {
            'values': [1, 2]
        },
        'num_mlp_layers_actor': {
            'values': [1, 2]
        },
        'hidden_dim_actor': {
            'values': [16, 32, 64]
        },
        'num_mlp_layers_critic': {
            'values': [1, 2]
        },
        'hidden_dim_critic': {
            'values': [16, 32, 64]
        },
        # Learning parameters
        'lr': {
            'values': [5e-6, 1e-5, 2e-5, 5e-5]
        },
        'k_epochs': {
            'values': [1, 2, 3]
        },
        'eps_clip': {
            'values': [0.1, 0.2, 0.3]
        },
        'entloss_coef': {
            'values': [0.01, 0.05, 0.1]
        },
        # Number of parallel environments
        'num_envs': {
            'values': [4, 8, 16]
        },
        # WSPT guidance duration (as a fraction of total episodes)
        'wspt_guidance_duration': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.5]
        },
        # Fixed parameters
        'max_updates': {'value': 1000},  # Total episodes per run
        'log_every': {'value': 50},  # Log every n episodes
        #'run_final_eval': {'value': True}  # Whether to run final evaluation
    }
}'''

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'validation_weighted_sum',
        'goal': 'minimize'
    },
    'parameters': {
        # Feature combinations (most important to test)
        'feature_set': {
            'values': [
                'LBs,finished_mark,weighted_priorities',
                'LBs,finished_mark,weighted_priorities,normalized_weights',
                'LBs,finished_mark,weighted_priorities,normalized_weights,remaining_weighted_work',
                'LBs,finished_mark,weighted_priorities,normalized_weights,machine_contention',
                'LBs,finished_mark,weighted_priorities,normalized_weights,remaining_weighted_work,machine_contention'
            ]
        },
        # Learning rate (critical for convergence)
        'lr': {
            'values': [1e-5, 2e-5, 5e-5]
        },
        # GNN hidden dimension (important for representation power)
        'hidden_dim': {
            'values': [64, 128]
        },
        # Actor network hidden dimension
        'hidden_dim_actor': {
            'values': [32, 64]
        },
        # Number of parallel environments
        'num_envs': {
            'values': [8, 16]
        },
        
        # Fixed parameters - keep these at reasonable defaults
        'num_layers': {'value': 3},
        'num_mlp_layers_feature_extract': {'value': 2},
        'num_mlp_layers_actor': {'value': 2},
        'num_mlp_layers_critic': {'value': 2},
        'hidden_dim_critic': {'value': 32},
        'k_epochs': {'value': 1},
        'eps_clip': {'value': 0.2},
        'entloss_coef': {'value': 0.01},
        'max_updates': {'value': 2000},
        'log_every': {'value': 100}
    }
}

# Initialize wandb
wandb.login()

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="WS_L2D", entity="marie-goffin-university-of-li-ge")

# Function for the wandb agent to call
def train_agent():
    # Initialize a wandb run first - this is critical!
    with wandb.init() as run:
        # Now you can safely access the config
        return train_l2d_multi_env(run.config)

# Start the sweep agent
wandb.agent(sweep_id, function=train_agent, count=15)  # Run 15 different configurations