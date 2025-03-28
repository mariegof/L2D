import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import time
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environments.JSSP_Env import SJSSP
from src.utils.live_visualization import LivePlotter
from src.agents.PPO_jssp import PPO, Memory
from src.utils.mb_agg import g_pool_cal
from src.utils.agent_utils import select_action, eval_actions, greedy_select_action
from src.utils.instance_gen import weighted_instance_gen
from src.utils.validation import validate_weighted, compare_validation_methods
from Params import configs

def parse_args():
    """Parse command line arguments with proper support for config file and WandB flags."""
    # Simply import and use the parser from Params
    from Params import parser
    args = parser.parse_args()
    return args

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_config_object(args, config_dict=None):
    """Create a configuration object by combining command-line args, config file, and defaults."""
    # Start with default configs
    config_obj = configs
    
    # Override with values from config file if provided
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                # Add new attributes that might not be in configs
                setattr(config_obj, key, value)
    
    # Override with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'sweep', 'no_wandb']:
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                # Add new attributes that might not be in configs
                setattr(config_obj, key, value)
    
    # Ensure required attributes are present with default values if needed
    if not hasattr(config_obj, 'validate_every'):
        setattr(config_obj, 'validate_every', 100)
    
    # Process feature_set from string to list if it's a string
    if hasattr(config_obj, 'feature_set') and isinstance(config_obj.feature_set, str):
        if ',' in config_obj.feature_set:
            config_obj.feature_set = [f.strip() for f in config_obj.feature_set.split(',')]
        else:
            # Handle the case where it might be a single string
            config_obj.feature_set = [config_obj.feature_set]
    # The fix: Update input_dim based on feature_set length
    if hasattr(config_obj, 'feature_set'):
        if isinstance(config_obj.feature_set, list):
            setattr(config_obj, 'input_dim', len(config_obj.feature_set))
    
    if not hasattr(config_obj, 'reward_strategy'):
        setattr(config_obj, 'reward_strategy', 'default')
    
    # Add flags for WandB usage
    setattr(config_obj, 'no_wandb', args.no_wandb)
    setattr(config_obj, 'sweep', args.sweep)
    
    return config_obj

def update_configs_from_wandb(wandb_config):
    """Update the global configs object with values from WandB config."""
    for key, value in wandb_config.items():
        if hasattr(configs, key):
            setattr(configs, key, value)
    return configs

def load_validation_data(config):
    """Load validation data for consistent evaluation from the correct data path."""
    # Determine weight type
    weight_type = "variable_weights" if config.weight_high > config.weight_low else "uniform_weights"
    
    # Construct the proper validation path
    validation_path = f'./data/instances/{weight_type}/validation_data_{config.n_j}x{config.n_m}_seed{config.np_seed_validation}.npy'
    
    try:
        print(f"Loading validation data from {validation_path}...")
        dataLoaded = np.load(validation_path)
        vali_data = []
        for i in range(dataLoaded.shape[0]):
            vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))
        print(f"Successfully loaded {len(vali_data)} validation instances")
        return vali_data
    except Exception as e:
        print(f"Error loading validation data: {str(e)}")
        print(f"Generating new validation instances...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(validation_path), exist_ok=True)
        
        np.random.seed(config.np_seed_validation)
        vali_data = []
        for i in range(100):  # Generate 100 validation instances
            instance = weighted_instance_gen(
                n_j=config.n_j, 
                n_m=config.n_m, 
                low=config.low, 
                high=config.high,
                weight_low=config.weight_low, 
                weight_high=config.weight_high
            )
            vali_data.append(instance)
        
        # Save validation data for future use
        validation_array = np.array([np.array([d[0], d[1], d[2]]) for d in vali_data])
        np.save(validation_path, validation_array)
        print(f"Generated and saved validation data to {validation_path}")
        return vali_data

def train_episode(envs, memories, agent, device, g_pool_step, config):
    """Train for one episode across multiple environments."""
    # Reset all environments with new instances
    ep_rewards = [0 for _ in range(len(envs))]
    weighted_sums = [0 for _ in range(len(envs))]
    adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []
    
    for i, env in enumerate(envs):
        # Generate new instance with appropriate weights
        instance = weighted_instance_gen(
            n_j=config.n_j, 
            n_m=config.n_m, 
            low=config.low, 
            high=config.high,
            weight_low=config.weight_low, 
            weight_high=config.weight_high
        )
        
        # Reset environment with the instance
        adj, fea, candidate, mask = env.reset(instance)
        adj_envs.append(adj)
        fea_envs.append(fea)
        candidate_envs.append(candidate)
        mask_envs.append(mask)
        ep_rewards[i] = - env.initQuality  # Negative because it's a minimization problem
    
    # Rollout in all environments
    while True:
        # Prepare tensors for all environments
        fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
        adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
        candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
        mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
        
        # Get actions for all environments
        with torch.no_grad():
            action_envs = []
            a_idx_envs = []
            for i in range(len(envs)):
                pi, _ = agent.policy_old(
                    x=fea_tensor_envs[i],
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor_envs[i],
                    candidate=candidate_tensor_envs[i].unsqueeze(0),
                    mask=mask_tensor_envs[i].unsqueeze(0)
                )
                
                action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                action_envs.append(action)
                a_idx_envs.append(a_idx)
        
        # Store experiences and step all environments
        adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []
        
        for i in range(len(envs)):
            # Store experiences
            memories[i].adj_mb.append(adj_tensor_envs[i])
            memories[i].fea_mb.append(fea_tensor_envs[i])
            memories[i].candidate_mb.append(candidate_tensor_envs[i])
            memories[i].mask_mb.append(mask_tensor_envs[i])
            memories[i].a_mb.append(a_idx_envs[i])
            
            # Take step
            adj, fea, reward, done, candidate, mask = envs[i].step(action_envs[i].item())
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] += reward
            memories[i].r_mb.append(reward)
            memories[i].done_mb.append(done)
        
        # Break if first environment is done
        if envs[0].done():
            break
    
    # Get final results from each environment
    for i in range(len(envs)):
        ep_rewards[i] -= envs[i].posRewards
        weighted_sums[i] = envs[i].weighted_sum
    
    return np.mean(ep_rewards), np.mean(weighted_sums)

def plot_weighted_learning_curves(rewards, losses, weighted_sums, figures_dir, 
                               validation_history=None, validation_rewards=None, validation_losses=None, 
                               log_every=100, current_episode=None):
    """
    Create a single three-panel plot with training and validation metrics.
    
    Args:
        rewards: List of training rewards
        losses: List of training losses
        weighted_sums: List of training weighted sums
        figures_dir: Directory to save figure
        validation_history: List of validation weighted sum means
        validation_rewards: List of validation rewards
        validation_losses: List of validation losses
        log_every: Episodes between validation checks
        current_episode: Current episode number for consistent filenames
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(figures_dir, exist_ok=True)
        
        # Apply smoothing to reduce noise
        window = min(100, max(10, len(rewards) // 10))
        
        # Create a figure with three subplots stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300)
        
        episodes = np.arange(len(rewards))
        
        # Common styling
        line_styles = {
            'train_raw': {'color': 'blue', 'alpha': 0.2, 'linewidth': 1},
            'train_smooth': {'color': 'blue', 'alpha': 1.0, 'linewidth': 2},
            'validation': {'color': 'red', 'marker': 'o', 'markersize': 5, 
                          'linewidth': 2, 'alpha': 0.8}
        }
        
        # 1. Weighted Sum plot (top)
        ax = axes[0]
        
        # Smooth the weighted sums
        if len(weighted_sums) > window:
            smooth_ws = pd.Series(weighted_sums).rolling(window=window, min_periods=1).mean().values
            
            # Plot raw and smoothed training data
            ax.plot(episodes, weighted_sums, label='Training', **line_styles['train_raw'])
            ax.plot(episodes, smooth_ws, label='Training (smoothed)', **line_styles['train_smooth'])
        else:
            ax.plot(episodes, weighted_sums, label='Training', **line_styles['train_smooth'])
        
        # Add validation data if available
        if validation_history and len(validation_history) > 0:
            val_steps = np.array([(i+1) * log_every for i in range(len(validation_history))])
            ax.plot(val_steps, validation_history, label='Validation', **line_styles['validation'])
            
            # Mark best validation point
            best_val = min(validation_history)
            best_idx = validation_history.index(best_val)
            ax.plot(val_steps[best_idx], best_val, 'ro', markersize=8)
            ax.text(val_steps[best_idx], best_val*0.97, f'Best: {best_val:.1f}', 
                   fontsize=10, ha='center')
        
        ax.set_title('Weighted Sum', fontsize=14)
        ax.set_ylabel('Weighted Sum')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 2. Loss plot (middle)
        ax = axes[1]
        
        # Smooth the losses
        if len(losses) > window:
            smooth_losses = pd.Series(losses).rolling(window=window, min_periods=1).mean().values
            
            # Plot raw and smoothed training data
            ax.plot(episodes, losses, label='Training', **line_styles['train_raw'])
            ax.plot(episodes, smooth_losses, label='Training (smoothed)', **line_styles['train_smooth'])
        else:
            ax.plot(episodes, losses, label='Training', **line_styles['train_smooth'])
        
        # Add validation data if available
        if validation_losses and len(validation_losses) > 0:
            val_steps = np.array([(i+1) * log_every for i in range(len(validation_losses))])
            ax.plot(val_steps, validation_losses, label='Validation', **line_styles['validation'])
        
        ax.set_title('Loss', fontsize=14)
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 3. Reward plot (bottom)
        ax = axes[2]
        
        # Smooth the rewards
        if len(rewards) > window:
            smooth_rewards = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
            
            # Plot raw and smoothed training data
            ax.plot(episodes, rewards, label='Training', **line_styles['train_raw'])
            ax.plot(episodes, smooth_rewards, label='Training (smoothed)', **line_styles['train_smooth'])
        else:
            ax.plot(episodes, rewards, label='Training', **line_styles['train_smooth'])
        
        # Add validation data if available
        if validation_rewards and len(validation_rewards) > 0:
            val_steps = np.array([(i+1) * log_every for i in range(len(validation_rewards))])
            ax.plot(val_steps, validation_rewards, label='Validation', **line_styles['validation'])
        
        ax.set_title('Reward', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add overall title 
        fig.suptitle(f'Training Progress - Episodes: {len(rewards)}', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Use episode number for filenames instead of timestamps (to ensure consistency)
        episode_str = str(current_episode) if current_episode is not None else str(len(rewards))
        unique_filename = os.path.join(figures_dir, f"learning_curves_{episode_str}.png")
        standard_filename = os.path.join(figures_dir, "learning_curves.png")
        
        # Use a BytesIO buffer to ensure atomic write
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Write buffer to file
        with open(unique_filename, 'wb') as f:
            f.write(buf.getbuffer())
            
        # Also save to standard filename (for WandB logging)
        with open(standard_filename, 'wb') as f:
            buf.seek(0)  # Reset buffer position
            f.write(buf.getbuffer())
            
        plt.close(fig)
        
        print(f"Updated learning curves plot at episode {episode_str}: {standard_filename} and {unique_filename}")
        return True, unique_filename, standard_filename
        
    except Exception as e:
        print(f"Error creating learning curves plot: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def train(config):
    """Main training function with WandB integration."""
    # Initialize wandb with proper error handling
    use_wandb = not config.no_wandb
    run = None
    try:
        if use_wandb:
            # Prepare a clean config dictionary for WandB
            wandb_config = {}
            for k, v in vars(config).items():
                if not k.startswith('_') and not callable(v):
                    # Convert non-serializable types
                    if isinstance(v, np.ndarray):
                        wandb_config[k] = v.tolist()
                    elif isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
                        wandb_config[k] = v
                    else:
                        wandb_config[k] = str(v)
            
            # For wandb 0.19.8, we don't need to call login() explicitly
            # Just use init() directly - the agent takes care of connecting to the right run
            run = wandb.init(config=wandb_config)
            
            # You can still add problem dimensions to the run name
            if wandb.run and wandb.run.name:
                wandb.run.name = f"{wandb.run.name}_{config.n_j}x{config.n_m}"
                # In wandb 0.19.8, we don't need to explicitly call save()
            
            # Update config with wandb values if in sweep mode
            if hasattr(wandb, 'config'):
                for key, value in wandb.config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
            # Debug info about training limits
            print(f"WandB initialized. Will train for exactly {config.max_updates} episodes.")
    except Exception as e:
        print(f"Error initializing WandB: {e}")
        use_wandb = False
    
    try:
        # Set device
        device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        np.random.seed(config.np_seed_train)
        
        # Initialize environments
        envs = [SJSSP(n_j=config.n_j, n_m=config.n_m, feature_set=config.feature_set, 
                     reward_strategy=config.reward_strategy) for _ in range(config.num_envs)]
        
        # Initialize memories for PPO
        memories = [Memory() for _ in range(config.num_envs)]
        
        # Initialize PPO agent
        agent = PPO(
            lr=config.lr, 
            gamma=config.gamma, 
            k_epochs=config.k_epochs, 
            eps_clip=config.eps_clip,
            n_j=config.n_j,
            n_m=config.n_m,
            num_layers=config.num_layers,
            neighbor_pooling_type=config.neighbor_pooling_type,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_mlp_layers_feature_extract=config.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=config.num_mlp_layers_actor,
            hidden_dim_actor=config.hidden_dim_actor,
            num_mlp_layers_critic=config.num_mlp_layers_critic,
            hidden_dim_critic=config.hidden_dim_critic
        )
        
        # Calculate graph pooling setup
        g_pool_step = g_pool_cal(
            graph_pool_type=config.graph_pool_type,
            batch_size=torch.Size([1, config.n_j*config.n_m, config.n_j*config.n_m]),
            n_nodes=config.n_j*config.n_m,
            device=device
        )
        
        # Load validation data
        validation_data = load_validation_data(config)
        
        # Initialize tracking metrics
        rewards_history = []
        weighted_sums_history = []
        loss_history = []
        validation_history = []
        validation_points = []
        win_rates_history = []
        validation_results_history = []  # Will store full validation results
        
        # Initialize live plotter if not using WandB
        live_plotter = None
        if not use_wandb:
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            live_plotter = LivePlotter(save_dir=results_dir)
        
        # For saving best model - now prioritizing weighted sum (lower is better)
        best_weighted_sum = float('inf')
        best_win_rate = 0  # Still tracking for reference
        no_improvement_counter = 0
        best_episode = 0
        model_dir = "./models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        print(f"Starting training for {config.max_updates} episodes with {config.num_envs} environments...")
        
        # Use only one progress bar to prevent duplication
        with tqdm(total=config.max_updates, desc="Training episodes", disable=use_wandb) as pbar:
            for episode in range(config.max_updates):
                episode_start = time.time()
                
                # Train for one episode
                reward, weighted_sum = train_episode(envs, memories, agent, device, g_pool_step, config)
                
                # Update policy using PPO
                loss, v_loss, p_loss, ent_loss = agent.update(memories, config.n_j * config.n_m, config.graph_pool_type, return_all_losses=True)
                for memory in memories:
                    memory.clear_memory()
                
                # Save history
                rewards_history.append(reward)
                weighted_sums_history.append(weighted_sum)
                loss_history.append(loss)
                
                # Update live plots if not using WandB
                if live_plotter is not None and not use_wandb:
                    live_plotter.update(episode, reward, weighted_sum, loss)
                
                # Log basic metrics to WandB
                if use_wandb:
                    # For wandb 0.19.8, we don't need to specify the step explicitly
                    wandb.log({
                        "episode": episode + 1,
                        "reward": reward,
                        "weighted_sum": weighted_sum,
                        "loss": loss,
                        "value_loss": v_loss,
                        "policy_loss": p_loss,
                        "entropy_loss": ent_loss,
                        "learning_rate": agent.scheduler.get_last_lr()[0]
                    })
                
                # Validation and extensive logging
                if (episode + 1) % config.validate_every == 0:
                    elapsed = time.time() - start_time
                    avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                    avg_weighted_sum = np.mean(weighted_sums_history[-100:]) if len(weighted_sums_history) >= 100 else np.mean(weighted_sums_history)
                    
                    print(f"Episode {episode+1}/{config.max_updates} | Reward: {reward:.2f} | "
                        f"Weighted Sum: {weighted_sum:.2f} | Loss: {loss:.6f}")
                    print(f"Avg(100): Reward={avg_reward:.2f}, Weighted Sum={avg_weighted_sum:.2f} | "
                        f"Episode time: {time.time()-episode_start:.2f}s | Total time: {elapsed:.2f}s")
                    
                    # Run validation
                    validation_results = validate_weighted(validation_data, agent.policy, config.feature_set)
                    validation_results_history.append(validation_results)
                    validation_weighted_sum_mean = validation_results['weighted_sum'].mean()
                    validation_history.append(validation_weighted_sum_mean)
                    validation_points.append(episode + 1)
                    
                    # Update live plotter with validation result
                    if live_plotter is not None:
                        live_plotter.update(
                            episode, reward, weighted_sum, loss,
                            validation_episode=episode+1, 
                            validation_weighted_sum=validation_weighted_sum_mean
                        )
                    
                    print(f"Validation Weighted Sum: {validation_weighted_sum_mean:.2f} | "
                        f"Reward-derived: {validation_results['reward_derived'].mean():.2f} | "
                        f"Improvement: {validation_results['improvement_pct'].mean():.2f}%")
                    
                    # Compare with baselines
                    comparison_metrics = compare_validation_methods(validation_data, agent.policy, config.feature_set)
                    win_rates_history.append(comparison_metrics)
                    
                    # Create/update learning curve plots after each validation
                    figures_dir = os.path.join(model_dir, "figures")
                    os.makedirs(figures_dir, exist_ok=True)
                    
                    # Extract validation rewards and losses
                    validation_rewards = []
                    validation_losses = []
                    
                    for results in validation_results_history:
                        # Validation reward is negative of weighted sum (since we're minimizing weighted sum)
                        validation_rewards.append(-results.get('weighted_sum').mean())
                        
                        # Extract value loss if available
                        if 'value_loss' in results:
                            validation_losses.append(results.get('value_loss'))
                    
                    # Update the plots with latest data
                    plot_success, unique_path, standard_path = plot_weighted_learning_curves(
                        rewards_history, 
                        loss_history, 
                        weighted_sums_history, 
                        figures_dir,
                        validation_history=validation_history,
                        validation_rewards=validation_rewards if validation_rewards else None,
                        validation_losses=validation_losses if validation_losses else None,
                        log_every=config.validate_every,
                        current_episode=episode+1
                    )
                    
                    # Create standardized metric dictionary for WandB
                    validation_metrics = {
                        "validation_weighted_sum": validation_weighted_sum_mean,
                        "validation_improvement_pct": validation_results['improvement_pct'].mean(),
                        "validation_win_rate": comparison_metrics['win_rate'],
                        "validation_win_vs_spt": comparison_metrics['win_vs_spt'],
                        "validation_win_vs_wspt": comparison_metrics['win_vs_wspt'],
                        "validation_win_vs_srpt": comparison_metrics['win_vs_srpt'],
                        "validation_improvement_over_spt": comparison_metrics['improvement_over_spt'],
                        "validation_improvement_over_wspt": comparison_metrics['improvement_over_wspt'],
                        "validation_improvement_over_srpt": comparison_metrics['improvement_over_srpt'] 
                    }
                    
                    # Log validation metrics and plot to WandB
                    if use_wandb:
                        # For wandb 0.19.8, we don't need to specify the step explicitly
                        wandb.log(validation_metrics)
                        
                        # Also add metrics to summary for sweep to find them
                        if hasattr(wandb, 'summary'):
                            for key, value in validation_metrics.items():
                                wandb.summary[key] = value
                        
                        # Log the main comparison plot to WandB
                        if plot_success and os.path.exists(standard_path):
                            try:
                                wandb.log({"learning_curves": wandb.Image(standard_path)})
                                print(f"Successfully logged learning curves to WandB")
                            except Exception as e:
                                print(f"Error logging plot to WandB: {e}")
                    
                    # CHANGED: Save model based on weighted sum instead of win rate
                    # Lower weighted sum is better, so we check if it's less than the best so far
                    if validation_weighted_sum_mean < best_weighted_sum:
                        best_weighted_sum = validation_weighted_sum_mean
                        best_win_rate = comparison_metrics['win_rate']  # Still track win rate for reporting
                        best_episode = episode + 1
                        no_improvement_counter = 0
                        
                        # Save best model
                        weight_type = "variable" if config.weight_high > config.weight_low else "uniform"
                        best_model_path = os.path.join(model_dir, f"l2d_{weight_type}_{config.n_j}x{config.n_m}_best.pth")
                        torch.save(agent.policy.state_dict(), best_model_path)
                        print(f"New best model saved! Weighted Sum: {best_weighted_sum:.2f}, Win Rate: {best_win_rate:.2f}%")
                        
                        if use_wandb:
                            # For wandb 0.19.8, log best metrics found so far
                            wandb.log({
                                "best_weighted_sum": best_weighted_sum,
                                "best_win_rate": best_win_rate, 
                                "best_episode": best_episode
                            })
                            
                            # Also update the summary with these values
                            if hasattr(wandb, 'summary'):
                                wandb.summary['best_weighted_sum'] = best_weighted_sum
                                wandb.summary['best_win_rate'] = best_win_rate
                                wandb.summary['best_episode'] = best_episode
                            
                            # Save model as WandB artifact
                            model_artifact = wandb.Artifact(
                                f"model_{config.n_j}x{config.n_m}", 
                                type="model",
                                description=f"Best model for {config.n_j}x{config.n_m} with weighted sum {best_weighted_sum:.2f}"
                            )
                            model_artifact.add_file(best_model_path)
                            wandb.log_artifact(model_artifact)
                    else:
                        no_improvement_counter += 1
                        print(f"No improvement for {no_improvement_counter} validation checks (best weighted sum: {best_weighted_sum:.2f})")
                    
                    # Save checkpoint every 5 validation checks
                    if (episode + 1) % (config.validate_every * 5) == 0:
                        checkpoint_path = os.path.join(model_dir, f"l2d_weighted_{config.n_j}x{config.n_m}_episode_{episode+1}.pth")
                        torch.save(agent.policy.state_dict(), checkpoint_path)
                        if use_wandb:
                            # Log checkpoint as artifact
                            checkpoint_artifact = wandb.Artifact(
                                f"checkpoint_{config.n_j}x{config.n_m}_ep{episode+1}", 
                                type="checkpoint",
                                description=f"Checkpoint at episode {episode+1}"
                            )
                            checkpoint_artifact.add_file(checkpoint_path)
                            wandb.log_artifact(checkpoint_artifact)
                            
                # Add informative stats to the progress bar
                pbar.set_postfix({
                    'reward': f"{reward:.2f}",
                    'weighted_sum': f"{weighted_sum:.2f}"
                })
                pbar.update(1)
        
        # Training complete
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Reached exactly {config.max_updates} episodes as configured")
        
        # Save final model
        final_model_path = os.path.join(model_dir, f"l2d_weighted_{config.n_j}x{config.n_m}_final.pth")
        torch.save(agent.policy.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save final plots when training finishes
        if live_plotter is not None:
            live_plotter.save_final_plots()
        
        # Save final metrics to WandB
        if use_wandb and wandb.run is not None:
            try:
                # Use metrics associated with the best model (lowest weighted sum)
                best_validation_idx = np.argmin([r['weighted_sum'].mean() for r in validation_results_history]) if validation_results_history else -1
                best_validation_results = validation_results_history[best_validation_idx] if best_validation_idx >= 0 and validation_results_history else None
                best_comparison_metrics = win_rates_history[best_validation_idx] if best_validation_idx >= 0 else None
                
                # Create standard final metrics dict using the best validation results
                final_metrics = {
                    "final_validation_weighted_sum": best_weighted_sum,
                    "final_win_rate": best_win_rate,
                    "final_win_vs_spt": best_comparison_metrics['win_vs_spt'] if best_comparison_metrics else 0,
                    "final_win_vs_wspt": best_comparison_metrics['win_vs_wspt'] if best_comparison_metrics else 0,
                    "final_win_vs_srpt": best_comparison_metrics['win_vs_srpt'] if best_comparison_metrics else 0,
                    "final_improvement_over_spt": best_comparison_metrics['improvement_over_spt'] if best_comparison_metrics else 0,
                    "final_improvement_over_wspt": best_comparison_metrics['improvement_over_wspt'] if best_comparison_metrics else 0,
                    "final_improvement_over_srpt": best_comparison_metrics['improvement_over_srpt'] if best_comparison_metrics else 0,
                    "best_weighted_sum": best_weighted_sum,
                    "best_win_rate": best_win_rate,
                    "best_episode": best_episode,
                    "training_duration_hours": hours + minutes/60 + seconds/3600,
                    "final_episode": config.max_updates
                }
                
                # For wandb 0.19.8, we don't need to specify the step explicitly
                wandb.log(final_metrics)
                
                # Save final model as artifact
                final_model_artifact = wandb.Artifact(
                    f"final_model_{config.n_j}x{config.n_m}", 
                    type="model",
                    description=f"Final model for {config.n_j}x{config.n_m}"
                )
                final_model_artifact.add_file(final_model_path)
                wandb.log_artifact(final_model_artifact)
                
                # Create and log a final summary using best validation results
                summary = (
                    f"Training Summary:\n"
                    f"- Problem: {config.n_j}x{config.n_m} JSSP with weighted sum objective\n"
                    f"- Episodes: {config.max_updates}\n"
                    f"- Feature set: {config.feature_set}\n"
                    f"- Reward strategy: {config.reward_strategy}\n"
                    f"- Best weighted sum: {best_weighted_sum:.2f} at episode {best_episode}\n"
                    f"- Best win rate: {best_win_rate:.2f}%\n"
                    f"- Training duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
                )
                
                # Set summary directly with wandb.summary for wandb 0.19.8
                if hasattr(wandb, 'summary'):
                    wandb.summary['training_summary'] = summary
                    
                    # Set config values
                    for key, value in wandb_config.items():
                        if isinstance(value, (int, float, str, bool)):
                            wandb.summary[f"config_{key}"] = value
                    
                    # Set metric values
                    for key, value in final_metrics.items():
                        wandb.summary[key] = value
                
                # Small delay to ensure sync completes
                time.sleep(2)
                
                # Ensure proper WandB finish for wandb 0.19.8
                wandb.finish()
                
            except Exception as e:
                print(f"Error during WandB finalization: {e}")
                if wandb.run is not None:
                    try:
                        wandb.finish(exit_code=0)  # Try to finish properly despite the error
                    except:
                        pass
        
        # Return best metrics for external use
        return best_weighted_sum, best_episode, best_win_rate
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure WandB is properly closed even on error
        if use_wandb and wandb.run is not None:
            try:
                wandb.finish(exit_code=1)
            except:
                pass
        
        # Re-raise the error to ensure it's properly handled by the calling code
        raise e

def main():
    """Main function to parse arguments and run training."""
    args = parse_args()
    
    # Load configuration
    config_dict = None
    if args.config:
        config_dict = load_config(args.config)
    
    # Create configuration object
    config = create_config_object(args, config_dict)
    
    # Print clear information about training limits
    print(f"Configuration loaded - will train for {config.max_updates} episodes")
    print(f"Validation will happen every {config.validate_every} episodes")
    
    # Run training
    start_time = time.time()
    if args.sweep:
        # Updated to match new train() return values: best_weighted_sum, best_episode, best_win_rate
        best_weighted_sum, best_episode, best_win_rate = train(config)
        total_time = time.time() - start_time
        
        # Format time nicely
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nSweep run complete after {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Completed exactly {config.max_updates} episodes")
        print(f"Best weighted sum: {best_weighted_sum:.2f} at episode {best_episode}")
        print(f"Best win rate: {best_win_rate:.2f}%")
    else:
        # Updated to match new train() return values: best_weighted_sum, best_episode, best_win_rate
        best_weighted_sum, best_episode, best_win_rate = train(config)
        total_time = time.time() - start_time
        
        # Format time nicely
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining complete after {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Completed exactly {config.max_updates} episodes")
        print(f"Best weighted sum: {best_weighted_sum:.2f} at episode {best_episode}")
        print(f"Best win rate: {best_win_rate:.2f}%")
        
        # Since we no longer return the full comparison dictionary, we can't print these details
        # You could run a final validation here if these metrics are important to show
        print("\nNote: For detailed comparison metrics against baselines,")
        print("check the wandb dashboard or model checkpoint directory.")

if __name__ == "__main__":
    main()