import os
import sys
import argparse
import yaml
import time
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    
    if not hasattr(config_obj, 'feature_set'):
        setattr(config_obj, 'feature_set', ['LBs', 'finished_mark', 'normalized_weights'])
    
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
    """Load validation data for consistent evaluation."""
    try:
        validation_path = f'./data/instances/variable_weights/validation_data_{config.n_j}x{config.n_m}_seed{config.np_seed_validation}.npy'
        dataLoaded = np.load(validation_path)
        vali_data = []
        for i in range(dataLoaded.shape[0]):
            vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))
        print(f"Loaded validation data from {validation_path}")
        return vali_data
    except:
        print(f"Could not load validation data, generating new instances...")
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
        os.makedirs(os.path.dirname(validation_path), exist_ok=True)
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

def plot_performance_profile(results, methods):
    """Create a performance profile plot for WandB logging."""
    # Calculate performance ratios
    best_values = []
    for i in range(len(results[methods[0]])):
        best_value = min([results[method][i] for method in methods])
        best_values.append(best_value)
    
    ratios = {}
    for method in methods:
        ratios[method] = [results[method][i] / best_values[i] for i in range(len(best_values))]
    
    # Create performance profile plot
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        sorted_ratios = np.sort(ratios[method])
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        plt.step(sorted_ratios, y, where='post', label=method)
    
    plt.title('Performance Profile')
    plt.xlabel('Performance Ratio (τ)')
    plt.ylabel('Probability P(r_{p,s} ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    return plt

def log_learning_curves(rewards, weighted_sums, losses, validation_history=None, validation_points=None):
    """Create learning curve plots for WandB logging."""
    # Training curves
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Rewards
    window_size = min(100, len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    episodes = list(range(len(rewards)))
    smoothed_episodes = list(range(window_size-1, len(rewards)))
    
    axs[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    axs[0].plot(smoothed_episodes, smoothed_rewards, color='blue', label=f'Smoothed (window={window_size})')
    axs[0].set_title('Training Rewards')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Weighted Sums
    smoothed_ws = np.convolve(weighted_sums, np.ones(window_size)/window_size, mode='valid')
    
    axs[1].plot(episodes, weighted_sums, alpha=0.3, color='red', label='Raw')
    axs[1].plot(smoothed_episodes, smoothed_ws, color='red', label=f'Smoothed (window={window_size})')
    
    # Add validation curve if provided
    if validation_history is not None and validation_points is not None:
        axs[1].plot(validation_points, validation_history, 'g-o', label='Validation')
    
    axs[1].set_title('Weighted Sum')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Weighted Sum')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Losses
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    
    axs[2].plot(episodes, losses, alpha=0.3, color='purple', label='Raw')
    axs[2].plot(smoothed_episodes, smoothed_losses, color='purple', label=f'Smoothed (window={window_size})')
    axs[2].set_title('Training Loss')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def log_win_rates_chart(win_rates_history):
    """Create win rates chart for WandB logging."""
    plt.figure(figsize=(10, 6))
    
    episodes = list(range(len(win_rates_history)))
    for method in ['win_vs_spt', 'win_vs_wspt']:
        values = [entry[method] for entry in win_rates_history]
        plt.plot(episodes, values, marker='o', label=f'{method.replace("win_vs_", "").upper()}')
    
    plt.title('Win Rates vs Baselines')
    plt.xlabel('Validation Check')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    return plt

def train(config):
    """Main training function with WandB integration."""
    # Initialize WandB if not disabled
    use_wandb = not config.no_wandb
    if use_wandb:
        run = wandb.init(project="jssp-weighted-sum", config=vars(config))
        # Update config with wandb values if in sweep mode
        if run.config.get('_wandb', {}).get('sweep', {}).get('count', 0) > 0:
            config = update_configs_from_wandb(wandb.config)
            
        # Log configuration parameters
        wandb.config.update(
            {param: getattr(config, param) for param in dir(config) 
             if not param.startswith('_') and not callable(getattr(config, param))}
        )
    
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
    
    # Initialize live plotter if not using WandB
    live_plotter = None
    if not use_wandb:
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        live_plotter = LivePlotter(save_dir=results_dir)
    
    # For saving best model
    best_weighted_sum = float('inf')
    best_win_rate = 0
    no_improvement_counter = 0
    best_episode = 0
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    start_time = time.time()
    print(f"Starting training for {config.max_updates} episodes with {config.num_envs} environments...")
    
    for episode in tqdm(range(config.max_updates), desc="Training episodes"):
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
            wandb.log({
                "episode": episode,
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
            
            # Log detailed validation metrics to WandB
            if use_wandb:
                # Log basic validation metrics
                wandb.log({
                    "validation_weighted_sum": validation_weighted_sum_mean,
                    "validation_improvement_pct": validation_results['improvement_pct'].mean(),
                    "validation_win_rate": comparison_metrics['win_rate'],
                    "validation_win_vs_spt": comparison_metrics['win_vs_spt'],
                    "validation_win_vs_wspt": comparison_metrics['win_vs_wspt'],
                    "validation_improvement_over_spt": comparison_metrics['improvement_over_spt'],
                    "validation_improvement_over_wspt": comparison_metrics['improvement_over_wspt']
                })
                
                # Create and log learning curve plots
                curves_fig = log_learning_curves(
                    rewards_history, 
                    weighted_sums_history, 
                    loss_history,
                    validation_history,
                    validation_points
                )
                wandb.log({"learning_curves": wandb.Image(curves_fig)})
                plt.close(curves_fig)
                
                # Create and log win rates chart
                win_rates_fig = log_win_rates_chart(win_rates_history)
                wandb.log({"win_rates": wandb.Image(win_rates_fig)})
                plt.close(win_rates_fig)
                
                # Create and log a performance profile
                # First, we need to extract the results from our validation
                validation_results_dict = {
                    "L2D": validation_results['weighted_sum'].tolist(),
                    "SPT": [], 
                    "WSPT": []
                }
                
                # We can create a partial performance profile with what we have
                perf_profile_fig = plot_performance_profile(
                    {"L2D": validation_results['weighted_sum']}, 
                    ["L2D"]
                )
                wandb.log({"performance_profile": wandb.Image(perf_profile_fig)})
                plt.close(perf_profile_fig)
            
            # Save model based on weighted sum or win rate
            improvement_metric = comparison_metrics['win_vs_wspt']  # Focus on WSPT which is generally the best baseline
            if improvement_metric > best_win_rate:
                best_win_rate = improvement_metric
                best_episode = episode + 1
                no_improvement_counter = 0
                
                # Save best model
                weight_type = "variable" if config.weight_high > config.weight_low else "uniform"
                best_model_path = os.path.join(model_dir, f"l2d_{weight_type}_{config.n_j}x{config.n_m}_best.pth")
                torch.save(agent.policy.state_dict(), best_model_path)
                print(f"New best model saved! Win rate vs WSPT: {improvement_metric:.2f}%")
                
                if use_wandb:
                    wandb.log({"best_win_rate": best_win_rate, "best_episode": best_episode})
                    wandb.save(best_model_path)
            else:
                no_improvement_counter += 1
                print(f"No improvement for {no_improvement_counter} validation checks (best: {best_win_rate:.2f}%)")
            
            # Save checkpoint every 5 validation checks
            if (episode + 1) % (config.validate_every * 5) == 0:
                checkpoint_path = os.path.join(model_dir, f"l2d_weighted_{config.n_j}x{config.n_m}_episode_{episode+1}.pth")
                torch.save(agent.policy.state_dict(), checkpoint_path)
                if use_wandb:
                    wandb.save(checkpoint_path)
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"l2d_weighted_{config.n_j}x{config.n_m}_final.pth")
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Final validation metrics
    final_validation = validate_weighted(validation_data, agent.policy, config.feature_set)
    final_comparison = compare_validation_methods(validation_data, agent.policy, config.feature_set)
    
    print(f"Final Validation Weighted Sum: {final_validation['weighted_sum'].mean():.2f}")
    print(f"Win rate vs SPT: {final_comparison['win_vs_spt']:.2f}%")
    print(f"Win rate vs WSPT: {final_comparison['win_vs_wspt']:.2f}%")
    
    # Save final plots when training finishes
    if live_plotter is not None:
        live_plotter.save_final_plots()
    
    # Save final metrics to WandB
    if use_wandb:
        # Create final summary performance profile
        final_results = {
            "L2D": final_validation['weighted_sum'].tolist(),
            "SPT": [],  # We'd need to run SPT on all validation instances
            "WSPT": []  # We'd need to run WSPT on all validation instances 
        }
        
        # Log final metrics
        wandb.log({
            "final_validation_weighted_sum": final_validation['weighted_sum'].mean(),
            "final_win_rate": final_comparison['win_rate'],
            "final_win_vs_spt": final_comparison['win_vs_spt'],
            "final_win_vs_wspt": final_comparison['win_vs_wspt'],
            "final_improvement_over_spt": final_comparison['improvement_over_spt'],
            "final_improvement_over_wspt": final_comparison['improvement_over_wspt'],
            "best_win_rate": best_win_rate,
            "best_episode": best_episode,
            "training_duration_hours": hours + minutes/60 + seconds/3600
        })
        
        # Create and log a final summary
        summary = (
            f"Training Summary:\n"
            f"- Problem: {config.n_j}x{config.n_m} JSSP with weighted sum objective\n"
            f"- Episodes: {config.max_updates}\n"
            f"- Feature set: {config.feature_set}\n"
            f"- Reward strategy: {config.reward_strategy}\n"
            f"- Best win rate vs WSPT: {best_win_rate:.2f}% at episode {best_episode}\n"
            f"- Final win rate vs WSPT: {final_comparison['win_vs_wspt']:.2f}%\n"
            f"- Final weighted sum: {final_validation['weighted_sum'].mean():.2f}\n"
            f"- Training duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
        )
        wandb.log({"training_summary": summary})
        
        wandb.finish()
    
    return best_win_rate, best_episode, final_comparison

def main():
    """Main function to parse arguments and run training."""
    args = parse_args()
    
    # Load configuration
    config_dict = None
    if args.config:
        config_dict = load_config(args.config)
    
    # Create configuration object
    config = create_config_object(args, config_dict)
    
    # Run training
    if args.sweep:
        best_win_rate, best_episode, _ = train(config)
        print(f"Sweep run complete. Best win rate: {best_win_rate:.2f}% at episode {best_episode}")
    else:
        best_win_rate, best_episode, final_comparison = train(config)
        print(f"Training complete. Best win rate: {best_win_rate:.2f}% at episode {best_episode}")
        print(f"Final win rates - Overall: {final_comparison['win_rate']:.2f}%, "
              f"vs SPT: {final_comparison['win_vs_spt']:.2f}%, "
              f"vs WSPT: {final_comparison['win_vs_wspt']:.2f}%")

if __name__ == "__main__":
    main()