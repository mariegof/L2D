# train_weighted_objective.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from the original implementation
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO, Memory
from mb_agg import g_pool_cal
from agent_utils import select_action
from uniform_instance_gen import weighted_instance_gen
from Params import configs
from LogExpResults import log_experiment_results
from WS.ws_validation import validate_weighted, compare_validation_methods
from WS.ws_eval import evaluate_policies
from WS.utils import setup_directories 

# Configuration parameters
CONFIG = {
    "training": {
        "enabled": True,
        "max_updates": 100,     # Number of episodes for training
        "save_every": 1000,      # Save model checkpoints every N episodes
        "log_every": 1,        # Log training statistics every N episodes
        'wspt_guidance_duration': 0
    },
    "validation": {
        "seed": 200,
    },
    "evaluation": {
        "enabled": True,
        "n_instances": 100,      # Number of test instances
        "use_trained_model": True,
        "model_path": None,      # Path to pre-trained model (if use_trained_model=False)
    },
    "output_dir": "./weighted_objective_experiments"  # Directory for saving results
}
device = torch.device(configs.device)

# Get configuration parameters
n_j = configs.n_j
n_m = configs.n_m
weight_low = configs.weight_low
weight_high = configs.weight_high
n_episodes = CONFIG["training"]["max_updates"]
save_every = CONFIG["training"]["save_every"]
log_every = CONFIG["training"]["log_every"]
wspt_guidance_duration = CONFIG["training"]['wspt_guidance_duration']
num_envs = configs.num_envs
feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'machine_contention', 'remaining_weighted_work', 'time_elapsed']

def train_l2d_multi_env():
    """
    Train L2D model for weighted sum objective using multiple environments,
    matching the approach in PPO_jssp_multiInstances.py.
    """
    print(f"\n{'='*50}")
    print(f"TRAINING L2D FOR WEIGHTED SUM OBJECTIVE ({n_j}x{n_m})")
    print(f"{'='*50}")
    
    # Setup directories
    models_dir, figures_dir, results_dir = setup_directories(CONFIG)
    
    # Initialize multiple environments
    envs = [SJSSP(n_j=n_j, n_m=n_m, feature_set=feature_set) for _ in range(num_envs)]
    
    # Load validation data
    dataLoaded = np.load('./DataGen/WeightData/weightedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))
        
    # Set seeds at the beginning of training
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)
    
    # Initialize memories for PPO
    memories = [Memory() for _ in range(configs.num_envs)]
    
    # Initialize PPO agent with correct input dimension
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,
              n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    
    # Calculate graph pooling setup
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
                            n_nodes=configs.n_j*configs.n_m,
                            device=device)
    
    # For tracking training progress
    rewards_history = []
    weighted_sums_history = []
    loss_history = []
    validation_history = []
        
    # For saving best model
    best_weighted_sum = float('inf')
    
    # Training loop
    start_time = time.time()
    print(f"Starting training for {n_episodes} episodes with {num_envs} environments...")
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Reset all environments with new instances
        ep_rewards = [0 for _ in range(num_envs)]
        weighted_sums = [0 for _ in range(num_envs)]
        adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []
        
        for i, env in enumerate(envs):
            # Generate weighted instances
            adj, fea, candidate, mask = env.reset(weighted_instance_gen(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high,weight_low=weight_low, weight_high=weight_high))
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality  # Negative because it's a minimization problem
        
        # Rollout in all environments
        while True:
            # Prepare tensors for all environments
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(torch.device(configs.device)) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(torch.device(configs.device)).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(torch.device(configs.device)) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(torch.device(configs.device)) for mask in mask_envs]
            
            # Get actions for all environments
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(num_envs):
                    pi, _ = ppo.policy_old(
                        x=fea_tensor_envs[i],
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor_envs[i],
                        candidate=candidate_tensor_envs[i].unsqueeze(0),
                        mask=mask_tensor_envs[i].unsqueeze(0)
                    )
                    
                    # Implement WSPT guidance in early training
                    if episode < n_episodes * wspt_guidance_duration:  # First 20% of training
                        wspt_influence = 0.5 * (1.0 - episode / (n_episodes * 0.2))
                        
                        # Calculate WSPT priorities
                        wspt_values = np.zeros(len(candidate_envs[i]))
                        for j, op in enumerate(candidate_envs[i]):
                            if not mask_envs[i][j]:
                                job_idx = op // n_m
                                op_idx = op % n_m
                                wspt_values[j] = envs[i].weights[job_idx] / envs[i].dur[job_idx, op_idx]
                        
                        # Normalize
                        if wspt_values.sum() > 0:
                            wspt_values = wspt_values / wspt_values.sum()
                        
                        # Blend policy with WSPT
                        pi_numpy = pi.detach().cpu().numpy().squeeze()
                        pi_final = (1 - wspt_influence) * pi_numpy + wspt_influence * wspt_values
                        pi = torch.tensor(pi_final).unsqueeze(0).to(torch.device(configs.device))
                    
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            
            # Store experiences and step all environments
            adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []
            
            for i in range(num_envs):
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
        for i in range(num_envs):
            ep_rewards[i] -= envs[i].posRewards
            weighted_sums[i] = envs[i].weighted_sum
        
        # Update policy using PPO
        loss, v_loss = ppo.update(memories, n_j * n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        
        # Save history
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        mean_weighted_sum = sum(weighted_sums) / len(weighted_sums)
        rewards_history.append(mean_reward)
        weighted_sums_history.append(mean_weighted_sum)
        loss_history.append(loss)
        
        # Validation and logging
        if (episode + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            avg_weighted_sum = np.mean(weighted_sums_history[-100:]) if len(weighted_sums_history) >= 100 else np.mean(weighted_sums_history)
            
            print(f"Episode {episode+1}/{n_episodes} | Reward: {mean_reward:.2f} | "
                  f"Weighted Sum: {mean_weighted_sum:.2f} | Loss: {loss:.6f}")
            print(f"Avg(100): Reward={avg_reward:.2f}, Weighted Sum={avg_weighted_sum:.2f} | "
                  f"Episode time: {time.time()-episode_start:.2f}s | Total time: {elapsed:.2f}s")
            
            # Run validation with our new function
            validation_results = validate_weighted(vali_data, ppo.policy, feature_set)
            validation_weighted_sum_mean = validation_results['weighted_sum'].mean()
            validation_history.append(validation_weighted_sum_mean)
            
            # Print consistent with how we're storing it (already negative)
            print(f"Validation Weighted Sum: {validation_weighted_sum_mean:.2f} | "
                f"Reward-derived: {validation_results['reward_derived'].mean():.2f} | "
                f"Improvement: {validation_results['improvement_pct'].mean():.2f}%")
            
            # Compare with baselines occasionally
            if (episode + 1) % (log_every * 5) == 0:
                comparison_metrics = compare_validation_methods(vali_data[:10], ppo.policy, feature_set=feature_set)

            # Save training statistics
            np.savez(
                os.path.join(results_dir, "training_stats.npz"),
                rewards=np.array(rewards_history),
                weighted_sums=np.array(weighted_sums_history),
                losses=np.array(loss_history),
                validation=np.array(validation_history)
            )
            
            # Plot learning curves using our improved plotting function
            if len(rewards_history) >= 1:
                 plot_weighted_learning_curves(
                    rewards=rewards_history, 
                    losses=loss_history, 
                    weighted_sums=weighted_sums_history,  # Correctly named parameter
                    figures_dir=figures_dir,
                    validation_history=validation_history,  # Pass the full history
                    log_every=log_every
                )
            
            # Save model if it's the best so far (smaller weighted sum is better)
            if validation_weighted_sum_mean < best_weighted_sum:
                best_model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_best.pth")
                torch.save(ppo.policy.state_dict(), best_model_path)
                print(f"New best model saved! Weighted sum: {-validation_weighted_sum_mean:.2f}")
                best_weighted_sum = validation_weighted_sum_mean
                best_episode = episode + 1  # Store the episode number of best performance
        
        # Save regular checkpoint
        if (episode + 1) % save_every == 0:
            model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_episode_{episode+1}.pth")
            torch.save(ppo.policy.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_final.pth")
    torch.save(ppo.policy.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Prepare comprehensive training results
    training_results = {
        'duration_minutes': (time.time() - start_time) / 60,
        'best_validation_episode': best_episode if 'best_episode' in locals() else None,
        'validation_weighted_sum': best_weighted_sum,
        'training_rewards': rewards_history,
        'training_weighted_sums': weighted_sums_history,
        'losses': loss_history,
        'validation_history': validation_history
    }
    
    # Add baseline comparison metrics to the results
    training_results.update(comparison_metrics)
    
    return ppo.policy, final_model_path, weighted_sums_history, validation_history, training_results

def plot_weighted_learning_curves(rewards, losses, weighted_sums, figures_dir, validation_history=None, log_every=100):
    """
    Plot and save comprehensive training curves for weighted sum objective with training vs validation comparisons.
    
    Args:
        rewards: List of training rewards
        losses: List of training losses
        weighted_sums: List of training weighted sums
        figures_dir: Directory to save figures
        validation_history: List of historical validation weighted sum means (one value per validation)
        log_every: Number of episodes between validation checks (for x-axis alignment)
    """
    window = 100  # For smoothing
    
    # Data validation and debug info
    print(f"Data lengths - Rewards: {len(rewards)}, Losses: {len(losses)}, Weighted Sums: {len(weighted_sums)}")
    if validation_history is not None:
        print(f"Validation history length: {len(validation_history)}")
        print(f"First few validation values: {validation_history[:5]}")
        print(f"Last few validation values: {validation_history[-5:]} (smaller is better)")
    
    # Calculate moving averages for smoothing
    reward_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    weighted_sum_avg = [np.mean(weighted_sums[max(0, i-window):i+1]) for i in range(len(weighted_sums))]
    loss_avg = [np.mean(losses[max(0, i-window):i+1]) for i in range(len(losses))]
    
    episodes = list(range(len(weighted_sums)))
    
    # 1. Main objective comparison plot (Training vs Validation)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, weighted_sums, 'b-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, weighted_sum_avg, 'b-', linewidth=2, label='Training (smoothed)')
    
    # Plot validation history with proper x-axis alignment
    if validation_history is not None and len(validation_history) > 0:
        # Generate proper x-positions based on logging interval
        val_episodes = [(i+1) * log_every for i in range(len(validation_history))]
        
        # Plot the validation history
        plt.plot(val_episodes, validation_history, 'r-o', linewidth=2, label='Validation')
        
        # Add best training and validation lines
        best_training = min(weighted_sum_avg)
        best_validation = min(validation_history)
        plt.axhline(y=best_training, color='b', linestyle='--', alpha=0.5, 
                    label=f'Best training: {best_training:.0f}')
        plt.axhline(y=best_validation, color='r', linestyle='--', alpha=0.5, 
                    label=f'Best validation: {best_validation:.0f}')
    
    plt.title('Weighted Sum Comparison: Training vs Validation')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Sum')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "weighted_sum_comparison.png"))
    plt.close()
    
    # 2. Loss plot (Training only)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, 'g-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, loss_avg, 'g-', linewidth=2, label='Training (smoothed)')
    
    plt.title('Loss During Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "loss_comparison.png"))
    plt.close()
    
    # 3. Reward plot (Training only)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, 'm-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, reward_avg, 'm-', linewidth=2, label='Training (smoothed)')
    
    plt.title('Reward During Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "reward_comparison.png"))
    plt.close()
    
    # 4. Smoothed weighted sums plot
    plt.figure(figsize=(10, 6))
    plt.plot(weighted_sum_avg)
    plt.title(f'Weighted Sum During Training ({window}-episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Weighted Sum')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "weighted_sums_smoothed.png"))
    plt.close()
    
    # 5. Raw weighted sums (last 1000 episodes for detail)
    plt.figure(figsize=(10, 6))
    if len(weighted_sums) > 1000:
        plt.plot(episodes[-1000:], weighted_sums[-1000:])
        plt.title('Weighted Sum (Last 1000 Episodes)')
    else:
        plt.plot(episodes, weighted_sums)
        plt.title('Weighted Sum (All Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Sum')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "weighted_sums_raw.png"))
    plt.close()
    
    # 6. Validation progress curve
    if validation_history is not None and len(validation_history) > 0:
        plt.figure(figsize=(10, 6))
        val_episodes = [(i+1) * log_every for i in range(len(validation_history))]
        plt.plot(val_episodes, validation_history, 'r-o')
        
        # Add horizontal line for best validation
        best_val = min(validation_history)
        best_idx = validation_history.index(best_val)
        plt.axhline(y=best_val, color='g', linestyle='--', 
                    label=f'Best: {best_val:.1f} at episode {(best_idx+1)*log_every}')
        
        plt.title('Validation Weighted Sum During Training')
        plt.xlabel('Validation Check (Episode)')
        plt.ylabel('Weighted Sum')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "validation_curve.png"))
        plt.close()
        
        # 7. Zoomed validation curve (last 20 validations)
        if len(validation_history) > 10:
            plt.figure(figsize=(10, 6))
            last_n = min(20, len(validation_history))
            plt.plot(val_episodes[-last_n:], validation_history[-last_n:], 'r-o')
            plt.title(f'Recent Validation Weighted Sum (Last {last_n} Checks)')
            plt.xlabel('Validation Check (Episode)')
            plt.ylabel('Weighted Sum')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(figures_dir, "recent_validation_curve.png"))
            plt.close()

if __name__ == "__main__":
    # Force CPU for consistency
    configs.device = "cpu"
    experiment_start_time = time.time()
    # Initialize metadata with automatically extracted values from CONFIG
    metadata = {
        'start_time': experiment_start_time,
        'feature_set': feature_set,
        'wspt_guidance_enabled': True if wspt_guidance_duration > 0 else False,
        'wspt_guidance_duration': wspt_guidance_duration,
        'n_j': n_j,
        'n_m': n_m,
        'num_envs': num_envs,
        'max_updates': n_episodes,
        'device': configs.device
    }
    
    policy = None
    model_path = None
    collected_results = {}
    
    # Run training if enabled
    if CONFIG["training"]["enabled"]:
        policy, model_path, weighted_sums, validation_history, training_results = train_l2d_multi_env()
        
        # Add all training and validation results to our collection
        collected_results.update(training_results)
        
        # Create a training phase marker for logging
        collected_results['training_completed'] = True
    
    # Run evaluation if enabled
    if CONFIG["evaluation"]["enabled"]:
        if not CONFIG["evaluation"]["use_trained_model"]:
            policy = None
            model_path = CONFIG["evaluation"]["model_path"]
        
        # Get evaluation results and add them to our collection
        results, ratios, evaluation_results = evaluate_policies(policy, model_path, conf=CONFIG, feature_set=feature_set)
    
        # Add testing results with distinct prefixes to avoid conflicts
        for key, value in evaluation_results.items():
            # Skip detailed results to avoid bloating the CSV
            if key != 'detailed_results' and key != 'win_counts':
                collected_results[f'test_{key}'] = value
        
        # Create a testing phase marker for logging
        collected_results['testing_completed'] = True
    
    # Calculate total experiment duration and update metadata
    metadata['duration_minutes'] = (time.time() - experiment_start_time) / 60
    
    # Log the experiment results with all collected metrics
    feature_set = metadata['feature_set']
    log_experiment_results(
        configs=configs, 
        results=collected_results,
        feature_set=feature_set,
        metadata=metadata
    )
    
    print("\nWeighted sum objective experiment completed!")
    print(f"Total experiment time: {metadata['duration_minutes']:.2f} minutes")
    print(f"Results logged to experiment_logs.csv")
    print("\nWeighted sum objective experiment completed!")