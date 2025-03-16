import wandb
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original implementation
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO, Memory
from mb_agg import g_pool_cal
from agent_utils import select_action
from uniform_instance_gen import weighted_instance_gen
from Params import configs
from WS.ws_validation import validate_weighted, compare_validation_methods
from WS.utils import setup_directories 

# Configuration parameters
CONFIG = {
    "training": {
        "enabled": True,
        "max_updates": 10000,     # Number of episodes for training
        "save_every": 1000,      # Save model checkpoints every N episodes
        "log_every": 100,        # Log training statistics every N episodes
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
wspt_guidance_duration = CONFIG["training"]['wspt_guidance_duration']
save_every = CONFIG["training"]["save_every"]

def train_l2d_multi_env(wandb_config=None):
    """
    Train L2D model for weighted sum objective using multiple environments,
    matching the approach in PPO_jssp_multiInstances.py.
    """
    # Apply wandb config if provided
    if wandb_config is not None:        
        # Extract parameters from wandb config
        feature_set = wandb_config.feature_set.split(',')
        num_layers = wandb_config.num_layers
        lr = wandb_config.lr
        hidden_dim = wandb_config.hidden_dim
        num_mlp_layers_feature_extract = wandb_config.num_mlp_layers_feature_extract
        num_mlp_layers_actor = wandb_config.num_mlp_layers_actor
        num_mlp_layers_critic = wandb_config.num_mlp_layers_critic
        hidden_dim_actor = wandb_config.hidden_dim_actor
        hidden_dim_critic = wandb_config.hidden_dim_critic
        k_epochs = wandb_config.k_epochs
        eps_clip = wandb_config.eps_clip
        entloss_coef = wandb_config.entloss_coef
        num_envs = wandb_config.num_envs
        n_episodes = wandb_config.max_updates
        log_every = wandb_config.log_every
    else:
        # Use default values if not in sweep
        feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights']
        lr = configs.lr
        hidden_dim = configs.hidden_dim
        hidden_dim_actor = configs.hidden_dim_actor
        hidden_dim_critic = configs.hidden_dim_critic
        k_epochs = configs.k_epochs
        eps_clip = configs.eps_clip
        entloss_coef = configs.entloss_coef
        num_envs = configs.num_envs
        n_episodes = CONFIG["training"]["max_updates"]
        log_every = CONFIG["training"]["log_every"]
    
    
    print(f"\n{'='*50}")
    print(f"TRAINING L2D FOR WEIGHTED SUM OBJECTIVE ({n_j}x{n_m})")
    print(f"{'='*50}")
    
    # Setup directories
    models_dir, figures_dir, results_dir = setup_directories(CONFIG)
    
    # Initialize multiple environments
    envs = [SJSSP(n_j=n_j, n_m=n_m, feature_set=feature_set) for _ in range(num_envs)]
    
    # Load validation data
    #dataLoaded = np.load('././DataGen/WeightData/weightedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    dataLoaded = np.load('/Users/mariegoffin/Documents/Master_3/IML/projects/L2D/DataGen/WeightData/weightedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))
        
    # Set seeds at the beginning of training
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)
    
    # Initialize memories for PPO
    memories = [Memory() for _ in range(num_envs)]
    
    # Initialize PPO agent with correct input dimension
    ppo = PPO(lr, configs.gamma, k_epochs, eps_clip,
              n_j=n_j,
              n_m=n_m,
              num_layers=num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=len(feature_set),
              hidden_dim=hidden_dim,
              num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
              num_mlp_layers_actor=num_mlp_layers_actor,
              hidden_dim_actor=hidden_dim_actor,
              num_mlp_layers_critic=num_mlp_layers_critic,
              hidden_dim_critic=hidden_dim_critic)
    
    # Calculate graph pooling setup
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, n_j*n_m, n_j*n_m]),
                            n_nodes=n_j*n_m,
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
                        # Fix: Ensure non-negative values and proper normalization
                        pi_final = np.maximum(pi_final, 0)  # Clip negative values to zero
                        if pi_final.sum() > 0:
                            pi_final = pi_final / pi_final.sum()  # Re-normalize
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
        
        # Log to wandb if in sweep mode
        if wandb_config is not None:
            wandb.log({
                "episode": episode,
                "training_reward": mean_reward,
                "training_weighted_sum": mean_weighted_sum,
                "loss": loss,
                "v_loss": v_loss
            })
        
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
                
                # Log comparison metrics to wandb
                if wandb_config is not None:
                    wandb.log({
                        "win_rate": comparison_metrics['win_rate'],
                        "win_vs_spt": comparison_metrics['win_vs_spt'],
                        "win_vs_wspt": comparison_metrics['win_vs_wspt'],
                        "improvement_over_spt": comparison_metrics['improvement_over_spt'],
                        "improvement_over_wspt": comparison_metrics['improvement_over_wspt']
                    })
                    
            # Log validation results to wandb
            if wandb_config is not None:
                wandb.log({
                    "validation_weighted_sum": validation_weighted_sum_mean,
                    "validation_improvement_pct": validation_results['improvement_pct'].mean()
                })

            # Save training statistics
            np.savez(
                os.path.join(results_dir, "training_stats.npz"),
                rewards=np.array(rewards_history),
                weighted_sums=np.array(weighted_sums_history),
                losses=np.array(loss_history),
                validation=np.array(validation_history)
            )
            
            # Save model if it's the best so far (smaller weighted sum is better)
            if validation_weighted_sum_mean < best_weighted_sum:
                best_model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_best.pth")
                torch.save(ppo.policy.state_dict(), best_model_path)
                
                 # If in wandb sweep, save a copy with run ID
                if wandb_config is not None:
                    best_run_model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_best_run_{wandb.run.id}.pth")
                    torch.save(ppo.policy.state_dict(), best_run_model_path)
                    wandb.save(best_run_model_path)
                
                print(f"New best model saved! Weighted sum: {validation_weighted_sum_mean:.2f}")
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
    
    # If in wandb sweep, return the optimization metric
    if wandb_config is not None:
        return best_weighted_sum
    
    else:
        return ppo.policy, final_model_path, weighted_sums_history, validation_history, training_results
