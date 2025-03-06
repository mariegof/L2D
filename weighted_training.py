# weighted_training.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
from mb_agg import *
from agent_utils import select_action
import torch
import time
import numpy as np
from Params import configs
from PPO_jssp_multiInstances import PPO, Memory

device = torch.device(configs.device)

feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'remaining_weighted_work']

def setup_result_dirs():
    """Create directory structure for saving weighted objective results."""
    base_dir = f"./WeightedResults/{configs.n_j}x{configs.n_m}"
    dirs = {
        "models": f"{base_dir}/models",
        "logs": f"{base_dir}/logs", 
        "figures": f"{base_dir}/figures"
    }
    
    # Create directories
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        
    return dirs

def plot_metrics(metrics, dirs, prefix=""):
    """Generate and save plots to visualize training progress."""
    # Extract metrics
    episodes = range(len(metrics["rewards"]))
    
    # Create a single combined plot file for rewards and losses
    plt.figure(figsize=(12, 8))
    
    # Top subplot for rewards
    plt.subplot(2, 1, 1)
    plt.plot(episodes, metrics["rewards"])
    plt.title('Rewards During Training')
    plt.ylabel('Average Reward')
    plt.grid(alpha=0.3)
    
    # Bottom subplot for losses
    plt.subplot(2, 1, 2)
    plt.plot(episodes, metrics["losses"])
    plt.title('Loss During Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig(f"{dirs['figures']}/{prefix}training_metrics.png")
    plt.close()
    
    # Create a single validation comparison plot if we have validation data
    if "validation" in metrics and len(metrics["validation"]) > 0:
        val_episodes = list(range(0, len(episodes), 100))[:len(metrics["validation"])]
        
        plt.figure(figsize=(10, 6))
        
        # Plot both training weighted sums and validation performance
        val_rewards = [metrics["rewards"][i] for i in val_episodes if i < len(metrics["rewards"])][:len(metrics["validation"])]
        
        plt.plot(val_episodes, [-r for r in val_rewards], label='Training')
        plt.plot(val_episodes, metrics["validation"], label='Validation')
        plt.title('Training vs Validation Weighted Sum')
        plt.xlabel('Episode')
        plt.ylabel('Weighted Sum')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{dirs['figures']}/{prefix}validation_comparison.png")
        plt.close()

def weighted_validate(vali_set, model):
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]

    from JSSP_Env import SJSSP
    from mb_agg import g_pool_cal
    from agent_utils import sample_select_action
    from agent_utils import greedy_select_action
    import numpy as np
    import torch
    from Params import configs
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set=feature_set)
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    objective = []
    # rollout using model
    for data in vali_set:
        # Extract weights from the last column of the weight matrix
        times = data[0]
        machines = data[1]
        weight_matrix = data[2]
        weights = weight_matrix[:, -1]  # Extract the last column
        
        # Create a new data tuple with the extracted weights
        data = (times, machines, weights)
        print(f"Sample weights: {weights[:5]}")  # First 5 weights
        
        adj, fea, candidate, mask = env.reset(data)
        rewards = - env.initQuality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
                
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            rewards += reward
            if done:
                break
        objective.append(rewards - env.posRewards)
        print(rewards - env.posRewards)
    return np.array(objective)

def main():
    # Setup result directories
    dirs = setup_result_dirs()

    # All your existing initialization code stays the same
    from JSSP_Env import SJSSP
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m, feature_set=feature_set) for _ in range(configs.num_envs)]
    
    from uniform_instance_gen import weighted_instance_gen
    data_generator = weighted_instance_gen

    dataLoaded = np.load('./DataGen/WeightData/weightedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1], dataLoaded[i][2]))

    # Your model setup code stays the same
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

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

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
                            n_nodes=configs.n_j*configs.n_m,
                            device=device)
    
    # Create metrics dictionary for tracking
    metrics = {
        "rewards": [],      # Average rewards per episode
        "losses": [],       # Loss values
        "v_losses": [],     # Value function losses
        "validation": [],   # Validation performance
        "wall_times": []    # Wall clock time per episode
    }
    
    # For calculating moving averages
    window_size = 10
    
    # Training loop variables
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    start_time = time.time()
    
    # Main training loop
    for i_update in range(configs.max_updates):
        # Your training code stays the same
        ep_rewards = [0 for _ in range(configs.num_envs)]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []
        
        # Environment reset and training steps
        for i, env in enumerate(envs):
            adj, fea, candidate, mask = env.reset(data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high,weight_low=configs.weight_low, weight_high=configs.weight_high))
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality
        
        # Environment rollout
        while True:
            # All your training code stays the same
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(x=fea_tensor_envs[i],
                                           graph_pool=g_pool_step,
                                           padded_nei=None,
                                           adj=adj_tensor_envs[i],
                                           candidate=candidate_tensor_envs[i].unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            
            # Step environments and collect data
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj, fea, reward, done, candidate, mask = envs[i].step(action_envs[i].item())
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
            
            if envs[0].done():
                break
        
        # Finalize episode rewards
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        # Update policy
        loss, v_loss = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        
        # Track metrics
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        metrics["rewards"].append(mean_rewards_all_env)
        metrics["losses"].append(loss)
        metrics["v_losses"].append(v_loss)
        metrics["wall_times"].append(time.time() - start_time)
        
        # Compute moving averages for logging
        recent_rewards = metrics["rewards"][-window_size:] if len(metrics["rewards"]) >= window_size else metrics["rewards"]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Log results with moving average
        print(f'Episode {i_update + 1}\t Last reward: {mean_rewards_all_env:.2f}\t Avg({window_size}): {avg_reward:.2f}\t Loss: {loss:.4f}')
        
        # Periodic validation
        if (i_update + 1) % 100 == 0:
            # Save training log
            with open(f"{dirs['logs']}/training_log_{i_update+1}.txt", 'w') as f:
                f.write(str(metrics))
            
            file_writing_obj = open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))
            
            # Run validation
            print("Running validation...")
            vali_result = weighted_validate(vali_data, ppo.policy).mean()
            metrics["validation"].append(vali_result)
            validation_log.append(vali_result)
            
            # Save model if it's the best
            if vali_result < record:
                model_path = f"{dirs['models']}/weighted_{configs.n_j}_{configs.n_m}_{i_update+1}.pth"
                torch.save(ppo.policy.state_dict(), model_path)
                record = vali_result
                print(f"New best model saved: {model_path} with weighted sum: {vali_result:.2f}")
            
            print(f'Validation weighted sum: {vali_result:.2f}')
            file_writing_obj1 = open('./' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))
            
            # Plot validation checkpoint metrics
            plot_metrics(metrics, dirs, prefix=f"checkpoint_")
    
    # After training is complete, plot final metrics (OUTSIDE the loop)
    plot_metrics(metrics, dirs, prefix="final_")
    
    # Save final model
    final_model_path = f"{dirs['models']}/weighted_{configs.n_j}_{configs.n_m}_final.pth"
    torch.save(ppo.policy.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Report total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == '__main__':
    
    total1 = time.time()
    main()
    total2 = time.time()
    print(total2 - total1)