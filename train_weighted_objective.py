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
from agent_utils import select_action, greedy_select_action
from uniform_instance_gen import weighted_instance_gen
from Params import configs

# Configuration parameters
CONFIG = {
    "training": {
        "enabled": True,
        "n_j": 6,                # Number of jobs
        "n_m": 6,                # Number of machines
        "max_updates": 100,     # Number of episodes for training
        "weight_low": 1,         # Minimum job weight
        "weight_high": 10,       # Maximum job weight
        "save_every": 1000,      # Save model checkpoints every N episodes
        "log_every": 1,        # Log training statistics every N episodes
        "num_envs": 8            # Number of environments for training
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

def setup_directories():
    """Create necessary directories for output."""
    # Main output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Models directory
    models_dir = os.path.join(CONFIG["output_dir"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Figures directory
    figures_dir = os.path.join(CONFIG["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Results directory
    results_dir = os.path.join(CONFIG["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return models_dir, figures_dir, results_dir

def train_l2d_multi_env():
    """
    Train L2D model for weighted sum objective using multiple environments,
    matching the approach in PPO_jssp_multiInstances.py.
    """
    print(f"\n{'='*50}")
    print(f"TRAINING L2D FOR WEIGHTED SUM OBJECTIVE ({CONFIG['training']['n_j']}x{CONFIG['training']['n_m']})")
    print(f"{'='*50}")
    
    # Get configuration parameters
    n_j = configs.n_j
    n_m = configs.n_m
    weight_low = configs.weight_low
    weight_high = configs.weight_high
    n_episodes = CONFIG["training"]["max_updates"]
    save_every = CONFIG["training"]["save_every"]
    log_every = CONFIG["training"]["log_every"]
    num_envs = configs.num_envs
    
    # Setup directories
    models_dir, figures_dir, results_dir = setup_directories()
    
    # Initialize multiple environments
    envs = [SJSSP(n_j=n_j, n_m=n_m, feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'machine_contention']) for _ in range(num_envs)]
    
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
                    if episode < n_episodes * 0.2:  # First 20% of training
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
            
            # Run validation
            validation_result = validate_weighted(vali_data, ppo.policy).mean()
            validation_history.append(-validation_result)  # Convert back to positive for tracking
            
            print(f"Validation Weighted Sum: {-validation_result:.2f}")
            
            # Compare with baselines occasionally
            if (episode + 1) % (log_every * 5) == 0:
                results, averages, win_counts = compare_validation_methods(vali_data[:10], ppo.policy)
            
            # Save training statistics
            np.savez(
                os.path.join(results_dir, "training_stats.npz"),
                rewards=np.array(rewards_history),
                weighted_sums=np.array(weighted_sums_history),
                losses=np.array(loss_history),
                validation=np.array(validation_history)
            )
            
            # Plot learning curves
            if len(rewards_history) >= 1:
                plot_learning_curves(rewards_history, weighted_sums_history, loss_history, validation_history, figures_dir)
            
            # Save model if it's the best so far
            if -validation_result < best_weighted_sum:
                best_model_path = os.path.join(models_dir, f"l2d_weighted_{n_j}x{n_m}_best.pth")
                torch.save(ppo.policy.state_dict(), best_model_path)
                print(f"New best model saved! Weighted sum: {-validation_result:.2f}")
                best_weighted_sum = -validation_result
        
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
    
    return ppo.policy, final_model_path, weighted_sums_history, validation_history

def validate_weighted(vali_set, model):
    """
    Custom validation function for weighted sum objectives.
    
    Args:
        vali_set: List of JSSP instances with weights
        model: L2D policy model
        
    Returns:
        weighted_sums: Array of weighted sum values for each instance
    """
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]

    from JSSP_Env import SJSSP
    from mb_agg import g_pool_cal
    from agent_utils import greedy_select_action
    import numpy as np
    import torch
    from Params import configs
    
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'machine_contention'])
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    
    # Track weighted sum and makespan for evaluation
    weighted_sums = []
    
    # Rollout using model
    for data in vali_set:
         # Extract weights from the last column of the weight matrix
        times = data[0]
        machines = data[1]
        weight_matrix = data[2]
        weights = weight_matrix[:, -1]  # Extract the last column
        
        # Create a new data tuple with the extracted weights
        data = (times, machines, weights)
        adj, fea, candidate, mask = env.reset(data)
        
        while not env.done():
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
            
            # Select best action greedy
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.step(action.item())
        
        # Record weighted sum metric
        weighted_sums.append(env.weighted_sum)
    
    # Return negative weighted sum (so smaller is better, for consistency with original code)
    return -np.array(weighted_sums)

def compare_validation_methods(vali_set, model):
    """
    Compare L2D with baseline methods on validation set.
    
    Args:
        vali_set: List of JSSP instances with weights
        model: L2D policy model
        
    Returns:
        results: Dictionary of method name -> performance metric
    """
    from JSSP_Env import SJSSP
    import numpy as np
    
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'machine_contention'])
    
    results = {
        "L2D": [],
        "SPT": [],
        "WSPT": []
    }
    
    # Evaluate each method on all instances
    for instance in vali_set:
        # Test SPT
        # Extract weights from the last column of the weight matrix
        times = instance[0]
        machines = instance[1]
        weight_matrix = instance[2]
        weights = weight_matrix[:, -1]  # Extract the last column

        # Create a new data tuple with the extracted weights
        instance = (times, machines, weights)
        adj, fea, candidate, mask = env.reset(instance)
        while not env.done():
            eligible_ops = candidate[~mask]
            proc_times = []
            for op in eligible_ops:
                job_idx = op // env.number_of_machines
                op_idx = op % env.number_of_machines
                proc_times.append(env.dur[job_idx, op_idx])
            action_idx = np.argmin(np.array(proc_times))
            action = eligible_ops[action_idx]
            adj, fea, reward, done, candidate, mask = env.step(action)
        results["SPT"].append(env.weighted_sum)
        
        # Test WSPT
        adj, fea, candidate, mask = env.reset(instance)
        while not env.done():
            eligible_ops = candidate[~mask]
            wspt_values = []
            for op in eligible_ops:
                job_idx = op // env.number_of_machines
                op_idx = op % env.number_of_machines
                wspt_values.append(env.weights[job_idx] / env.dur[job_idx, op_idx])
            action_idx = np.argmax(np.array(wspt_values))
            action = eligible_ops[action_idx]
            adj, fea, reward, done, candidate, mask = env.step(action)
        results["WSPT"].append(env.weighted_sum)
    
    # Use the validate_weighted function for L2D
    l2d_results = -validate_weighted(vali_set, model)
    results["L2D"] = l2d_results.tolist()
    
    # Calculate average metrics
    averages = {}
    win_counts = {"L2D": 0, "SPT": 0, "WSPT": 0}
    
    for method in results:
        averages[method] = np.mean(results[method])
    
    # Track win statistics
    for i in range(len(vali_set)):
        best_method = min(results.keys(), key=lambda m: results[m][i])
        win_counts[best_method] += 1
    
    print("\nValidation Results:")
    print("Average Weighted Sums:")
    for method, avg in averages.items():
        print(f"{method}: {avg:.2f}")
    
    print("\nWin Statistics:")
    for method, count in win_counts.items():
        percent = (count / len(vali_set)) * 100
        print(f"{method}: {count}/{len(vali_set)} ({percent:.2f}%)")
    
    return results, averages, win_counts

def plot_learning_curves(rewards, weighted_sums, losses, validation_history, figures_dir, 
                         validation_losses=None, validation_rewards=None):
    """
    Plot and save comprehensive training curves with training vs validation comparisons.
    
    Args:
        rewards: List of training rewards
        weighted_sums: List of training weighted sums
        losses: List of training losses
        validation_history: List of validation weighted sums
        figures_dir: Directory to save figures
        validation_losses: Optional list of validation losses
        validation_rewards: Optional list of validation rewards
    """
    window = 100  # For smoothing
    
    # Calculate moving averages for smoothing
    reward_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    weighted_sum_avg = [np.mean(weighted_sums[max(0, i-window):i+1]) for i in range(len(weighted_sums))]
    loss_avg = [np.mean(losses[max(0, i-window):i+1]) for i in range(len(losses))]
    
    episodes = range(len(weighted_sums))
    
    # 1. Weighted Sum Comparison (Training vs Validation)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, weighted_sums, 'b-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, weighted_sum_avg, 'b-', linewidth=2, label='Training (smoothed)')
    
    if len(validation_history) > 0:
        val_episodes = list(range(len(validation_history)))
        plt.plot(val_episodes, validation_history, 'r-o', linewidth=2, label='Validation')
        
        # Add best training and validation lines
        best_training = min(weighted_sum_avg)
        best_validation = min(validation_history)
        plt.axhline(y=best_training, color='b', linestyle='--', alpha=0.5, label=f'Best training: {best_training:.0f}')
        plt.axhline(y=best_validation, color='r', linestyle='--', alpha=0.5, label=f'Best validation: {best_validation:.0f}')
    
    plt.title('Weighted Sum Comparison: Training vs Validation')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Sum')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "weighted_sum_comparison.png"))
    plt.close()
    
    # 2. Loss Comparison (Training vs Validation)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, 'g-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, loss_avg, 'g-', linewidth=2, label='Training (smoothed)')
    
    if validation_losses is not None and len(validation_losses) > 0:
        val_episodes = list(range(len(validation_losses)))
        plt.plot(val_episodes, validation_losses, 'r-o', linewidth=2, label='Validation')
    
    plt.title('Loss Comparison: Training vs Validation')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "loss_comparison.png"))
    plt.close()
    
    # 3. Reward Comparison (Training vs Validation)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, 'm-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, reward_avg, 'm-', linewidth=2, label='Training (smoothed)')
    
    if validation_rewards is not None and len(validation_rewards) > 0:
        val_episodes = list(range(len(validation_rewards)))
        plt.plot(val_episodes, validation_rewards, 'r-o', linewidth=2, label='Validation')
    
    plt.title('Reward Comparison: Training vs Validation')
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
        plt.plot(weighted_sums[-1000:])
        plt.title('Weighted Sum (Last 1000 Episodes)')
    else:
        plt.plot(weighted_sums)
        plt.title('Weighted Sum (All Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Weighted Sum')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "weighted_sums_raw.png"))
    plt.close()
    
    # 6. Validation progress curve if available
    if validation_history and len(validation_history) > 0:
        plt.figure(figsize=(10, 6))
        val_episodes = list(range(len(validation_history)))
        plt.plot(val_episodes, validation_history, 'r-o')
        plt.title('Validation Weighted Sum During Training')
        plt.xlabel('Validation Check')
        plt.ylabel('Weighted Sum')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "validation_curve.png"))
        plt.close()

def test_l2d_weighted(env, instance, policy):
    """Apply trained L2D policy to an instance and return weighted sum objective."""
    times, machines, weights = instance
    
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=torch.device("cpu")
    )
    
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        # Convert to tensors
        fea_tensor = torch.from_numpy(np.copy(fea)).to(torch.device("cpu"))
        adj_tensor = torch.from_numpy(np.copy(adj)).to(torch.device("cpu")).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(torch.device("cpu"))
        mask_tensor = torch.from_numpy(np.copy(mask)).to(torch.device("cpu"))
        
        # Get action from policy
        with torch.no_grad():
            pi, _ = policy(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
        
        # Select action (greedy)
        _, idx = pi.max(1)
        action = candidate[idx.item()]
        
        # Take step
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.weighted_sum

def test_spt(env, instance):
    """Apply SPT dispatching rule and return weighted sum objective."""
    times, machines, weights = instance
    
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate processing time for each eligible operation
        proc_times = np.array([env.dur[op // env.number_of_machines, op % env.number_of_machines] 
                              for op in eligible_ops])
        
        # Select operation with minimum processing time
        action_idx = np.argmin(proc_times)
        action = eligible_ops[action_idx]
        
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.weighted_sum

def test_srpt(env, instance):
    """Apply SRPT dispatching rule and return weighted sum objective."""
    times, machines, weights = instance
    
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate remaining processing time for each job with eligible operation
        remaining_times = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            op_idx = op % env.number_of_machines
            
            # Sum remaining processing times for this job
            remaining = 0
            for m in range(op_idx, env.number_of_machines):
                if env.finished_mark[job_idx, m] == 0:  # If operation not completed
                    remaining += env.dur[job_idx, m]
            
            remaining_times.append(remaining)
        
        # Select operation from job with minimum remaining time
        action_idx = np.argmin(np.array(remaining_times))
        action = eligible_ops[action_idx]
        
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.weighted_sum

def test_wspt(env, instance):
    """Apply Weighted Shortest Processing Time rule and return weighted sum objective."""
    times, machines, weights = instance
    
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate weight/processing time ratio for each eligible operation
        wspt_values = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            op_idx = op % env.number_of_machines
            proc_time = env.dur[job_idx, op_idx]
            weight = weights[job_idx]
            # Higher value means higher priority (weight/time)
            wspt_values.append(weight / proc_time)
        
        # Select operation with maximum ratio
        action_idx = np.argmax(np.array(wspt_values))
        action = eligible_ops[action_idx]
        
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.weighted_sum

def evaluate_policies(policy=None, model_path=None):
    """Evaluate policies on random instances for weighted sum objective."""
    print(f"\n{'='*50}")
    print(f"EVALUATING ON {CONFIG['evaluation']['n_instances']} RANDOM WEIGHTED INSTANCES")
    print(f"{'='*50}")
    
    # Get configuration parameters
    n_j = CONFIG["training"]["n_j"]
    n_m = CONFIG["training"]["n_m"]
    weight_low = CONFIG["training"]["weight_low"]
    weight_high = CONFIG["training"]["weight_high"]
    n_instances = CONFIG["evaluation"]["n_instances"]
    
    # Setup directories
    _, _, results_dir = setup_directories()
    
    # Load policy if needed
    if policy is None and model_path is not None:
        policy = PPO(
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            n_j=n_j,
            n_m=n_m,
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,  # Use 3 for weighted feature
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic
        ).policy
        
        policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded policy from {model_path}")
    
    if policy is None:
        print("Error: No policy available for evaluation.")
        return
    
    # Initialize environment
    env = SJSSP(n_j=n_j, n_m=n_m, feature_set = ['LBs', 'finished_mark', 'weighted_priorities', 'normalized_weights', 'machine_contention'])
    
    # Generate random weighted instances
    print(f"Generating {n_instances} random weighted instances...")
    instances = [
        weighted_instance_gen(
            n_j=n_j, n_m=n_m, 
            low=configs.low, high=configs.high,
            weight_low=weight_low, weight_high=weight_high
        ) 
        for _ in range(n_instances)
    ]
    
    # Results dictionary for weighted sum objective
    results = {
        "L2D": [],
        "SPT": [],
        "SRPT": [],
        "WSPT": []
    }
    
    # Test all methods on all instances
    print("Testing policies on instances...")
    for i, instance in enumerate(instances):
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{n_instances} instances")
        
        # Test L2D
        l2d_weighted_sum = test_l2d_weighted(env, instance, policy)
        results["L2D"].append(l2d_weighted_sum)
        
        # Test SPT
        spt_weighted_sum = test_spt(env, instance)
        results["SPT"].append(spt_weighted_sum)
        
        # Test SRPT
        srpt_weighted_sum = test_srpt(env, instance)
        results["SRPT"].append(srpt_weighted_sum)
        
        # Test WSPT
        wspt_weighted_sum = test_wspt(env, instance)
        results["WSPT"].append(wspt_weighted_sum)
    
    # Calculate averages
    avg_results = {method: np.mean(weighted_sums) for method, weighted_sums in results.items()}
    
    print("\nEvaluation Results:")
    print("\nAverage Weighted Sums:")
    for method, avg in avg_results.items():
        print(f"{method}: {avg:.2f}")
    
    # Calculate improvement percentages
    l2d_avg = avg_results["L2D"]
    for method in ["SPT", "SRPT", "WSPT"]:
        if method != "L2D":
            improvement = ((avg_results[method] - l2d_avg) / avg_results[method]) * 100
            print(f"L2D improvement over {method}: {improvement:.2f}%")
    
    # Calculate performance ratios
    ratios = defaultdict(list)
    
    for i in range(n_instances):
        best_weighted_sum = min(results["L2D"][i], results["SPT"][i], 
                              results["SRPT"][i], results["WSPT"][i])
        
        for method in results:
            ratio = results[method][i] / best_weighted_sum
            ratios[method].append(ratio)
    
    # Get win statistics
    wins = {method: 0 for method in results}
    for i in range(n_instances):
        best_method = min(results.keys(), key=lambda m: results[m][i])
        wins[best_method] += 1
    
    print("\nWin Statistics:")
    for method, count in wins.items():
        print(f"{method}: {count}/{n_instances} ({count/n_instances*100:.2f}%)")
    
    # Save results
    results_file = os.path.join(results_dir, "weighted_comparison_results.npz")
    np.savez(results_file, **results)
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate performance profile
    plot_performance_profile(ratios, results_dir)
    
    return results, ratios

def plot_performance_profile(ratios, results_dir):
    """Plot performance profiles for the weighted sum objective results."""
    plt.figure(figsize=(10, 6))
    
    for method, method_ratios in ratios.items():
        sorted_ratios = np.sort(method_ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        plt.step(sorted_ratios, y, where='post', label=method)
    
    plt.title('Performance Profile for Weighted Sum Objective')
    plt.xlabel('Performance Ratio (τ)')
    plt.ylabel('Probability P(r_{p,s} ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "weighted_performance_profile.png"))
    
    # Also create a log2 performance profile
    plt.figure(figsize=(10, 6))
    
    for method, method_ratios in ratios.items():
        sorted_ratios = np.sort(method_ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        log_ratios = np.log2(sorted_ratios)
        plt.step(log_ratios, y, where='post', label=method)
    
    plt.title('Log2-Scaled Performance Profile for Weighted Sum Objective')
    plt.xlabel('log2(Performance Ratio)')
    plt.ylabel('Probability P(log2(r_{p,s}) ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "weighted_performance_profile_log2.png"))
    plt.close()
    
    print(f"Performance profiles saved to {results_dir}")

if __name__ == "__main__":
    # Force CPU for consistency
    configs.device = "cpu"
    
    policy = None
    model_path = None
    
    # Run training if enabled
    if CONFIG["training"]["enabled"]:
        policy, model_path, weighted_sums, validation_sums = train_l2d_multi_env()
    
    # Run evaluation if enabled
    if CONFIG["evaluation"]["enabled"]:
        if not CONFIG["evaluation"]["use_trained_model"]:
            policy = None
            model_path = CONFIG["evaluation"]["model_path"]
        
        evaluate_policies(policy, model_path)
    
    print("\nWeighted sum objective experiment completed!")