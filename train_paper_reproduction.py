# run_paper_experiment.py
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
from uniform_instance_gen import uni_instance_gen
from Params import configs
from validation import validate

device = torch.device(configs.device)

# Configuration parameters - modify these directly instead of using command-line arguments
CONFIG = {
    "training": {
        "enabled": True,         # Set to True to run training
        "n_j": 6,                # Number of jobs
        "n_m": 6,                # Number of machines
        "n_episodes": 10000,     # Number of episodes for training (paper used 10000)
        "save_every": 1000,      # Save model checkpoint every N episodes
        "log_every": 100,        # Log training statistics every N episodes
    },
    "evaluation": {
        "enabled": True,         # Set to True to run evaluation
        "n_instances": 100,      # Number of instances for testing
        "use_trained_model": True, # Whether to use model from training or load from file
        "model_path": None,      # Path to pre-trained model (used if use_trained_model=False)
    },
    "output_dir": "./paper_experiments"  # Directory for saving all results
}

# Get configuration parameters
n_j = CONFIG["training"]["n_j"]
n_m = CONFIG["training"]["n_m"]
n_episodes = CONFIG["training"]["n_episodes"]
save_every = CONFIG["training"]["save_every"]
log_every = CONFIG["training"]["log_every"]

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

def train_l2d():
    """Train L2D model according to paper specifications."""
    print(f"\n{'='*50}")
    print(f"TRAINING L2D ON {CONFIG['training']['n_j']}x{CONFIG['training']['n_m']} INSTANCES")
    print(f"{'='*50}")
    
    # Setup directories
    models_dir, figures_dir, results_dir = setup_directories()
    
    data_generator = uni_instance_gen
    
    # Initialize environments
    envs = [SJSSP(n_j=n_j, n_m=n_m) for _ in range(configs.num_envs)]
    
    # Load validation data
    dataLoaded = np.load('./DataGen/generatedData' + str(n_j) + '_' + str(n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
        
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)
    
    memories = [Memory() for _ in range(configs.num_envs)]
    
    # Initialize PPO agent
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=n_j,
              n_m=n_m,
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
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=torch.device("cpu")
    )
    
    # For tracking training progress
    rewards_history = []
    makespans_history = []
    loss_history = []
    validation_history = []
    
    # For saving best model
    best_makespan = float('inf')
    
    # Training loop
    start_time = time.time()
    print(f"Starting training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Reset all environments with new instances
        ep_rewards = [0 for _ in range(configs.num_envs)]
        makespans = [0 for _ in range(configs.num_envs)]
        adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []
        
        for i, env in enumerate(envs):
            adj, fea, candidate, mask = env.reset(data_generator(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high))
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality
        # rollout the env
        while True:
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
            # Saving episode data
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
        for i in range(configs.num_envs):
            ep_rewards[i] -= envs[i].posRewards
            makespans[i] = envs[i].LBs.max()  # Get actual makespan directly
            
        # Update policy using PPO
        loss, v_loss = ppo.update(memories, n_j * n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
            
        # Save history
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        mean_makespan = sum(makespans) / len(makespans)
        rewards_history.append(mean_reward)
        makespans_history.append(mean_makespan)
        loss_history.append(loss)
        
        # Run validation
        if (episode + 1) % log_every == 0:
            elapsed = time.time() - start_time
            
            # Run validation with enhanced function
            validation_results = validate(vali_data, ppo.policy)
            validation_history.append(validation_results['makespan'].mean())
            
            # Calculate moving averages for reporting
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            avg_makespan = np.mean(makespans_history[-100:]) if len(makespans_history) >= 100 else np.mean(makespans_history)
            
            # Print comprehensive metrics
            print(f"Episode {episode+1}/{n_episodes} | Reward: {mean_reward:.2f} | "
                  f"Makespan: {mean_makespan:.2f} | Loss: {loss:.6f}")
            print(f"Avg(100): Reward={avg_reward:.2f}, Makespan={avg_makespan:.2f} | "
                  f"Episode time: {time.time()-episode_start:.2f}s | Total time: {elapsed:.2f}s")
            print(f"Validation Makespan: {validation_results['makespan'].mean():.2f} | "
                  f"Original Metric: {validation_results['reward_derived'].mean():.2f} | "
                  f"Improvement: {validation_results['improvement_pct'].mean():.2f}%")
            
            # Save training statistics
            np.savez(
                os.path.join(results_dir, "training_stats.npz"),
                rewards=np.array(rewards_history),
                makespans=np.array(makespans_history),
                losses=np.array(loss_history),
                validation=np.array(validation_history),
                validation_full=validation_results
            )
            
            # Plot learning curves
            plot_learning_curves(
                rewards=rewards_history, 
                losses=loss_history, 
                makespans=makespans_history, 
                figures_dir=figures_dir,
                validation_history=validation_history,
            )
            
            # Use actual makespan for model saving decisions
            if validation_results['makespan'].mean() < best_makespan:
                best_model_path = os.path.join(models_dir, f"l2d_makespan_{n_j}x{n_m}_best.pth")
                torch.save(ppo.policy.state_dict(), best_model_path)
                print(f"New best model saved! Makespan: {validation_results['makespan'].mean():.2f}")
                best_makespan = validation_results['makespan'].mean()
                
        # Save regular checkpoint
        if (episode + 1) % save_every == 0:
            model_path = os.path.join(models_dir, f"l2d_makespan_{n_j}x{n_m}_episode_{episode+1}.pth")
            torch.save(ppo.policy.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")
        
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"l2d_makespan_{n_j}x{n_m}_final.pth")
    torch.save(ppo.policy.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return ppo.policy, final_model_path

def plot_learning_curves(rewards, losses, makespans, figures_dir, validation_history=None):
    """
    Plot and save comprehensive training curves for makespan objective with training vs validation comparisons.
    
    Args:
        rewards: List of training rewards
        losses: List of training losses
        makespans: List of training makespans
        figures_dir: Directory to save figures
        validation_history: List of historical validation makespan means (one value per validation)
        log_every: Number of episodes between validation checks (for x-axis alignment)
    """
    window = 100  # For smoothing
    
    # Data validation and debug info
    print(f"Data lengths - Rewards: {len(rewards)}, Losses: {len(losses)}, Makespans: {len(makespans)}")
    if validation_history is not None:
        print(f"Validation history length: {len(validation_history)}")
        print(f"First few validation values: {validation_history[:5]}")
        print(f"Last few validation values: {validation_history[-5:]} (smaller is better)")
    
    # Calculate moving averages for smoothing
    reward_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    makespan_avg = [np.mean(makespans[max(0, i-window):i+1]) for i in range(len(makespans))]
    loss_avg = [np.mean(losses[max(0, i-window):i+1]) for i in range(len(losses))]
    
    episodes = list(range(len(makespans)))
    
    # 1. Main objective comparison plot (Training vs Validation)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, makespans, 'b-', alpha=0.2, label='Training (per episode)')
    plt.plot(episodes, makespan_avg, 'b-', linewidth=2, label='Training (smoothed)')
    
    # Plot validation history with proper x-axis alignment
    if validation_history is not None and len(validation_history) > 0:
        # Generate proper x-positions based on logging interval
        val_episodes = [(i+1) * log_every for i in range(len(validation_history))]
        
        # Plot the validation history
        plt.plot(val_episodes, validation_history, 'r-o', linewidth=2, label='Validation')
        
        # Add best training and validation lines
        best_training = min(makespan_avg)
        best_validation = min(validation_history)
        plt.axhline(y=best_training, color='b', linestyle='--', alpha=0.5, 
                    label=f'Best training: {best_training:.0f}')
        plt.axhline(y=best_validation, color='r', linestyle='--', alpha=0.5, 
                    label=f'Best validation: {best_validation:.0f}')
    
    plt.title('Makespan Comparison: Training vs Validation')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "makespan_comparison.png"))
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
    
    # 4. Smoothed makespans plot
    plt.figure(figsize=(10, 6))
    plt.plot(makespan_avg)
    plt.title(f'Makespan During Training ({window}-episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Makespan')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "makespans_smoothed.png"))
    plt.close()
    
    # 5. Raw makespans (last 1000 episodes for detail)
    plt.figure(figsize=(10, 6))
    if len(makespans) > 1000:
        plt.plot(episodes[-1000:], makespans[-1000:])
        plt.title('Makespan (Last 1000 Episodes)')
    else:
        plt.plot(episodes, makespans)
        plt.title('Makespan (All Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "makespans_raw.png"))
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
        
        plt.title('Validation Makespan During Training')
        plt.xlabel('Validation Check (Episode)')
        plt.ylabel('Makespan')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(figures_dir, "validation_curve.png"))
        plt.close()
        
        # 7. Zoomed validation curve (last 20 validations)
        if len(validation_history) > 10:
            plt.figure(figsize=(10, 6))
            last_n = min(20, len(validation_history))
            plt.plot(val_episodes[-last_n:], validation_history[-last_n:], 'r-o')
            plt.title(f'Recent Validation Makespan (Last {last_n} Checks)')
            plt.xlabel('Validation Check (Episode)')
            plt.ylabel('Makespan')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(figures_dir, "recent_validation_curve.png"))
            plt.close()

def test_l2d(env, instance, policy):
    """Apply trained L2D policy to an instance."""
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
    
    return env.LBs.max()

def test_spt(env, instance):
    """Apply SPT dispatching rule to an instance."""
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        eligible_ops = candidate[~mask]
        proc_times = np.array([env.dur[op // env.number_of_machines, op % env.number_of_machines] 
                              for op in eligible_ops])
        
        action_idx = np.argmin(proc_times)
        action = eligible_ops[action_idx]
        
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.LBs.max()

def test_srpt(env, instance):
    """Apply SRPT dispatching rule to an instance."""
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        eligible_ops = candidate[~mask]
        
        remaining_times = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            op_idx = op % env.number_of_machines
            
            remaining = 0
            for m in range(op_idx, env.number_of_machines):
                if env.finished_mark[job_idx, m] == 0:
                    remaining += env.dur[job_idx, m]
            
            remaining_times.append(remaining)
        
        action_idx = np.argmin(np.array(remaining_times))
        action = eligible_ops[action_idx]
        
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.LBs.max()

def evaluate_policies(policy=None, model_path=None):
    """Evaluate L2D, SPT, and SRPT on random instances."""
    print(f"\n{'='*50}")
    print(f"EVALUATION ON {CONFIG['evaluation']['n_instances']} RANDOM INSTANCES")
    print(f"{'='*50}")
    
    # Get configuration parameters
    n_j = CONFIG["training"]["n_j"]
    n_m = CONFIG["training"]["n_m"]
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
            input_dim=configs.input_dim,
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
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Generate random instances
    print(f"Generating {n_instances} random {n_j}x{n_m} instances...")
    instances = [uni_instance_gen(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high) 
                for _ in range(n_instances)]
    
    # Results dictionary
    results = {
        "L2D": [],
        "SPT": [],
        "SRPT": []
    }
    
    # Test all methods on all instances
    print("Testing policies on instances...")
    for i, instance in enumerate(instances):
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{n_instances} instances")
        
        # Test L2D
        l2d_makespan = test_l2d(env, instance, policy)
        results["L2D"].append(l2d_makespan)
        
        # Test SPT
        spt_makespan = test_spt(env, instance)
        results["SPT"].append(spt_makespan)
        
        # Test SRPT
        srpt_makespan = test_srpt(env, instance)
        results["SRPT"].append(srpt_makespan)
    
    # Calculate averages
    avg_results = {method: np.mean(makespans) for method, makespans in results.items()}
    
    print("\nEvaluation Results:")
    print("\nAverage Makespans:")
    for method, avg in avg_results.items():
        print(f"{method}: {avg:.2f}")
    
    # Calculate improvement percentages
    l2d_avg = avg_results["L2D"]
    spt_improvement = ((avg_results["SPT"] - l2d_avg) / avg_results["SPT"]) * 100
    srpt_improvement = ((avg_results["SRPT"] - l2d_avg) / avg_results["SRPT"]) * 100
    
    print(f"\nL2D improvement over SPT: {spt_improvement:.2f}%")
    print(f"L2D improvement over SRPT: {srpt_improvement:.2f}%")
    
    # Calculate performance ratios
    ratios = defaultdict(list)
    
    for i in range(n_instances):
        best_makespan = min(results["L2D"][i], results["SPT"][i], results["SRPT"][i])
        
        for method in results:
            ratio = results[method][i] / best_makespan
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
    results_file = os.path.join(results_dir, "comparison_results.npz")
    np.savez(results_file, **results)
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate performance profile
    plot_performance_profile(ratios, results_dir)
    
    return results, ratios

def plot_performance_profile(ratios, results_dir):
    """Plot performance profiles for the results."""
    plt.figure(figsize=(10, 6))
    
    for method, method_ratios in ratios.items():
        sorted_ratios = np.sort(method_ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        plt.step(sorted_ratios, y, where='post', label=method)
    
    plt.title('Performance Profile')
    plt.xlabel('Performance Ratio (τ)')
    plt.ylabel('Probability P(r_{p,s} ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "performance_profile.png"))
    
    # Also create a log2 performance profile
    plt.figure(figsize=(10, 6))
    
    for method, method_ratios in ratios.items():
        sorted_ratios = np.sort(method_ratios)
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        log_ratios = np.log2(sorted_ratios)
        plt.step(log_ratios, y, where='post', label=method)
    
    plt.title('Log2-Scaled Performance Profile')
    plt.xlabel('log2(Performance Ratio)')
    plt.ylabel('Probability P(log2(r_{p,s}) ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "performance_profile_log2.png"))
    plt.close()
    
    print(f"Performance profiles saved to {results_dir}")

if __name__ == "__main__":
    # Main execution
    policy = None
    model_path = None
    
    # Run training if enabled
    if CONFIG["training"]["enabled"]:
        policy, model_path = train_l2d()
    
    # Run evaluation if enabled
    if CONFIG["evaluation"]["enabled"]:
        if not CONFIG["evaluation"]["use_trained_model"]:
            policy = None
            model_path = CONFIG["evaluation"]["model_path"]
        
        evaluate_policies(policy, model_path)
    
    print("\nExperiment completed!")