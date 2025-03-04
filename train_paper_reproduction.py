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
    
    # Get configuration parameters
    n_j = CONFIG["training"]["n_j"]
    n_m = CONFIG["training"]["n_m"]
    n_episodes = CONFIG["training"]["n_episodes"]
    save_every = CONFIG["training"]["save_every"]
    log_every = CONFIG["training"]["log_every"]
    
    # Setup directories
    models_dir, figures_dir, _ = setup_directories()
    
    # Initialize environment and PPO agent
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Initialize PPO agent
    ppo = PPO(
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
    )
    
    # Calculate graph pooling setup
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=torch.device("cpu")
    )
    
    # For tracking training progress
    rewards_history = []
    loss_history = []
    makespans_history = []
    
    # Training loop
    start_time = time.time()
    print(f"Starting training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Generate random instance
        instance_data = uni_instance_gen(n_j=n_j, n_m=n_m, low=configs.low, high=configs.high)
        
        # Reset environment
        adj, fea, candidate, mask = env.reset(instance_data)
        
        # Initialize memory
        memory = Memory()
        
        # Run episode
        ep_reward = -env.initQuality
        
        while not env.done():
            # Prepare tensors
            fea_tensor = torch.from_numpy(np.copy(fea)).to(torch.device("cpu"))
            adj_tensor = torch.from_numpy(np.copy(adj)).to(torch.device("cpu")).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(torch.device("cpu"))
            mask_tensor = torch.from_numpy(np.copy(mask)).to(torch.device("cpu"))
            
            # Get action from policy
            with torch.no_grad():
                pi, _ = ppo.policy_old(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
            
            # Select action
            action, a_idx = select_action(pi, candidate, memory)
            
            # Store state in memory
            memory.adj_mb.append(adj_tensor)
            memory.fea_mb.append(fea_tensor)
            memory.candidate_mb.append(candidate_tensor)
            memory.mask_mb.append(mask_tensor)
            memory.a_mb.append(a_idx)
            
            # Take action in environment
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            
            # Store reward and done signal
            memory.r_mb.append(reward)
            memory.done_mb.append(done)
            
            # Update episode reward
            ep_reward += reward
        
        # Get final makespan
        makespan = env.LBs.max()
        
        # Subtract positive rewards
        ep_reward -= env.posRewards
        
        # Update policy
        loss, v_loss = ppo.update([memory], n_j * n_m, configs.graph_pool_type)
        memory.clear_memory()
        
        # Save history
        rewards_history.append(ep_reward)
        loss_history.append(loss)
        makespans_history.append(makespan)
        
        # Log progress
        episode_time = time.time() - episode_start
        
        if (episode + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            avg_makespan = np.mean(makespans_history[-100:]) if len(makespans_history) >= 100 else np.mean(makespans_history)
            
            print(f"Episode {episode+1}/{n_episodes} | Reward: {ep_reward:.2f} | "
                  f"Makespan: {makespan:.2f} | Loss: {loss:.6f}")
            print(f"Avg(100): Reward={avg_reward:.2f}, Makespan={avg_makespan:.2f} | "
                  f"Episode time: {episode_time:.2f}s | Total time: {elapsed:.2f}s")
            
            # Save training stats
            np.savez(
                os.path.join(CONFIG["output_dir"], "training_stats.npz"),
                rewards=rewards_history,
                losses=loss_history,
                makespans=makespans_history
            )
            
            # Plot learning curves (once we have enough data)
            if len(rewards_history) >= 100:
                plot_learning_curves(rewards_history, loss_history, makespans_history, figures_dir)
        
        # Save model checkpoint
        if (episode + 1) % save_every == 0 or episode == n_episodes - 1:
            model_path = os.path.join(models_dir, f"l2d_{n_j}x{n_m}_episode_{episode+1}.pth")
            torch.save(ppo.policy.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"l2d_{n_j}x{n_m}_final.pth")
    torch.save(ppo.policy.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return ppo.policy, final_model_path

def plot_learning_curves(rewards, losses, makespans, figures_dir):
    """Plot and save training curves."""
    # Calculate moving averages for smoother visualization
    window = 100
    
    # Calculate moving averages
    reward_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    makespan_avg = [np.mean(makespans[max(0, i-window):i+1]) for i in range(len(makespans))]
    
    # Plot smoothed rewards
    plt.figure(figsize=(10, 6))
    plt.plot(reward_avg)
    plt.title(f'Rewards During Training ({window}-episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "rewards_smoothed.png"))
    plt.close()
    
    # Plot smoothed makespans
    plt.figure(figsize=(10, 6))
    plt.plot(makespan_avg)
    plt.title(f'Makespan During Training ({window}-episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Makespan')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "makespans_smoothed.png"))
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