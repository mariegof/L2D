#!/usr/bin/env python
"""
Test script to verify WandB integration for the L2D-weighted project.
This script performs a short training run with a small problem instance
and verifies that all necessary metrics and plots are properly logged to WandB.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from pathlib import Path
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.JSSP_Env import SJSSP
from src.agents.PPO_jssp import PPO, Memory
from src.utils.mb_agg import g_pool_cal
from src.utils.agent_utils import select_action
from src.utils.instance_gen import weighted_instance_gen
from src.utils.validation import validate_weighted, compare_validation_methods
from Params import configs

# DIRECT PARAMETERS - Edit these instead of using command line arguments 
PARAMS = {
    "project": "jssp-weighted-sum-test",  # WandB project name
    "entity": None,                       # WandB entity (username) - set to None if not using
    "episodes": 5,                        # Number of training episodes to run
    "n_j": 6,                             # Number of jobs
    "n_m": 6,                             # Number of machines
    "weighted": True,                     # Use weighted instances
    "output_dir": "./results/wandb_test", # Output directory
    "run_name": "wandb-test",             # Name for this test run
    "feature_set": ['LBs', 'finished_mark', 'normalized_weights'],  # Features to use
    "reward_strategy": 'default',         # Reward function to use
    "save_checkpoint_every": 2,           # Save checkpoint every N episodes
    "save_best_model": True               # Save best model based on validation
}

def create_test_instances(n_j, n_m, num_instances=10):
    """Create a small set of test instances."""
    np.random.seed(42)  # Use fixed seed for reproducibility
    instances = []
    
    for _ in range(num_instances):
        instance = weighted_instance_gen(
            n_j=n_j, 
            n_m=n_m, 
            low=configs.low, 
            high=configs.high,
            weight_low=configs.weight_low, 
            weight_high=configs.weight_high
        )
        instances.append(instance)
    
    return instances

def log_learning_curves(rewards, weighted_sums, losses):
    """Create learning curve plots for WandB logging with both raw and smoothed lines."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Calculate smoothing window size (min 2, max 20% of data points)
    window_size = max(2, min(len(rewards) // 5, 10))
    
    # Plot rewards
    episodes = list(range(len(rewards)))
    axs[0].plot(episodes, rewards, 'b-', alpha=0.4, label='Raw')
    
    # Add smoothed line if we have enough data points
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = list(range(window_size-1, len(rewards)))
        axs[0].plot(smoothed_episodes, smoothed_rewards, 'b-', linewidth=2, label='Smoothed')
    
    axs[0].set_title('Training Rewards')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Plot weighted sums
    axs[1].plot(episodes, weighted_sums, 'r-', alpha=0.4, label='Raw')
    
    # Add smoothed line if we have enough data points
    if len(weighted_sums) >= window_size:
        smoothed_ws = np.convolve(weighted_sums, np.ones(window_size)/window_size, mode='valid')
        axs[1].plot(smoothed_episodes, smoothed_ws, 'r-', linewidth=2, label='Smoothed')
    
    axs[1].set_title('Weighted Sum')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Weighted Sum')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # Plot losses
    axs[2].plot(episodes, losses, 'g-', alpha=0.4, label='Raw')
    
    # Add smoothed line if we have enough data points
    if len(losses) >= window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        axs[2].plot(smoothed_episodes, smoothed_losses, 'g-', linewidth=2, label='Smoothed')
    
    axs[2].set_title('Training Loss')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    plt.tight_layout()
    return fig

def run_test():
    """Run a short training test with WandB logging."""
    print("Starting WandB integration test...")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    run_dir = os.path.join(PARAMS["output_dir"], f"{timestamp}_{PARAMS['run_name']}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create model directories if they don't exist
    checkpoint_dir = os.path.join("models", "checkpoints", f"{timestamp}_{PARAMS['run_name']}")
    best_model_dir = os.path.join("models", "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Save parameters to the run directory
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(PARAMS, f, indent=2)
    
    # Initialize WandB
    wandb.init(
        project=PARAMS["project"],
        entity=PARAMS["entity"],
        name=f"{PARAMS['run_name']}-{PARAMS['n_j']}x{PARAMS['n_m']}",
        config={
            "n_j": PARAMS["n_j"],
            "n_m": PARAMS["n_m"],
            "episodes": PARAMS["episodes"],
            "feature_set": PARAMS["feature_set"],
            "reward_strategy": PARAMS["reward_strategy"],
            "test_run": True,
            "weighted": PARAMS["weighted"]
        }
    )
    
    # Set device
    device = torch.device(configs.device)
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Initialize environment
    env = SJSSP(
        n_j=PARAMS["n_j"], 
        n_m=PARAMS["n_m"], 
        feature_set=PARAMS["feature_set"],
        reward_strategy=PARAMS["reward_strategy"]
    )
    
    # Initialize memory for PPO
    memory = Memory()
    
    # Initialize PPO agent
    agent = PPO(
        lr=configs.lr, 
        gamma=configs.gamma, 
        k_epochs=configs.k_epochs, 
        eps_clip=configs.eps_clip,
        n_j=PARAMS["n_j"],
        n_m=PARAMS["n_m"],
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
        batch_size=torch.Size([1, PARAMS["n_j"]*PARAMS["n_m"], PARAMS["n_j"]*PARAMS["n_m"]]),
        n_nodes=PARAMS["n_j"]*PARAMS["n_m"],
        device=device
    )
    
    # Create test instances
    validation_data = create_test_instances(PARAMS["n_j"], PARAMS["n_m"])
    print(f"Created {len(validation_data)} test instances")
    
    # Initialize tracking metrics
    rewards_history = []
    weighted_sums_history = []
    loss_history = []
    best_win_rate = 0.0
    
    # Run short training loop
    print(f"Running {PARAMS['episodes']} training episodes...")
    
    for episode in range(PARAMS["episodes"]):
        # Generate new instance
        instance = weighted_instance_gen(
            n_j=PARAMS["n_j"], 
            n_m=PARAMS["n_m"], 
            low=configs.low, 
            high=configs.high,
            weight_low=configs.weight_low, 
            weight_high=configs.weight_high
        )
        
        # Reset environment
        adj, fea, candidate, mask = env.reset(instance)
        ep_reward = -env.initQuality
        
        # Training loop for one episode
        while not env.done():
            # Process state
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            
            # Get action
            with torch.no_grad():
                pi, _ = agent.policy_old(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                
                action, a_idx = select_action(pi, candidate, memory)
            
            # Store experiences
            memory.adj_mb.append(adj_tensor)
            memory.fea_mb.append(fea_tensor)
            memory.candidate_mb.append(candidate_tensor)
            memory.mask_mb.append(mask_tensor)
            memory.a_mb.append(a_idx)
            
            # Take step
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            ep_reward += reward
            memory.r_mb.append(reward)
            memory.done_mb.append(done)
        
        # Update policy
        loss, v_loss, p_loss, ent_loss = agent.update([memory], PARAMS["n_j"] * PARAMS["n_m"], configs.graph_pool_type, return_all_losses=True)
        memory.clear_memory()
        
        # Final results
        ep_reward -= env.posRewards
        weighted_sum = env.weighted_sum
        
        # Store history
        rewards_history.append(ep_reward)
        weighted_sums_history.append(weighted_sum)
        loss_history.append(loss)
        
        # Log to WandB
        wandb.log({
            "episode": episode,
            "reward": ep_reward,
            "weighted_sum": weighted_sum,
            "loss": loss,
            "value_loss": v_loss,
            "policy_loss": p_loss,
            "entropy_loss": ent_loss
        })
        
        print(f"Episode {episode+1}/{PARAMS['episodes']} | Reward: {ep_reward:.2f} | "
              f"Weighted Sum: {weighted_sum:.2f} | Loss: {loss:.6f}")
        
        # Save checkpoint if needed
        if (episode + 1) % PARAMS["save_checkpoint_every"] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode+1}.pth")
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save with WandB
            wandb.save(checkpoint_path)
    
    # Run validation after training
    print("Running validation...")
    validation_results = validate_weighted(validation_data[:5], agent.policy, 
                                           PARAMS["feature_set"])
    validation_weighted_sum_mean = validation_results['weighted_sum'].mean()
    
    # Compare with baselines
    comparison_metrics = compare_validation_methods(validation_data[:5], agent.policy, 
                                                   PARAMS["feature_set"])
    
    # Save validation results
    validation_file = os.path.join(run_dir, "validation_results.json")
    with open(validation_file, "w") as f:
        json.dump({
            "weighted_sum_mean": float(validation_weighted_sum_mean),
            "improvement_pct_mean": float(validation_results['improvement_pct'].mean()),
            "win_rate": comparison_metrics['win_rate'],
            "win_vs_spt": comparison_metrics['win_vs_spt'],
            "win_vs_wspt": comparison_metrics['win_vs_wspt'],
            "improvement_over_spt": comparison_metrics['improvement_over_spt'],
            "improvement_over_wspt": comparison_metrics['improvement_over_wspt']
        }, f, indent=2)
    
    # Log validation metrics
    wandb.log({
        "validation_weighted_sum": validation_weighted_sum_mean,
        "validation_improvement_pct": validation_results['improvement_pct'].mean(),
        "validation_win_rate": comparison_metrics['win_rate'],
        "validation_win_vs_spt": comparison_metrics['win_vs_spt'],
        "validation_win_vs_wspt": comparison_metrics['win_vs_wspt'],
        "validation_improvement_over_spt": comparison_metrics['improvement_over_spt'],
        "validation_improvement_over_wspt": comparison_metrics['improvement_over_wspt']
    })
    
    # Save best model if this has the best win rate
    current_win_rate = comparison_metrics['win_rate']
    if PARAMS["save_best_model"] and current_win_rate > best_win_rate:
        best_win_rate = current_win_rate
        best_model_path = os.path.join(best_model_dir, f"l2d_weighted_{PARAMS['n_j']}x{PARAMS['n_m']}_test.pth")
        torch.save(agent.policy.state_dict(), best_model_path)
        print(f"Saved best model with win rate {best_win_rate}% to {best_model_path}")
        wandb.save(best_model_path)
    
    # Generate learning curve plots with raw and smoothed lines
    curves_fig = log_learning_curves(rewards_history, weighted_sums_history, loss_history)
    # Save figure locally
    curves_file = os.path.join(run_dir, "learning_curves.png")
    curves_fig.savefig(curves_file, dpi=300, bbox_inches='tight')
    # Log to WandB
    wandb.log({"learning_curves": wandb.Image(curves_fig)})
    plt.close(curves_fig)
    
    # Create a summary file
    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"WandB Test Run: {PARAMS['run_name']}-{PARAMS['n_j']}x{PARAMS['n_m']}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Episodes: {PARAMS['episodes']}\n\n")
        f.write(f"Final Weighted Sum: {weighted_sums_history[-1]:.2f}\n")
        f.write(f"Final Loss: {loss_history[-1]:.6f}\n\n")
        f.write("Validation Results:\n")
        f.write(f"  Mean Weighted Sum: {validation_weighted_sum_mean:.2f}\n")
        f.write(f"  Win Rate: {comparison_metrics['win_rate']:.2f}%\n")
        f.write(f"  Win vs SPT: {comparison_metrics['win_vs_spt']:.2f}%\n")
        f.write(f"  Win vs WSPT: {comparison_metrics['win_vs_wspt']:.2f}%\n")
        f.write(f"  Improvement vs SPT: {comparison_metrics['improvement_over_spt']:.2f}%\n")
        f.write(f"  Improvement vs WSPT: {comparison_metrics['improvement_over_wspt']:.2f}%\n\n")
        f.write(f"WandB URL: {wandb.run.get_url()}\n")
    
    print(f"Results saved to {run_dir}")
    print("WandB test completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    run_test()