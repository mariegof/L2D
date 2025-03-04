# File: debug_training.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

# Import from the original implementation
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO, Memory
from mb_agg import g_pool_cal
from agent_utils import select_action
from Params import configs
from uniform_instance_gen import uni_instance_gen

def debug_training(
    n_j=3,            # Jobs - small for faster debugging
    n_m=3,            # Machines - small for faster debugging
    max_episodes=50,  # Limit for faster debugging
    debug_interval=5, # How often to print detailed stats 
    save_dir="debug_output"
):
    """
    Run training with comprehensive debugging and visualization.
    """
    print(f"Starting training debug for {n_j}x{n_m} instances")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(f"{save_dir}/tensorboard")
    
    # Use CPU for easier debugging (unless you need CUDA)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create a simple test instance for validation during training
    known_instance = (
        np.array([
            [3, 1, 2],  # Job 1 processing times
            [2, 3, 4],  # Job 2 processing times 
            [3, 2, 1]   # Job 3 processing times
        ]),
        np.array([
            [0, 1, 2],  # Job 1 machine order
            [0, 2, 1],  # Job 2 machine order
            [1, 0, 2]   # Job 3 machine order
        ])
    )
    # Known optimal makespan for this instance is 11
    
    # Initialize environment
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Initialize PPO agent (with smaller network for faster training)
    ppo = PPO(
        lr=2e-5,              # Slightly higher learning rate 
        gamma=1.0,            # No discount for JSSP
        k_epochs=1,           # Number of PPO optimization iterations
        eps_clip=0.2,         # PPO clipping parameter
        n_j=n_j,
        n_m=n_m,
        num_layers=2,
        neighbor_pooling_type="sum",
        input_dim=configs.input_dim,
        hidden_dim=64,
        num_mlp_layers_feature_extract=2,
        num_mlp_layers_actor=2,
        hidden_dim_actor=32,
        num_mlp_layers_critic=2,
        hidden_dim_critic=32
    )
    
    # Initialize graph pooling
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=device
    )
    
    # Lists to store metrics
    episode_rewards = []
    makespans = []
    losses = []
    value_losses = []
    policy_losses = []
    entropy_values = []
    param_changes = []
    validation_makespans = []
    
    # Keep copy of initial parameters to track changes
    initial_params = {name: param.clone().detach() for name, param in ppo.policy.named_parameters()}
    
    # Training loop
    print("\nStarting training loop...")
    print("=" * 50)
    start_time = time.time()
    
    for episode in range(max_episodes):
        episode_start = time.time()
        
        # Generate a random instance for this episode
        instance_data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=10)
        
        # Reset environment with this instance
        adj, fea, candidate, mask = env.reset(instance_data)
        
        # Initialize memory for this episode
        memory = Memory()
        
        # Initialize metrics for this episode
        ep_reward = -env.initQuality
        actions_taken = []
        
        # Print instance details
        if episode % debug_interval == 0:
            print(f"\n--- Episode {episode+1} ---")
            print(f"Initial makespan LB: {env.initQuality}")
            print(f"Initial candidates: {candidate}")
        
        # Run episode
        step = 0
        while True:
            step += 1
            
            # Convert state to tensors
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            
            # Get action from policy
            with torch.no_grad():
                pi, val = ppo.policy_old(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
            
            # Log action distribution for first step of episodes at debug_interval
            if step == 1 and episode % debug_interval == 0:
                probs = pi.squeeze().cpu().numpy()
                for i, cand in enumerate(candidate):
                    writer.add_scalar(f'Policy/Action_{cand}', probs[i], episode)
                print(f"Action distribution: min={probs.min():.4f}, max={probs.max():.4f}, "
                      f"mean={probs.mean():.4f}, std={probs.std():.4f}")
                print(f"Entropy: {-(probs * np.log(probs + 1e-8)).sum():.4f}")
                print(f"Value estimate: {val.item():.4f}")
            
            # Select action
            action, a_idx = select_action(pi, candidate, memory)
            actions_taken.append(action.item())
            
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
            
            # Print step details at debug_interval
            if episode % debug_interval == 0 and step <= 3:  # Just the first 3 steps
                print(f"Step {step}: Action={action.item()}, Reward={reward}, "
                      f"Makespan LB={env.LBs.max()}")
            
            if done:
                break
        
        # Subtract positive rewards
        ep_reward -= env.posRewards
        
        # Record metrics
        episode_rewards.append(ep_reward)
        makespan = env.LBs.max()
        makespans.append(makespan)
        
        # Update policy
        loss, v_loss, p_loss, ent = ppo.update([memory], n_j * n_m, configs.graph_pool_type, return_all_losses=True)
        losses.append(loss)
        value_losses.append(v_loss)
        policy_losses.append(p_loss)
        entropy_values.append(ent)
        
        # Calculate parameter changes
        total_param_change = 0
        for name, param in ppo.policy.named_parameters():
            if name in initial_params:
                change = torch.norm(param.data - initial_params[name]) / torch.norm(initial_params[name])
                total_param_change += change.item()
        param_changes.append(total_param_change)
        
        # Reset initial_params for incremental changes
        initial_params = {name: param.clone().detach() for name, param in ppo.policy.named_parameters()}
        
        # Validation on known instance
        if episode % 10 == 0 or episode == max_episodes - 1:
            adj, fea, candidate, mask = env.reset(known_instance)
            validation_reward = -env.initQuality
            validation_actions = []
            
            with torch.no_grad():
                while True:
                    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                    adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                    
                    pi, _ = ppo.policy(
                        x=fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                    
                    # Greedy action selection for evaluation
                    _, idx = pi.max(1)
                    action = candidate[idx.item()]
                    validation_actions.append(action)
                    
                    adj, fea, reward, done, candidate, mask = env.step(action)
                    validation_reward += reward
                    
                    if done:
                        break
            
            validation_makespan = env.LBs.max()
            validation_makespans.append(validation_makespan)
            
            print(f"\nValidation (Episode {episode+1}): Makespan={validation_makespan} (optimal is 11)")
            print(f"Validation actions: {validation_actions}")
        
        # Log metrics to tensorboard
        writer.add_scalar('Training/Reward', ep_reward, episode)
        writer.add_scalar('Training/Makespan', makespan, episode)
        writer.add_scalar('Training/Loss/Total', loss, episode)
        writer.add_scalar('Training/Loss/Value', v_loss, episode)
        writer.add_scalar('Training/Loss/Policy', p_loss, episode)
        writer.add_scalar('Training/Entropy', ent, episode)
        writer.add_scalar('Training/ParameterChange', total_param_change, episode)
        
        # Print episode summary
        episode_time = time.time() - episode_start
        print(f"Episode {episode+1}: Makespan={makespan}, Reward={ep_reward:.2f}, "
              f"Loss={loss:.4f}, Time={episode_time:.2f}s")
    
    # Training completed - generate plots
    print("\nTraining completed. Generating analysis plots...")
    
    # Makespan plot
    plt.figure(figsize=(10, 6))
    plt.plot(makespans)
    plt.title(f'Makespan During Training ({n_j}x{n_m})')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/makespan_curve.png")
    
    # Reward plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title(f'Rewards During Training ({n_j}x{n_m})')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/reward_curve.png")
    
    # Loss components plot
    plt.figure(figsize=(10, 6))
    plt.plot(value_losses, label='Value Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(entropy_values, label='Entropy')
    plt.title(f'Loss Components ({n_j}x{n_m})')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/loss_components.png")
    
    # Parameter change plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_changes)
    plt.title(f'Parameter Changes ({n_j}x{n_m})')
    plt.xlabel('Episode')
    plt.ylabel('Relative Change')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/parameter_changes.png")
    
    # Validation plot
    if validation_makespans:
        plt.figure(figsize=(10, 6))
        validation_episodes = [e for e in range(1, max_episodes+1) if e == 1 or e % 10 == 0 or e == max_episodes][:len(validation_makespans)]
        plt.plot(validation_episodes, validation_makespans)
        plt.axhline(y=11, color='r', linestyle='--', label='Optimal')
        plt.title(f'Validation Makespan ({n_j}x{n_m})')
        plt.xlabel('Episode')
        plt.ylabel('Makespan')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{save_dir}/validation_makespan.png")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Final validation makespan: {validation_makespans[-1]} (optimal is 11)")
    print(f"Best makespan during training: {min(makespans)}")
    
    # Save training stats to file
    np.savez(
        f"{save_dir}/training_stats.npz",
        episode_rewards=episode_rewards,
        makespans=makespans,
        value_losses=value_losses,
        policy_losses=policy_losses,
        entropy_values=entropy_values,
        param_changes=param_changes,
        validation_makespans=validation_makespans
    )
    
    # Return results for further analysis
    return {
        'episode_rewards': episode_rewards,
        'makespans': makespans,
        'value_losses': value_losses,
        'policy_losses': policy_losses, 
        'entropy_values': entropy_values,
        'param_changes': param_changes,
        'validation_makespans': validation_makespans
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug L2D training process")
    parser.add_argument("--jobs", type=int, default=3, help="Number of jobs")
    parser.add_argument("--machines", type=int, default=3, help="Number of machines")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--interval", type=int, default=5, help="Debug print interval")
    parser.add_argument("--outdir", type=str, default="debug_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Add to PPO_jssp_multiInstances.py to expose all losses:
    # In PPO class, modify the update method to return policy loss and entropy:
    #
    # def update(self, memories, n_tasks, g_pool, return_all_losses=False):
    #     ... existing code ...
    #     if return_all_losses:
    #         return loss_sum.mean().item(), vloss_sum.mean().item(), p_loss.mean().item(), ent_loss.mean().item()
    #     return loss_sum.mean().item(), vloss_sum.mean().item()
    
    results = debug_training(
        n_j=args.jobs,
        n_m=args.machines,
        max_episodes=args.episodes,
        debug_interval=args.interval,
        save_dir=args.outdir
    )