
import numpy as np
import torch

# Import from the original implementation
from JSSP_Env import SJSSP
from mb_agg import g_pool_cal
from Params import configs
from PPO_jssp_multiInstances import PPO

def test_3x3_weighted_instance():
    """Test different scheduling strategies on a fixed 3x3 instance."""
    # Define a simple 3x3 instance
    times = np.array([
        [3, 2, 4],  # Job 1
        [1, 4, 2],  # Job 2
        [3, 2, 1]   # Job 3
    ])
    
    machines = np.array([
        [0, 1, 2],  # Job 1 - machine sequence
        [1, 0, 2],  # Job 2 - machine sequence
        [2, 1, 0]   # Job 3 - machine sequence
    ])
    
    # Test with different weight combinations
    weight_scenarios = [
        np.array([1, 1, 1]),     # Equal weights
        np.array([3, 1, 1]),     # Job 1 high priority
        np.array([1, 3, 1]),     # Job 2 high priority 
        np.array([1, 1, 3]),     # Job 3 high priority
        np.array([3, 2, 1])      # Descending priority
    ]
    
    # Initialize environment
    env = SJSSP(n_j=3, n_m=3)
    
    # For testing L2D model
    model_path = "./weighted_objective_experiments/models/l2d_weighted_6x6_final.pth"  # Update with actual path
    device = torch.device(configs.device)
    
    # Initialize PPO agent to load model
    ppo = PPO(
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        n_j=3,
        n_m=3,
        num_layers=configs.num_layers,
        neighbor_pooling_type=configs.neighbor_pooling_type,
        input_dim=4,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic
    )
    
    try:
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
        l2d_available = True
        print(f"Loaded L2D model from {model_path}")
    except:
        l2d_available = False
        print(f"Could not load L2D model from {model_path}, skipping L2D evaluation")
    
    # Setup graph pooling for L2D
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, 9, 9]),  # 3x3 = 9 operations
        n_nodes=9,
        device=device
    )
    
    # For tracking results
    all_results = {}
    
    for i, weights in enumerate(weight_scenarios):
        scenario_name = f"Scenario {i+1}: Weights {weights}"
        print(f"\n{scenario_name}")
        
        instance = (times, machines, weights)
        results = {}
        
        # Test SPT
        adj, fea, candidate, mask = env.reset(instance)
        sequence = []
        while not env.done():
            eligible_ops = candidate[~mask]
            proc_times = []
            for op in eligible_ops:
                job_idx = op // env.number_of_machines
                op_idx = op % env.number_of_machines
                proc_times.append(env.dur[job_idx, op_idx])
            action_idx = np.argmin(np.array(proc_times))
            action = eligible_ops[action_idx]
            sequence.append(action)
            adj, fea, reward, done, candidate, mask = env.step(action)
        results["SPT"] = {"weighted_sum": env.weighted_sum, "sequence": sequence}
        
        # Test WSPT
        adj, fea, candidate, mask = env.reset(instance)
        sequence = []
        while not env.done():
            eligible_ops = candidate[~mask]
            wspt_values = []
            for op in eligible_ops:
                job_idx = op // env.number_of_machines
                op_idx = op % env.number_of_machines
                wspt_values.append(env.weights[job_idx] / env.dur[job_idx, op_idx])
            action_idx = np.argmax(np.array(wspt_values))
            action = eligible_ops[action_idx]
            sequence.append(action)
            adj, fea, reward, done, candidate, mask = env.step(action)
        results["WSPT"] = {"weighted_sum": env.weighted_sum, "sequence": sequence}
        
        # Test L2D if available
        if l2d_available:
            adj, fea, candidate, mask = env.reset(instance)
            sequence = []
            while not env.done():
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        x=fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                
                # Select best action
                _, idx = pi.max(1)
                action = candidate[idx.item()]
                sequence.append(action)
                adj, fea, reward, done, candidate, mask = env.step(action)
            results["L2D"] = {"weighted_sum": env.weighted_sum, "sequence": sequence}
        
        # Print results for this scenario
        print("Method | Weighted Sum | Sequence")
        print("-" * 50)
        for method, result in results.items():
            seq_str = ", ".join([f"{op//3}-{op%3}" for op in result["sequence"]])
            print(f"{method.ljust(6)} | {result['weighted_sum']:12.2f} | {seq_str}")
        
        all_results[scenario_name] = results
    
    return all_results

if __name__ == "__main__":
    test_3x3_weighted_instance()
    