from JSSP_Env import SJSSP
from mb_agg import g_pool_cal
from agent_utils import greedy_select_action
import numpy as np
import torch
from Params import configs

def validate_weighted(vali_set, model, feature_set):
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
    
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set = feature_set)
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    
   # Metrics to track (parallel to original validate function)
    weighted_sums = []
    reward_derived = []
    improvement_pct = []
    
    # Print some validation set details for debugging
    print(f"Validating on {len(vali_set)} instances")
    
    # Rollout using model
    for idx, data in enumerate(vali_set):
         # Extract weights from the last column of the weight matrix
        times = data[0]
        machines = data[1]
        weight_matrix = data[2]
        weights = weight_matrix[:, -1]  # Extract the last column
        
        # Debug print for first few instances
        if idx < 3:
            print(f"\nInstance {idx} weights (first 3): {weights[:3]}")
        
        # Create a new data tuple with the extracted weights
        data = (times, machines, weights)
        adj, fea, candidate, mask = env.reset(data)
        
        # Store initial weighted sum estimate for calculating improvement later
        initial_weighted_sum = env._calculate_weighted_sum_estimate()
        
        if idx < 3:
            print(f"Initial weighted sum: {initial_weighted_sum:.2f}")
        
        # Track cumulative reward for reward-derived metric
        total_reward = - env.initQuality
        
        steps = 0
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
            total_reward += reward
            steps += 1
        
        # Calculate final metrics
        final_weighted_sum = env.weighted_sum
        reward_based_metric = total_reward - env.posRewards
        # Calculate percentage improvement (negative because we want to minimize weighted sum)
        # Handle zero division case
        if initial_weighted_sum != 0:
            percent_improvement = ((initial_weighted_sum - final_weighted_sum) / initial_weighted_sum) * 100
        else:
            percent_improvement = 0
            
        # Debug print for first few instances
        if idx < 3:
            print(f"Final weighted sum: {final_weighted_sum:.2f}")
            print(f"Total steps: {steps}")
            print(f"Reward-derived: {reward_based_metric:.2f}")
            print(f"Improvement: {percent_improvement:.2f}%")
            
        # Store all metrics
        weighted_sums.append(final_weighted_sum)
        reward_derived.append(reward_based_metric)
        improvement_pct.append(percent_improvement)
        
    # Compute statistics
    ws_array = np.array(weighted_sums)
    print(f"\nWeighted sum statistics:")
    print(f"Mean: {ws_array.mean():.2f}")
    print(f"Min: {ws_array.min():.2f}")
    print(f"Max: {ws_array.max():.2f}")
    print(f"Std: {ws_array.std():.2f}")
    
    # Return dictionary with all metrics, with negative weighted_sums for consistency
    return {
        'weighted_sum': np.array(weighted_sums),
        'reward_derived': np.array(reward_derived),
        'improvement_pct': np.array(improvement_pct)
    }
    
def compare_validation_methods(vali_set, model, feature_set):
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
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set = feature_set)
    
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
    validation_results = validate_weighted(vali_set, model, feature_set=feature_set)
    results["L2D"] = validation_results['weighted_sum'].tolist() 
    
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
        
    # Calculate your validation_results dictionary
    comparison_metrics = {
        'win_rate': (win_counts["L2D"] / len(vali_set)) * 100 if len(vali_set) > 0 else 0,
        'win_vs_spt': (win_counts["L2D"] / (win_counts["L2D"] + win_counts["SPT"])) * 100 
                     if (win_counts["L2D"] + win_counts["SPT"]) > 0 else 0,
        'win_vs_wspt': (win_counts["L2D"] / (win_counts["L2D"] + win_counts["WSPT"])) * 100 
                      if (win_counts["L2D"] + win_counts["WSPT"]) > 0 else 0,
        'avg_weighted_sum': averages["L2D"],
        'improvement_over_spt': ((averages["SPT"] - averages["L2D"]) / averages["SPT"]) * 100 
                               if averages["SPT"] != 0 else 0,
        'improvement_over_wspt': ((averages["WSPT"] - averages["L2D"]) / averages["WSPT"]) * 100 
                                if averages["WSPT"] != 0 else 0,
    }
    
    return comparison_metrics