import time
import numpy as np
import torch
from src.environments.JSSP_Env import SJSSP
from src.utils.mb_agg import g_pool_cal
from src.utils.agent_utils import greedy_select_action
from Params import configs
from src.utils.testing import *

def validate_weighted(vali_set, model, feature_set=None):
    """
    Validate the model on a set of instances for weighted sum objective with enhanced metrics.
    
    Args:
        vali_set: List of JSSP instances with weights
        model: L2D policy model
        feature_set: Set of features to use
        
    Returns:
        dict: Dictionary containing validation metrics including rewards and losses
    """
    # Set default feature set if not provided
    if feature_set is None:
        feature_set = ['LBs', 'finished_mark', 'normalized_weights']
    
    # Get dimensions from the first instance
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    
    # Initialize environment
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set=feature_set)
    
    # Setup device and graph pooling
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=device
    )
    
    # Metrics to track
    weighted_sums = []
    reward_derived = []
    improvement_pct = []
    value_losses = []    # For critic loss tracking
    policy_entropies = [] # For policy quality tracking
    
    # Rollout using model
    for idx, data in enumerate(vali_set):
        # Extract times and machines
        times = data[0]
        machines = data[1]
        
        # Extract weights - handle both formats
        if isinstance(data[2], np.ndarray) and len(data[2].shape) == 2:
            # Weight matrix format - weights are in the last column
            weight_matrix = data[2]
            weights = weight_matrix[:, -1]
        else:
            # Direct weights format
            weights = data[2]
        
        # Create instance
        instance = (times, machines, weights)
        adj, fea, candidate, mask = env.reset(instance)
        
        # Store initial weighted sum estimate for calculating improvement later
        initial_weighted_sum = env._calculate_weighted_sum_estimate()
        
        # Track cumulative reward for reward-derived metric
        total_reward = - env.initQuality
        
        # For value loss calculation
        instance_value_losses = []
        instance_entropies = []
        
        # Solve instance using policy
        while not env.done():
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            
            with torch.no_grad():
                pi, value = model(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                
                # Calculate policy entropy (a measure of confidence/uncertainty)
                # Lower entropy = more confident predictions
                log_pi = torch.log(pi + 1e-8)  # Add small epsilon to avoid log(0)
                entropy = -torch.sum(pi * log_pi)
                instance_entropies.append(entropy.item())
            
            # Select best action greedily
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            
            # Update cumulative reward
            total_reward += reward
            
            # Since we can't easily get the "true" target value for critic during validation,
            # we'll use a proxy approach: measure how consistent the value predictions are
            # by tracking the difference between consecutive value estimates
            if len(instance_value_losses) > 0:
                # Calculate "temporal difference error" between current and previous value
                prev_value = instance_value_losses[-1]
                # In PPO, the value function is trained to predict discounted returns
                # Here we use a simple TD(0) style comparison
                value_loss = ((value.item() - (prev_value + reward)) ** 2)
                instance_value_losses.append(value.item())
            else:
                # First step has no previous value
                instance_value_losses.append(value.item())
        
        # Calculate final metrics
        final_weighted_sum = env.weighted_sum
        reward_based_metric = total_reward - env.posRewards
        
        # Calculate percentage improvement
        if initial_weighted_sum != 0:
            percent_improvement = ((initial_weighted_sum - final_weighted_sum) / initial_weighted_sum) * 100
        else:
            percent_improvement = 0
            
        # Store metrics
        weighted_sums.append(final_weighted_sum)
        reward_derived.append(reward_based_metric)
        improvement_pct.append(percent_improvement)
        
        # Calculate average losses for this instance
        if len(instance_value_losses) > 1:  # Need at least 2 values to calculate losses
            value_losses.append(np.mean(instance_value_losses[1:]))  # Skip first value
        
        if instance_entropies:
            policy_entropies.append(np.mean(instance_entropies))
    
    # Compute statistics
    ws_array = np.array(weighted_sums)
    print(f"\nValidation weighted sum statistics:")
    print(f"Mean: {ws_array.mean():.2f}")
    print(f"Min: {ws_array.min():.2f}")
    print(f"Max: {ws_array.max():.2f}")
    print(f"Std: {ws_array.std():.2f}")
    
    # Print value loss statistics if available
    if value_losses:
        value_loss_mean = np.mean(value_losses)
        print(f"\nValidation value loss: {value_loss_mean:.4f}")
    
    # Print entropy statistics if available
    if policy_entropies:
        entropy_mean = np.mean(policy_entropies)
        print(f"Validation policy entropy: {entropy_mean:.4f}")
    
    # Return dictionary with all metrics
    result = {
        'weighted_sum': np.array(weighted_sums),
        'reward_derived': np.array(reward_derived),
        'improvement_pct': np.array(improvement_pct)
    }
    
    # Add loss metrics if available
    if value_losses:
        result['value_loss'] = np.mean(value_losses)
    
    if policy_entropies:
        result['policy_entropy'] = np.mean(policy_entropies)
        
    return result

def compare_validation_methods(vali_set, model, feature_set=None):
    """
    Compare L2D with baseline methods including SRPT on validation set.
    
    Args:
        vali_set: List of JSSP instances with weights
        model: L2D policy model
        feature_set: Set of features to use
        
    Returns:
        dict: Dictionary with performance metrics and raw weighted sum values
    """
    # Set default feature set if not provided
    if feature_set is None:
        feature_set = ['LBs', 'finished_mark', 'normalized_weights']
    
    # Get dimensions from the first instance
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    
    # Initialize environment
    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES, feature_set=feature_set)
    
    # Results dictionary for storing weighted sums
    results = {
        "L2D": [],
        "SPT": [],
        "WSPT": [],
        "SRPT": []
    }
    
    # Device setup
    device = torch.device(configs.device)
    
    # Evaluate baseline methods on all instances
    start_time = time.time()
    
    for instance_idx, instance_data in enumerate(vali_set):
        # Extract times and machines
        times = instance_data[0]
        machines = instance_data[1]
        
        # Extract weights - handle both formats
        if isinstance(instance_data[2], np.ndarray) and len(instance_data[2].shape) == 2:
            # Weight matrix format - weights are in the last column
            weight_matrix = instance_data[2]
            weights = weight_matrix[:, -1]
        else:
            # Direct weights format
            weights = instance_data[2]
        
        # Create instance
        instance = (times, machines, weights)
        
        # Test SPT, WSPT, and SRPT using their test functions
        _, spt_weighted_sum = test_spt(env, instance)
        results["SPT"].append(spt_weighted_sum)
        
        _, wspt_weighted_sum = test_wspt(env, instance)
        results["WSPT"].append(wspt_weighted_sum)
        
        _, srpt_weighted_sum = test_srpt(env, instance)
        results["SRPT"].append(srpt_weighted_sum)
        
        # Print progress every 10 instances
        if (instance_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Validated {instance_idx + 1}/{len(vali_set)} instances in {elapsed:.2f}s")
    
    # Use the original validate_weighted function for L2D
    validation_results = validate_weighted(vali_set, model, feature_set=feature_set)
    results["L2D"] = validation_results['weighted_sum'].tolist()
    
    # Calculate average metrics
    averages = {}
    win_counts = {"L2D": 0, "SPT": 0, "WSPT": 0, "SRPT": 0}
    baseline_weighted_sums = {}
    
    for method in results:
        # Convert to numpy array for easier calculations
        if isinstance(results[method], list):
            results[method] = np.array(results[method])
        
        averages[method] = np.mean(results[method])
        
        # Store baseline weighted sums for the return dictionary
        if method != "L2D":
            baseline_weighted_sums[f"baseline_{method.lower()}_weighted_sum"] = averages[method]
    
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
    
    # Calculate performance metrics
    total_instances = len(vali_set) if len(vali_set) > 0 else 1
    comparison_metrics = {
        # Win rate metrics for L2D (renamed for clarity)
        'l2d_win_rate': (win_counts["L2D"] / total_instances) * 100,
        
        # Head-to-head win rates
        'l2d_win_vs_spt': (win_counts["L2D"] / (win_counts["L2D"] + win_counts["SPT"])) * 100 
                     if (win_counts["L2D"] + win_counts["SPT"]) > 0 else 0,
        'l2d_win_vs_wspt': (win_counts["L2D"] / (win_counts["L2D"] + win_counts["WSPT"])) * 100 
                      if (win_counts["L2D"] + win_counts["WSPT"]) > 0 else 0,
        'l2d_win_vs_srpt': (win_counts["L2D"] / (win_counts["L2D"] + win_counts["SRPT"])) * 100 
                     if (win_counts["L2D"] + win_counts["SRPT"]) > 0 else 0,
        
        # Overall win rates for all methods (new)
        'win_rate_l2d': (win_counts["L2D"] / total_instances) * 100,
        'win_rate_spt': (win_counts["SPT"] / total_instances) * 100,
        'win_rate_wspt': (win_counts["WSPT"] / total_instances) * 100,
        'win_rate_srpt': (win_counts["SRPT"] / total_instances) * 100,
        
        # Weighted sum average
        'avg_weighted_sum': averages["L2D"],
        
        # Improvement percentages
        'improvement_over_spt': ((averages["SPT"] - averages["L2D"]) / averages["SPT"]) * 100 
                               if averages["SPT"] != 0 else 0,
        'improvement_over_wspt': ((averages["WSPT"] - averages["L2D"]) / averages["WSPT"]) * 100 
                                if averages["WSPT"] != 0 else 0,
        'improvement_over_srpt': ((averages["SRPT"] - averages["L2D"]) / averages["SRPT"]) * 100 
                               if averages["SRPT"] != 0 else 0
    }
    
    # For backward compatibility (to be phased out)
    comparison_metrics['win_rate'] = comparison_metrics['l2d_win_rate']
    comparison_metrics['win_vs_spt'] = comparison_metrics['l2d_win_vs_spt'] 
    comparison_metrics['win_vs_wspt'] = comparison_metrics['l2d_win_vs_wspt']
    comparison_metrics['win_vs_srpt'] = comparison_metrics['l2d_win_vs_srpt']
    
    # Include value loss and policy entropy if available
    if 'value_loss' in validation_results:
        comparison_metrics['value_loss'] = validation_results['value_loss']
    
    if 'policy_entropy' in validation_results:
        comparison_metrics['policy_entropy'] = validation_results['policy_entropy']
    
    # Merge baseline weighted sums into the metrics
    comparison_metrics.update(baseline_weighted_sums)
    
    # Also include the actual results for detailed analysis
    comparison_metrics['method_results'] = {
        method: values.tolist() if isinstance(values, np.ndarray) else values 
        for method, values in results.items()
    }
    
    # Add the win counts directly
    for method in win_counts:
        comparison_metrics[f'win_count_{method.lower()}'] = win_counts[method]
    
    return comparison_metrics