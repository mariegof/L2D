
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from the original implementation
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
from WS.ws_test import test_l2d_weighted
from uniform_instance_gen import weighted_instance_gen
from Params import configs
from WS.utils import setup_directories

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

def evaluate_policies(policy=None, model_path=None, conf=None, feature_set=None):
    """Evaluate policies on random instances for weighted sum objective."""
    print(f"\n{'='*50}")
    print(f"EVALUATING ON {conf['evaluation']['n_instances']} RANDOM WEIGHTED INSTANCES")
    print(f"{'='*50}")
    
    # Get configuration parameters
    n_j = configs.n_j
    n_m = configs.n_m
    low = configs.low
    high = configs.high
    weight_low = configs.weight_low
    weight_high = configs.weight_high
    n_instances = conf["evaluation"]["n_instances"]
    
    # Setup directories
    _, _, results_dir = setup_directories(conf)
    
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
    env = SJSSP(n_j=n_j, n_m=n_m, feature_set = feature_set)
    
    # Generate random weighted instances
    print(f"Generating {n_instances} random weighted instances...")
    instances = [
        weighted_instance_gen(
            n_j=n_j, n_m=n_m, 
            low=low, high=high,
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
    
    # Collect evaluation results
    evaluation_results = {
        'win_rate': wins["L2D"] / n_instances * 100,
        'win_vs_spt': (wins["L2D"] / (wins["L2D"] + wins["SPT"])) * 100 if (wins["L2D"] + wins["SPT"]) > 0 else 0,
        'win_vs_srpt': (wins["L2D"] / (wins["L2D"] + wins["SRPT"])) * 100 if (wins["L2D"] + wins["SRPT"]) > 0 else 0,
        'win_vs_wspt': (wins["L2D"] / (wins["L2D"] + wins["WSPT"])) * 100 if (wins["L2D"] + wins["WSPT"]) > 0 else 0,
        'avg_weighted_sum': avg_results["L2D"],
        'improvement_over_spt': ((avg_results["SPT"] - avg_results["L2D"]) / avg_results["SPT"]) * 100 if avg_results["SPT"] != 0 else 0,
        'improvement_over_srpt': ((avg_results["SRPT"] - avg_results["L2D"]) / avg_results["SRPT"]) * 100 if avg_results["SRPT"] != 0 else 0,
        'improvement_over_wspt': ((avg_results["WSPT"] - avg_results["L2D"]) / avg_results["WSPT"]) * 100 if avg_results["WSPT"] != 0 else 0,
        'win_counts': wins,
        'detailed_results': results
    }
    
    return results, ratios, evaluation_results

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