import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.JSSP_Env import SJSSP
from src.agents.PPO_jssp import PPO
from src.utils.mb_agg import g_pool_cal
from src.utils.agent_utils import greedy_select_action
from src.utils.instance_gen import weighted_instance_gen
from Params import configs

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained L2D model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to configuration file')
    parser.add_argument('--n_instances', type=int, default=100, help='Number of test instances')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for test instances')
    parser.add_argument('--output', type=str, default='./data/results', help='Output directory for results')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_l2d(env, instance, policy, device):
    """Apply trained L2D model to solve instance and return weighted sum objective."""
    times, machines, weights = instance
    
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=device
    )
    
    adj, fea, candidate, mask = env.reset(instance)
    
    total_reward = -env.initQuality
    
    while not env.done():
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        with torch.no_grad():
            pi, _ = policy(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
        
        # Select best action greedily
        action = greedy_select_action(pi, candidate)
        adj, fea, reward, done, candidate, mask = env.step(action.item())
        total_reward += reward
    
    return env.weighted_sum, total_reward - env.posRewards

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

def analyze_instance_characteristics(instances, l2d_results, wspt_results):
    """Analyze characteristics of instances and their relationship to algorithm performance."""
    stats = []
    
    for i, instance in enumerate(instances):
        times, machines, weights = instance
        
        # Calculate basic statistics
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        weight_ratio = np.max(weights) / np.min(weights) if np.min(weights) > 0 else np.max(weights)
        
        # Calculate processing time statistics
        proc_time_mean = np.mean(times)
        proc_time_std = np.std(times)
        
        # Calculate WSPT ratios (weight/processing_time)
        wspt_ratios = []
        for j in range(len(weights)):
            job_proc_time_total = np.sum(times[j])
            wspt_ratios.append(weights[j] / job_proc_time_total)
        
        wspt_ratio_std = np.std(wspt_ratios)
        
        # Calculate weight concentration (Gini coefficient)
        sorted_weights = np.sort(weights)
        cumsum_weights = np.cumsum(sorted_weights)
        gini = 1 - 2 * np.sum((cumsum_weights - sorted_weights/2) / cumsum_weights[-1]) / len(weights)
        
        # Compare algorithm performance
        l2d_score = l2d_results[i]
        wspt_score = wspt_results[i]
        winner = "L2D" if l2d_score < wspt_score else "WSPT" if wspt_score < l2d_score else "TIE"
        improvement = ((wspt_score - l2d_score) / wspt_score) * 100 if wspt_score != 0 else 0
        
        stats.append({
            'instance': i,
            'winner': winner,
            'improvement': improvement,
            'weight_mean': weight_mean,
            'weight_std': weight_std,
            'weight_ratio': weight_ratio,
            'proc_time_mean': proc_time_mean,
            'proc_time_std': proc_time_std,
            'wspt_ratio_std': wspt_ratio_std,
            'gini_coefficient': gini,
            'l2d_score': l2d_score,
            'wspt_score': wspt_score,
        })
    
    # Convert to pandas DataFrame for easier analysis
    try:
        import pandas as pd
        df = pd.DataFrame(stats)
        
        # Print summary statistics
        print("\n===== Instance Characteristics Analysis =====")
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"Average weight mean: {df['weight_mean'].mean():.2f}")
        print(f"Average weight std: {df['weight_std'].mean():.2f}")
        print(f"Average weight ratio (max/min): {df['weight_ratio'].mean():.2f}")
        print(f"Average Gini coefficient: {df['gini_coefficient'].mean():.2f}")
        
        # Statistics by winner
        print("\nCharacteristics by Winner:")
        winner_stats = df.groupby('winner').mean()
        print(winner_stats[['weight_std', 'weight_ratio', 'gini_coefficient', 'wspt_ratio_std']])
        
        return df
    except ImportError:
        print("Pandas not available for detailed analysis.")
        return stats

def plot_performance_profile(ratios, output_dir):
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
    plt.savefig(os.path.join(output_dir, "weighted_performance_profile.png"))
    plt.close()
    
    print(f"Performance profile saved to {output_dir}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device(configs.device)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize environment
    feature_set = config.get('feature_set', ['LBs', 'finished_mark', 'normalized_weights'])
    env = SJSSP(n_j=config.get('n_j', 6), n_m=config.get('n_m', 6), feature_set=feature_set)
    
    # Load policy
    policy = PPO(
        lr=config.get('lr', 2e-5),
        gamma=config.get('gamma', 1.0),
        k_epochs=config.get('k_epochs', 10),
        eps_clip=config.get('eps_clip', 0.1),
        n_j=config.get('n_j', 6),
        n_m=config.get('n_m', 6),
        num_layers=config.get('num_layers', 3),
        neighbor_pooling_type=config.get('neighbor_pooling_type', 'sum'),
        input_dim=config.get('input_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        num_mlp_layers_feature_extract=config.get('num_mlp_layers_feature_extract', 2),
        num_mlp_layers_actor=config.get('num_mlp_layers_actor', 2),
        hidden_dim_actor=config.get('hidden_dim_actor', 32),
        num_mlp_layers_critic=config.get('num_mlp_layers_critic', 2),
        hidden_dim_critic=config.get('hidden_dim_critic', 32)
    ).policy
    
    policy.load_state_dict(torch.load(args.model, map_location=device))
    policy.eval()
    
    print(f"Model loaded from {args.model}")
    
    # Generate or load test instances
    n_j = config.get('n_j', 6)
    n_m = config.get('n_m', 6)
    low = config.get('low', 1)
    high = config.get('high', 99)
    weight_low = config.get('weight_low', 1)
    weight_high = config.get('weight_high', 10)
    
    # Define path for test instances
    test_file_path = f'./data/instances/variable_weights/test_data_{n_j}x{n_m}_seed{args.seed}.npy'
    
    try:
        # Try to load existing test instances
        data_loaded = np.load(test_file_path)
        instances = []
        
        for i in range(min(len(data_loaded), args.n_instances)):
            times = data_loaded[i][0]
            machines = data_loaded[i][1]
            weights = data_loaded[i][2]
            instances.append((times, machines, weights))
        
        print(f"Loaded {len(instances)} test instances from {test_file_path}")
        
    except:
        # Generate new test instances
        print(f"Generating {args.n_instances} test instances...")
        instances = []
        
        for _ in range(args.n_instances):
            instance = weighted_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high,
                                            weight_low=weight_low, weight_high=weight_high)
            instances.append(instance)
        
        # Save test instances
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        test_data = np.array([np.array([instance[0], instance[1], instance[2]]) for instance in instances])
        np.save(test_file_path, test_data)
        print(f"Saved test instances to {test_file_path}")
    
    # Test all methods on all instances
    results = {
        "L2D": [],
        "SPT": [],
        "WSPT": []
    }
    
    for i, instance in enumerate(instances):
        if (i + 1) % 10 == 0:
            print(f"Testing instance {i+1}/{len(instances)}")
        
        # Test L2D
        l2d_weighted_sum, _ = test_l2d(env, instance, policy, device)
        results["L2D"].append(l2d_weighted_sum)
        
        # Test SPT
        spt_weighted_sum = test_spt(env, instance)
        results["SPT"].append(spt_weighted_sum)
        
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
    for method in ["SPT", "WSPT"]:
        if method != "L2D":
            improvement = ((avg_results[method] - l2d_avg) / avg_results[method]) * 100
            print(f"L2D improvement over {method}: {improvement:.2f}%")
    
    # Calculate performance ratios
    ratios = defaultdict(list)
    
    for i in range(len(instances)):
        best_weighted_sum = min(results["L2D"][i], results["SPT"][i], results["WSPT"][i])
        
        for method in results:
            ratio = results[method][i] / best_weighted_sum
            ratios[method].append(ratio)
    
    # Get win statistics
    wins = {method: 0 for method in results}
    for i in range(len(instances)):
        best_method = min(results.keys(), key=lambda m: results[m][i])
        wins[best_method] += 1
    
    print("\nWin Statistics:")
    for method, count in wins.items():
        print(f"{method}: {count}/{len(instances)} ({count/len(instances)*100:.2f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save results
    results_file = os.path.join(args.output, f"weighted_comparison_results_{n_j}x{n_m}.npz")
    np.savez(results_file, **results)
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate performance profile
    plot_performance_profile(ratios, args.output)
    
    # Analyze instance characteristics
    stats_df = analyze_instance_characteristics(instances, results["L2D"], results["WSPT"])
    
    # Generate LaTeX table if pandas is available
    try:
        import pandas as pd
        
        # Create a summary table
        summary_data = {
            "Method": list(results.keys()),
            "Avg. Weighted Sum": [avg_results[method] for method in results.keys()],
            "Win Rate (\%)": [wins[method]/len(instances)*100 for method in results.keys()]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add improvement over WSPT
        for i, method in enumerate(summary_df["Method"]):
            if method != "WSPT":
                summary_df.loc[i, "Improvement over WSPT (\%)"] = ((avg_results["WSPT"] - avg_results[method]) / avg_results["WSPT"]) * 100
            else:
                summary_df.loc[i, "Improvement over WSPT (\%)"] = 0.0
        
        # Generate LaTeX table
        latex_table = summary_df.to_latex(index=False, float_format="%.2f")
        
        # Save LaTeX table
        latex_table_file = os.path.join(args.output, f"weighted_comparison_table_{n_j}x{n_m}.tex")
        with open(latex_table_file, "w") as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to {latex_table_file}")
        
        # Save instance characteristics to CSV
        if isinstance(stats_df, pd.DataFrame):
            stats_file = os.path.join(args.output, f"instance_characteristics_{n_j}x{n_m}.csv")
            stats_df.to_csv(stats_file, index=False)
            print(f"Instance characteristics saved to {stats_file}")
    
    except ImportError:
        print("Pandas not available for generating LaTeX tables.")

if __name__ == "__main__":
    main()