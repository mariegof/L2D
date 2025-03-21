
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd

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
    # Define a separate directory for test instances (distinct from validation data)
    TEST_DATA_DIR = './DataGen/TestData'
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    seed = 2025
    
    # Define path for test instances
    test_file_path = os.path.join(TEST_DATA_DIR, f'weightedData{n_j}_{n_m}_Seed{seed}.npy')
    
    # Initialize environment
    env = SJSSP(n_j=n_j, n_m=n_m, feature_set=feature_set)
    
    # Try to load existing test instances
    instances = []
    try:
        print(f"Attempting to load test instances from {test_file_path}")
        data_loaded = np.load(test_file_path)
        
        print(f"Successfully loaded data with shape {data_loaded.shape}")
        
        # Process the loaded data into the required format
        for i in range(len(data_loaded)):
            times = data_loaded[i][0]
            machines = data_loaded[i][1]
            weight_matrix = data_loaded[i][2]
            
            # Extract weights from the last column of weight_matrix
            weights = weight_matrix[:, -1]
            
            # Create instance tuple with the correct format
            instances.append((times, machines, weights))
        
        print(f"Successfully processed {len(instances)} test instances")
        
    except (FileNotFoundError, IOError) as e:
        print(f"Could not load test instances: {e}")
        print(f"Generating {n_instances} new test instances with seed {seed}...")
        
        # Set seed for reproducible instance generation
        np.random.seed(seed)
        
        # Generate instances following the exact format from the code snippet
        generated_data = []
        for _ in range(n_instances):
            times, machines, weights = weighted_instance_gen(
                n_j=n_j, n_m=n_m, low=low, high=high,
                weight_low=weight_low, weight_high=weight_high
            )
            
            # Create a matrix of zeros with the same shape as times
            weight_matrix = np.zeros((n_j, n_m), dtype=int)
            
            # Place the weights in the last column
            weight_matrix[:, -1] = weights
            
            # Store all components for saving
            generated_data.append(np.array([times, machines, weight_matrix]))
            
            # Add to instances list for immediate use
            instances.append((times, machines, weights))
        
        # Save the generated instances
        np.save(test_file_path, np.array(generated_data))
        print(f"Generated and saved {n_instances} test instances to {test_file_path}")
    
    # Use only the requested number of instances (in case the file has more)
    instances = instances[:n_instances]
    print(f"Using {len(instances)} instances for evaluation")
    
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
    
    stats_df = analyze_instance_characteristics(instances, results["L2D"], results["WSPT"])
    stats_df.to_csv(f"{results_dir}/instance_analysis.csv", index=False)
    visualize_instance_characteristics(stats_df, results_dir) 
    
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
        
        # Calculate machine contention metrics
        n_machines = int(np.max(machines))  # Get the highest machine number
        machine_loads = np.zeros(n_machines)
        for j in range(len(weights)):
            for m_idx, machine in enumerate(machines[j]):
                machine_loads[machine-1] += times[j][m_idx]  # Add processing time to machine load

        machine_load_std = np.std(machine_loads)  # Standard deviation of machine loads
        machine_load_ratio = np.max(machine_loads) / np.min(machine_loads) if np.min(machine_loads) > 0 else np.max(machine_loads)

        # Add bottleneck analysis - identify how many operations use the most loaded machine
        bottleneck_machine = np.argmax(machine_loads)
        bottleneck_operations = 0
        for j in range(len(weights)):
            for m_idx, machine in enumerate(machines[j]):
                if machine-1 == bottleneck_machine:
                    bottleneck_operations += 1
        
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
            'machine_load_std': machine_load_std,
            'machine_load_ratio': machine_load_ratio,
            'bottleneck_operations': bottleneck_operations,
        })
    
    # Convert to pandas DataFrame for easier analysis
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
    print(winner_stats[['weight_std', 'weight_ratio', 'gini_coefficient', 'wspt_ratio_std', 'machine_load_std', 'machine_load_ratio']])
    
    # Find patterns where L2D outperforms WSPT
    print("\nWhen L2D wins (top 5 by improvement margin):")
    l2d_wins = df[df['winner'] == 'L2D'].sort_values('improvement', ascending=False).head(5)
    print(l2d_wins[['instance', 'improvement', 'weight_std', 'gini_coefficient']])
    
    # Find patterns where WSPT outperforms L2D
    print("\nWhen WSPT wins (top 5 by negative improvement margin):")
    wspt_wins = df[df['winner'] == 'WSPT'].sort_values('improvement').head(5)
    print(wspt_wins[['instance', 'improvement', 'weight_std', 'gini_coefficient']])
    
    return df

def visualize_instance_characteristics(df, results_dir):
    """
    Create comprehensive visualizations of instance characteristics and their
    relationship with algorithm performance.
    
    Args:
        df: DataFrame with instance statistics from analyze_instance_characteristics
        results_dir: Directory to save visualizations
    """
    
    # Ensure visualization directory exists
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Feature correlation heatmap
    plt.figure(figsize=(14, 12))
    corr_columns = ['improvement', 'weight_mean', 'weight_std', 'weight_ratio', 'proc_time_mean', 
                    'proc_time_std', 'wspt_ratio_std', 'gini_coefficient']
    if 'machine_load_std' in df.columns:
        corr_columns.extend(['machine_load_std', 'machine_load_ratio', 'bottleneck_operations'])
        
    corr = df[corr_columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, vmin=-.8, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    plt.title('Correlation Between Instance Features and Algorithm Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 2. Box plots by winner
    feature_pairs = [
        ['weight_std', 'weight_ratio'],
        ['gini_coefficient', 'wspt_ratio_std'],
        ['proc_time_mean', 'proc_time_std']
    ]
    
    if 'machine_load_std' in df.columns:
        feature_pairs.append(['machine_load_std', 'machine_load_ratio'])
    
    for pair in feature_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, feature in enumerate(pair):
            sns.boxplot(x='winner', y=feature, data=df, ax=axes[i], hue='winner', legend=False)
            axes[i].set_title(f'{feature} by Winner', fontsize=14)
            axes[i].set_xlabel('Winner Algorithm', fontsize=12)
            axes[i].set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"boxplot_{'_'.join(pair)}.png"), dpi=300)
        plt.close()
    
    # 3. Scatter plots of key relationships
    key_features = ['weight_std', 'weight_ratio', 'gini_coefficient', 'wspt_ratio_std']
    if 'machine_load_std' in df.columns:
        key_features.append('machine_load_std')
    
    for feature in key_features:
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(x=feature, y='improvement', hue='winner', 
                                 data=df, palette='viridis', s=100, alpha=0.7)
        
        # Add regression line
        sns.regplot(x=feature, y='improvement', data=df, scatter=False, 
                   line_kws={"color": "red", "alpha": 0.5, "lw": 2})
        
        plt.title(f'Impact of {feature.replace("_", " ").title()} on L2D vs WSPT Performance', fontsize=16)
        plt.xlabel(feature.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Improvement % (L2D vs WSPT)', fontsize=14)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"scatter_{feature}_improvement.png"), dpi=300)
        plt.close()
    
    # 4. Weight distribution histograms by winner
    plt.figure(figsize=(14, 10))
    g = sns.FacetGrid(df, col="winner", height=5, aspect=1.2)
    g.map(sns.histplot, "weight_std", kde=True)
    g.set_axis_labels("Weight Standard Deviation", "Count")
    g.set_titles("{col_name}")
    g.tight_layout()
    plt.savefig(os.path.join(viz_dir, "weight_distribution_by_winner.png"), dpi=300)
    plt.close()
    
    # 5. Relative performance by Gini coefficient (weight concentration)
    plt.figure(figsize=(12, 8))
    bins = np.linspace(df['gini_coefficient'].min(), df['gini_coefficient'].max(), 10)
    df['gini_bin'] = pd.cut(df['gini_coefficient'], bins)
    
    gini_grouped = df.groupby('gini_bin')['improvement'].mean().reset_index()
    plt.bar(range(len(gini_grouped)), gini_grouped['improvement'], 
            width=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(gini_grouped))))
    
    plt.xticks(range(len(gini_grouped)), [f"{b.left:.2f}-{b.right:.2f}" for b in gini_grouped['gini_bin']], 
              rotation=45, ha='right')
    plt.title('Mean L2D vs WSPT Improvement by Weight Concentration (Gini Coefficient)', fontsize=16)
    plt.xlabel('Gini Coefficient Range', fontsize=14)
    plt.ylabel('Mean Improvement %', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "improvement_by_gini.png"), dpi=300)
    plt.close()
    
    # 6. Create summary visualization
    plt.figure(figsize=(16, 12))
    
    # Main summary chart
    summary_data = df.groupby('winner').agg({
        'instance': 'count',
        'improvement': 'mean',
        'weight_std': 'mean',
        'weight_ratio': 'mean',
        'gini_coefficient': 'mean',
        'wspt_ratio_std': 'mean'
    }).reset_index()
    
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    winner_counts = summary_data['instance']
    bars = plt.bar(summary_data['winner'], winner_counts, color=['#3498db', '#e74c3c'])
    plt.title('Instance Count by Winner with Key Characteristics', fontsize=16)
    plt.ylabel('Number of Instances', fontsize=14)
    
    # Add percentages
    total = winner_counts.sum()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height/total*100:.1f}%', ha='center', fontsize=12)
    
    # Add key stats as table
    cell_text = []
    for _, row in summary_data.iterrows():
        cell_text.append([
            f"{row['weight_std']:.2f}",
            f"{row['weight_ratio']:.2f}",
            f"{row['gini_coefficient']:.2f}",
            f"{row['wspt_ratio_std']:.2f}"
        ])
    
    table = plt.table(cellText=cell_text,
                      rowLabels=summary_data['winner'],
                      colLabels=['Weight StdDev', 'Weight Ratio', 'Gini Coef', 'WSPT Ratio StdDev'],
                      loc='bottom', bbox=[0.0, -0.5, 1.0, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.tight_layout()
    
    # Performance improvement distribution
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    sns.histplot(df['improvement'], kde=True, ax=ax2)
    ax2.set_title('Distribution of L2D vs WSPT Improvement %', fontsize=14)
    ax2.set_xlabel('Improvement %', fontsize=12)
    ax2.axvline(x=0, color='red', linestyle='--')
    
    # Scatter of most predictive feature
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    best_feature = corr.loc['improvement'].drop('improvement').abs().idxmax()
    sns.scatterplot(x=best_feature, y='improvement', hue='winner', data=df, ax=ax3)
    ax3.set_title(f'Most Predictive Feature: {best_feature.replace("_", " ").title()}', fontsize=14)
    ax3.set_xlabel(best_feature.replace('_', ' ').title(), fontsize=12)
    ax3.set_ylabel('Improvement %', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(viz_dir, "summary_visualization.png"), dpi=300)
    plt.close()
    
    print(f"All visualizations saved to {viz_dir}")
