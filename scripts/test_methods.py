import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.JSSP_Env import SJSSP
from src.agents.PPO_jssp import PPO
from src.utils.testing import test_l2d, test_spt, test_wspt, test_srpt
from src.utils.instance_gen import weighted_instance_gen
from Params import configs

# DIRECT PARAMETERS - Edit these instead of using command line arguments
PARAMS = {
    "model_path": "models/l2d_weighted_6x6_best.pth",  # Path to the model
    "n_j": 6,                         # Number of jobs
    "n_m": 6,                         # Number of machines
    "n_instances": 100,               # Number of instances to test
    "seed": 300,                      # Random seed
    "weighted": False,                # Use weighted instances (True) or uniform weights (False)
    "output_dir": "./results/performance_test",  # Output directory
}

def plot_performance_profile(results, output_path):
    """Generate performance profile plot for the results."""
    # Calculate performance ratios
    ratios = defaultdict(list)
    best_values = []
    
    for i in range(len(results['L2D'])):
        best_value = min(results['L2D'][i], results['SPT'][i], 
                        results['WSPT'][i], results['SRPT'][i])
        best_values.append(best_value)
    
    for method in ['L2D', 'SPT', 'WSPT', 'SRPT']:
        ratios[method] = [results[method][i] / best_values[i] for i in range(len(best_values))]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for method in ['L2D', 'SPT', 'WSPT', 'SRPT']:
        sorted_ratios = np.sort(ratios[method])
        y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        plt.step(sorted_ratios, y, where='post', label=method)
    
    plt.title('Performance Profile for Weighted Sum Objective')
    plt.xlabel('Performance Ratio (τ)')
    plt.ylabel('Probability P(r_{p,s} ≤ τ)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return ratios

def generate_latex_table(methods, avg_results, wins, instance_count, params):
    """Generate LaTeX table with comprehensive metadata."""
    try:
        import pandas as pd
        
        # Create summary data
        summary_data = {
            "Method": list(methods),
            "Avg. Weighted Sum": [avg_results[method] for method in methods],
            "Win Count": [wins[method] for method in methods],
            "Win Rate (%)": [wins[method]/instance_count*100 for method in methods]
        }
        
        # Add improvement column
        for method in methods:
            if method != "WSPT":
                summary_data.setdefault("Improvement vs WSPT (%)", []).append(
                    ((avg_results["WSPT"] - avg_results[method]) / avg_results["WSPT"]) * 100
                )
            else:
                summary_data.setdefault("Improvement vs WSPT (%)", []).append(0.0)
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Extract model info from path
        model_name = os.path.basename(params["model_path"])
        
        # Generate LaTeX table with table environment
        latex_table = """\\begin{table}[ht]
\\centering
\\begin{tabular}{%s}
\\toprule
%s
\\midrule
%s
\\bottomrule
\\end{tabular}
\\caption{Performance comparison of dispatching rules on %d instances 
with %d$\\times$%d problem size and %s weights. 
Model: \\texttt{%s}.}
\\label{tab:results_%dx%d_%s}
\\end{table}
""" % ('l' + 'r' * (len(df.columns) - 1),  # Column formats
       ' & '.join(df.columns) + ' \\\\',  # Header row
       ' \\\\\n'.join([' & '.join([str(row[0])] + 
                                 [f"{x:.2f}" for x in row[1:]]) 
                       for _, row in df.iterrows()]),  # Data rows
       instance_count,  # Number of instances
       params["n_j"], params["n_m"],  # Problem size
       "variable" if params["weighted"] else "uniform",  # Weight type
       model_name,  # Model name
       params["n_j"], params["n_m"],  # Label
       "weighted" if params["weighted"] else "uniform")  # Label
        
        return latex_table
    except ImportError:
        print("Pandas not available - couldn't generate LaTeX table")
        return None

def main():
    # Create output directory
    os.makedirs(PARAMS["output_dir"], exist_ok=True)
    
    # Set random seed
    np.random.seed(PARAMS["seed"])
    
    # Set device
    device = torch.device(configs.device)
    
    # Load model
    model = PPO(
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
    ).policy
    
    model.load_state_dict(torch.load(PARAMS["model_path"], map_location=device))
    model.eval()
    print(f"Model loaded from {PARAMS['model_path']}")
    
    # Initialize environment
    env = SJSSP(n_j=PARAMS["n_j"], n_m=PARAMS["n_m"], 
               feature_set=['LBs', 'finished_mark', 'normalized_weights'])
    
    # Load or generate instances
    instance_type = "variable" if PARAMS["weighted"] else "uniform"
    test_file_path = f'./data/instances/{instance_type}_weights/test_data_{PARAMS["n_j"]}x{PARAMS["n_m"]}_seed{PARAMS["seed"]}.npy'
    
    try:
        # Try to load instances
        print(f"Attempting to load test instances from {test_file_path}")
        data_loaded = np.load(test_file_path)
        instances = []
        
        for i in range(min(len(data_loaded), PARAMS["n_instances"])):
            times = data_loaded[i][0]
            machines = data_loaded[i][1]
            
            # Handle different weight formats
            if len(data_loaded[i]) > 2:
                if isinstance(data_loaded[i][2], np.ndarray) and len(data_loaded[i][2].shape) > 1:
                    # Weight matrix format - weights in last column
                    weight_matrix = data_loaded[i][2]
                    weights = weight_matrix[:, -1]
                else:
                    # Direct weights
                    weights = data_loaded[i][2]
            else:
                # Default to uniform weights
                weights = np.ones(PARAMS["n_j"])
            
            instances.append((times, machines, weights))
        
        print(f"Loaded {len(instances)} test instances")
        
    except:
        # Generate new instances
        print(f"Generating {PARAMS['n_instances']} new test instances")
        instances = []
        
        for _ in range(PARAMS["n_instances"]):
            if PARAMS["weighted"]:
                # Variable weights
                instance = weighted_instance_gen(
                    n_j=PARAMS["n_j"], 
                    n_m=PARAMS["n_m"], 
                    low=configs.low, 
                    high=configs.high,
                    weight_low=configs.weight_low, 
                    weight_high=configs.weight_high
                )
            else:
                # Uniform weights
                times, machines = weighted_instance_gen(
                    n_j=PARAMS["n_j"],
                    n_m=PARAMS["n_m"],
                    low=configs.low,
                    high=configs.high
                )[:2]
                weights = np.ones(PARAMS["n_j"])
                instance = (times, machines, weights)
            
            instances.append(instance)
        
        # Save instances for future use
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        
        # Convert instances to the right format for saving
        save_data = []
        for times, machines, weights in instances:
            weight_matrix = np.zeros((PARAMS["n_j"], PARAMS["n_m"]), dtype=np.int32)
            weight_matrix[:, -1] = weights
            save_data.append(np.array([times, machines, weight_matrix]))
        
        np.save(test_file_path, np.array(save_data))
        print(f"Saved test instances to {test_file_path}")
    
    # Test all methods on all instances
    results = {
        'L2D': [],
        'SPT': [],
        'WSPT': [],
        'SRPT': []
    }
    
    # Main testing loop
    for i, instance in enumerate(instances):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Testing instance {i+1}/{len(instances)}")
        
        # Test each method
        _, l2d_ws = test_l2d(env, instance, model, device)
        results['L2D'].append(l2d_ws)
        
        _, spt_ws = test_spt(env, instance)
        results['SPT'].append(spt_ws)
        
        _, wspt_ws = test_wspt(env, instance)
        results['WSPT'].append(wspt_ws)
        
        _, srpt_ws = test_srpt(env, instance)
        results['SRPT'].append(srpt_ws)
    
    # Calculate statistics
    avg_results = {method: np.mean(values) for method, values in results.items()}
    
    # Calculate win counts
    wins = {method: 0 for method in results}
    for i in range(len(instances)):
        best_method = min(results.keys(), key=lambda m: results[m][i])
        wins[best_method] += 1
    
    # Print results
    print("\nAverage Weighted Sums:")
    for method, avg in avg_results.items():
        print(f"{method}: {avg:.2f}")
    
    print("\nWin Counts:")
    for method, count in wins.items():
        print(f"{method}: {count}/{len(instances)} ({count/len(instances)*100:.2f}%)")
    
    # Calculate improvements over WSPT
    print("\nImprovements over WSPT:")
    for method in ['L2D', 'SPT', 'SRPT']:
        improvement = ((avg_results['WSPT'] - avg_results[method]) / avg_results['WSPT']) * 100
        print(f"{method}: {improvement:.2f}%")
    
    # Generate performance profile
    profile_filename = f"perf_profile_{PARAMS['n_j']}x{PARAMS['n_m']}_{len(instances)}inst_{'weighted' if PARAMS['weighted'] else 'uniform'}.png"
    profile_path = os.path.join(PARAMS["output_dir"], profile_filename)
    
    ratios = plot_performance_profile(results, profile_path)
    print(f"Performance profile saved to {profile_path}")
    
    # Save detailed results
    results_filename = f"results_{PARAMS['n_j']}x{PARAMS['n_m']}_{len(instances)}inst_{'weighted' if PARAMS['weighted'] else 'uniform'}.npz"
    results_path = os.path.join(PARAMS["output_dir"], results_filename)
    
    np.savez(results_path, **results)
    print(f"Detailed results saved to {results_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(
        results.keys(), 
        avg_results, 
        wins, 
        len(instances),
        PARAMS
    )
    
    if latex_table:
        latex_filename = f"table_{PARAMS['n_j']}x{PARAMS['n_m']}_{len(instances)}inst_{'weighted' if PARAMS['weighted'] else 'uniform'}.tex"
        latex_path = os.path.join(PARAMS["output_dir"], latex_filename)
        
        with open(latex_path, "w") as f:
            f.write(latex_table)
        
        print(f"LaTeX table saved to {latex_path}")

if __name__ == "__main__":
    main()