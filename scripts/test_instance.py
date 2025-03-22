import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.JSSP_Env import SJSSP
from src.agents.PPO_jssp import PPO
from src.utils.testing import test_l2d, test_spt, test_wspt, test_srpt
from src.utils.visualize import compare_schedules, draw_disjunctive_graph, create_precedence_graph
from src.utils.instance_gen import weighted_instance_gen
from Params import configs

# DIRECT PARAMETERS - Edit these instead of using command line arguments
PARAMS = {
    "model_path": "models/l2d_weighted_6x6_best.pth",  # Path to the model
    "n_j": 6,                         # Number of jobs
    "n_m": 6,                         # Number of machines
    "seed": 42,                       # Random seed
    "weighted": True,                # Use weighted instance (True) or uniform weights (False)
    "output_dir": "./results",        # Output directory
}

def main():
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
    
    # Generate instance
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
        # Uniform weights (all 1)
        times, machines = weighted_instance_gen(
            n_j=PARAMS["n_j"],
            n_m=PARAMS["n_m"],
            low=configs.low,
            high=configs.high
        )[:2]
        weights = np.ones(PARAMS["n_j"])
        instance = (times, machines, weights)
    
    times, machines, weights = instance
    print(f"Generated instance with {PARAMS['n_j']} jobs, {PARAMS['n_m']} machines")
    print(f"Weights: {weights}")
    
    # Create output directory
    os.makedirs(PARAMS["output_dir"], exist_ok=True)
    
    # Test all methods
    print("Testing L2D...")
    l2d_ops, l2d_ws = test_l2d(env, instance, model, device)
    
    print("Testing SPT...")
    spt_ops, spt_ws = test_spt(env, instance)
    
    print("Testing WSPT...")
    wspt_ops, wspt_ws = test_wspt(env, instance)
    
    print("Testing SRPT...")
    srpt_ops, srpt_ws = test_srpt(env, instance)
    
    # Compare results
    print("\nResults:")
    print(f"L2D: {l2d_ws:.2f}")
    print(f"SPT: {spt_ws:.2f}")
    print(f"WSPT: {wspt_ws:.2f}")
    print(f"SRPT: {srpt_ws:.2f}")
    
    # Calculate best method
    methods = {"L2D": l2d_ws, "SPT": spt_ws, "WSPT": wspt_ws, "SRPT": srpt_ws}
    best_method = min(methods, key=methods.get)
    print(f"\nBest method: {best_method} with weighted sum {methods[best_method]:.2f}")
    
    # Visualize Gantt charts for each method
    gantt_filename = f"gantt_charts_{PARAMS['n_j']}x{PARAMS['n_m']}_{'weighted' if PARAMS['weighted'] else 'uniform'}.png"
    gantt_path = os.path.join(PARAMS["output_dir"], gantt_filename)
    
    fig, _ = compare_schedules(
        instance,
        [l2d_ops, spt_ops, wspt_ops, srpt_ops],
        [f"L2D ({l2d_ws:.2f})", f"SPT ({spt_ws:.2f})", 
         f"WSPT ({wspt_ws:.2f})", f"SRPT ({srpt_ws:.2f})"],
        [l2d_ws, spt_ws, wspt_ws, srpt_ws]  # Pass the pre-calculated weighted sums
    )
    
    # Save Gantt charts
    plt.savefig(gantt_path, dpi=300, bbox_inches='tight')
    print(f"Gantt charts saved to {gantt_path}")
    plt.close(fig)
    
    # Draw disjunctive graph (without solution)
    disjunctive_filename = f"disjunctive_graph_{PARAMS['n_j']}x{PARAMS['n_m']}_{'weighted' if PARAMS['weighted'] else 'uniform'}.png"
    disjunctive_path = os.path.join(PARAMS["output_dir"], disjunctive_filename)

    fig_raw = draw_disjunctive_graph(
        times, 
        machines, 
        title=f"Disjunctive Graph for {PARAMS['n_j']}×{PARAMS['n_m']} Instance"
    )
    plt.savefig(disjunctive_path, dpi=300, bbox_inches='tight')
    print(f"Disjunctive graph saved to {disjunctive_path}")
    plt.close()
    
    # Draw disjunctive graph with L2D solution
    disjunctive_soln_filename = f"disjunctive_with_solution_{PARAMS['n_j']}x{PARAMS['n_m']}_{'weighted' if PARAMS['weighted'] else 'uniform'}.png"
    disjunctive_soln_path = os.path.join(PARAMS["output_dir"], disjunctive_soln_filename)

    fig_soln = draw_disjunctive_graph(
        times, 
        machines, 
        operations_order=l2d_ops,
        title=f"Disjunctive Graph with L2D Solution (Weighted Sum: {l2d_ws:.2f})"
    )
    plt.savefig(disjunctive_soln_path, dpi=300, bbox_inches='tight')
    print(f"Disjunctive graph with solution saved to {disjunctive_soln_path}")
    plt.close()
    
    # Generate method comparison bar chart
    comparison_filename = f"method_comparison_{PARAMS['n_j']}x{PARAMS['n_m']}_{'weighted' if PARAMS['weighted'] else 'uniform'}.png"
    comparison_path = os.path.join(PARAMS["output_dir"], comparison_filename)
    
    plot_method_comparison(methods, comparison_path)

def plot_method_comparison(methods, output_path):
    """Create a bar chart comparing methods"""
    plt.figure(figsize=(10, 6))
    
    # Get values and method names
    method_names = list(methods.keys())
    values = list(methods.values())
    
    # Calculate improvements relative to WSPT
    wspt_value = methods["WSPT"]
    improvements = [(wspt_value - v) / wspt_value * 100 if v != wspt_value else 0 for v in values]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot weighted sums
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax1.bar(method_names, values, color=colors)
    
    # Highlight best method
    best_method = min(methods, key=methods.get)
    best_idx = method_names.index(best_method)
    bars[best_idx].set_color('#9b59b6')
    
    ax1.set_title('Weighted Sum Comparison', fontsize=14)
    ax1.set_ylabel('Weighted Sum (Lower is Better)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', fontsize=10)
    
    # Plot improvements vs WSPT
    improvements_to_plot = []
    names_to_plot = []
    
    for i, method in enumerate(method_names):
        if method != "WSPT":
            improvements_to_plot.append(improvements[i])
            names_to_plot.append(method)
    
    bars = ax2.bar(names_to_plot, improvements_to_plot, color=colors[:3])
    
    # Color code: positive (improvement) = green, negative = red
    for i, imp in enumerate(improvements_to_plot):
        bars[i].set_color('#2ecc71' if imp > 0 else '#e74c3c')
    
    ax2.set_title('Improvement Over WSPT (%)', fontsize=14)
    ax2.set_ylabel('Improvement Percentage', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add improvement labels
    for bar in bars:
        height = bar.get_height()
        label_pos = height + 0.5 if height > 0 else height - 2
        ax2.text(bar.get_x() + bar.get_width()/2., label_pos,
                f'{height:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Method comparison saved to {output_path}")

if __name__ == "__main__":
    main()