# feature_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os

def analyze_feature_importance(model, env, feature_names=None):
    """Analyze first layer weights to understand feature importance"""
    # Extract weights from the first layer of the feature extraction network
    first_layer = model.feature_extract.mlps[0].linears[0]
    weights = first_layer.weight.data.cpu().numpy()
    
    # Calculate importance as average magnitude of weights for each input feature
    importance = np.mean(np.abs(weights), axis=0)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # Print and visualize results
    for i, (name, score) in enumerate(zip(feature_names, importance)):
        print(f"{name}: {score:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(importance)), importance, color='blue')
    plt.xticks(range(len(importance)), feature_names, rotation=45, ha="right")
    plt.title("Feature Importance from First Layer Weights")
    plt.ylabel("Average Absolute Weight")
    plt.tight_layout()
    
    # Save figure
    os.makedirs("analysis", exist_ok=True)
    plt.savefig("analysis/feature_importance_weights.png")
    
    return importance

def feature_occlusion_test(model, env, instance, g_pool_step, feature_names=None, device="cpu", verbose=True):
    """Test model performance with each feature zeroed out"""
    # Set up basic tracking
    results = {}
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    
    # Define feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(fea.shape[1])]
    
    # Get baseline prediction
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
    
    with torch.no_grad():
        baseline_pi, _ = model(
            x=fea_tensor,
            graph_pool=g_pool_step,
            padded_nei=None,
            adj=adj_tensor,
            candidate=candidate_tensor.unsqueeze(0),
            mask=mask_tensor.unsqueeze(0)
        )
        baseline_probs = torch.softmax(baseline_pi, dim=1).cpu().numpy()
        baseline_action = np.argmax(baseline_probs)
    
    # Test each feature
    feature_impacts = []
    for i in range(fea.shape[1]):
        # Create modified input with this feature zeroed
        modified_fea = np.copy(fea)
        modified_fea[:, i] = 0
        
        # Convert to tensor
        mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
        
        # Get prediction with feature zeroed
        with torch.no_grad():
            mod_pi, _ = model(
                x=mod_fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            mod_probs = torch.softmax(mod_pi, dim=1).cpu().numpy()
            mod_action = np.argmax(mod_probs)
        
        # Calculate impact
        prob_diff = np.mean(np.abs(baseline_probs - mod_probs))
        action_changed = baseline_action != mod_action
        
        feature_impacts.append({
            'feature': feature_names[i],
            'probability_change': prob_diff,
            'action_changed': action_changed
        })
        
       # Only print if verbose is True
        if verbose:
            print(f"Feature {feature_names[i]}:")
            print(f"  Probability change: {prob_diff:.4f}")
            print(f"  Action changed: {action_changed}")
    
    # Sort by impact
    feature_impacts.sort(key=lambda x: x['probability_change'], reverse=True)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    features = [impact['feature'] for impact in feature_impacts]
    changes = [impact['probability_change'] for impact in feature_impacts]
    colors = ['red' if impact['action_changed'] else 'blue' for impact in feature_impacts]
    
    bars = plt.bar(range(len(features)), changes, color=colors)
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.title("Feature Impact - Occlusion Test")
    plt.ylabel("Probability Distribution Change")
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Changed action'),
        Patch(facecolor='blue', label='Same action')
    ]
    plt.legend(handles=legend_elements)
    
    # Save figure
    plt.savefig("analysis/feature_occlusion_test.png")
    
    return feature_impacts

def plot_feature_importance(feature_importance, feature_names, output_dir="analysis"):
    """Create a bar chart showing the importance of each feature based on model weights"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_importance)))
    
    # Sort features by importance
    indices = np.argsort(feature_importance)
    sorted_importance = feature_importance[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    bars = plt.barh(range(len(sorted_importance)), sorted_importance, color=colors)
    plt.yticks(range(len(sorted_importance)), sorted_names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance from Model Weights')
    plt.tight_layout()
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center')
    
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()
    
    return os.path.join(output_dir, "feature_importance.png")

def plot_feature_impacts(feature_impacts, output_dir="analysis"):
    """Create a visualization of feature impacts from occlusion testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    features = [impact['feature'] for impact in feature_impacts]
    prob_changes = [impact['probability_change'] for impact in feature_impacts]
    action_changed = [impact['action_changed'] for impact in feature_impacts]
    
    # Sort by impact
    sorted_indices = np.argsort(prob_changes)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_changes = [prob_changes[i] for i in sorted_indices]
    sorted_action_changed = [action_changed[i] for i in sorted_indices]
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    colors = ['#ff6b6b' if changed else '#4dabf7' for changed in sorted_action_changed]
    
    bars = plt.barh(range(len(sorted_features)), sorted_changes, color=colors)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Impact on Model Decisions')
    plt.title('Feature Impact Analysis (Occlusion Test)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='Changed decision'),
        Patch(facecolor='#4dabf7', label='Same decision')
    ]
    plt.legend(handles=legend_elements)
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_impact_analysis.png"))
    plt.close()
    
    return os.path.join(output_dir, "feature_impact_analysis.png")

def create_feature_summary_report(feature_importance, feature_impacts, feature_names, output_dir="analysis"):
    """Create a text report summarizing feature analysis findings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort features by importance
    importance_order = np.argsort(feature_importance)[::-1]
    
    # Sort impacts by probability change
    impacts_sorted = sorted(feature_impacts, key=lambda x: x['probability_change'], reverse=True)
    
    with open(os.path.join(output_dir, "feature_analysis_summary.txt"), "w") as f:
        f.write("FEATURE ANALYSIS SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write("1. FEATURE IMPORTANCE (FROM MODEL WEIGHTS)\n")
        f.write("------------------------------------------\n")
        for i in importance_order:
            f.write(f"{feature_names[i]}: {feature_importance[i]:.4f}\n")
        
        f.write("\n2. FEATURE IMPACT (FROM OCCLUSION TESTING)\n")
        f.write("------------------------------------------\n")
        for impact in impacts_sorted:
            decision_effect = "CHANGES DECISIONS" if impact['action_changed'] else "preserves decisions"
            f.write(f"{impact['feature']}: {impact['probability_change']:.4f} ({decision_effect})\n")
        
        f.write("\n3. KEY OBSERVATIONS\n")
        f.write("------------------\n")
        
        # Identify most important feature from weights
        most_important = feature_names[importance_order[0]]
        f.write(f"- Most influential feature from model weights: {most_important}\n")
        
        # Identify feature with biggest decision impact
        most_impactful = impacts_sorted[0]['feature']
        f.write(f"- Feature with largest decision impact: {most_impactful}\n")
        
        # Identify discrepancies between importance and impact
        if most_important != most_impactful:
            f.write(f"- Note: Different features identified as important by different methods\n")
        
        # Count features that change decisions
        decision_changers = sum(1 for impact in feature_impacts if impact['action_changed'])
        f.write(f"- Number of features that can change decisions when removed: {decision_changers}\n")
    
    return os.path.join(output_dir, "feature_analysis_summary.txt")

def run_multi_instance_occlusion_analysis(model, env, instances, g_pool_step, feature_names=None, 
                                          device="cpu", num_instances=10, output_dir="analysis"):
    """Run occlusion tests across multiple instances for statistical robustness."""
    
    # Use subset of instances for efficiency if needed
    test_instances = instances[:num_instances]
    feature_impacts_all = []
    
    # For storing results
    feature_impact_stats = {feature: {'impacts': [], 'decision_changes': 0} 
                          for feature in feature_names}
    
    print(f"Running occlusion analysis on {len(test_instances)} instances...")
    for i, instance in enumerate(tqdm(test_instances)):
        # Run occlusion test on this instance
        impacts = feature_occlusion_test(model, env, instance, g_pool_step, 
                                       feature_names, device, verbose=False)
        
        # Store results
        for impact in impacts:
            feature = impact['feature']
            feature_impact_stats[feature]['impacts'].append(impact['probability_change'])
            if impact['action_changed']:
                feature_impact_stats[feature]['decision_changes'] += 1
    
    # Calculate statistics
    results = []
    for feature, stats in feature_impact_stats.items():
        impacts = stats['impacts']
        results.append({
            'feature': feature,
            'mean_impact': np.mean(impacts),
            'std_dev': np.std(impacts),
            'max_impact': np.max(impacts),
            'decision_change_rate': stats['decision_changes'] / len(test_instances),
            'confidence_95': 1.96 * np.std(impacts) / np.sqrt(len(impacts))
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_impact', ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "multi_instance_occlusion_results.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    x = np.arange(len(results_df))
    width = 0.35
    
    # Plot mean impacts with error bars
    plt.bar(x, results_df['mean_impact'], width, 
            yerr=results_df['confidence_95'],
            color=[plt.cm.RdYlBu(0.1 + 0.8 * rate) for rate in results_df['decision_change_rate']])
    
    plt.xlabel('Feature')
    plt.ylabel('Mean Probability Distribution Change')
    plt.title('Feature Impact Analysis Across Multiple Instances')
    plt.xticks(x, results_df['feature'], rotation=45, ha='right')
    plt.tight_layout()
    
    # Add color bar for decision change rate
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=plt.cm.RdYlBu, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Decision Change Rate', rotation=270, labelpad=15)
    
    plt.savefig(os.path.join(output_dir, "multi_instance_feature_impact.png"))
    plt.close()
    
    return results_df

def analyze_scheduling_stage_importance(model, env, instance, g_pool_step, feature_names=None, 
                                        device="cpu", output_dir="analysis"):
    """Analyze how feature importance changes during different stages of scheduling."""
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(env.number_of_tasks)]
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    
    # For storing results at each stage
    stages = ['early', 'middle', 'late']
    stage_boundaries = [0.3, 0.7]  # Completion percentage thresholds
    stage_results = {stage: [] for stage in stages}
    
    # Track completion percentage
    total_operations = env.number_of_jobs * env.number_of_machines
    completed_operations = 0
    current_stage = 'early'
    
    # Run through scheduling process
    while not env.done():
        # Check current stage
        completion_pct = completed_operations / total_operations
        if completion_pct > stage_boundaries[1]:
            current_stage = 'late'
        elif completion_pct > stage_boundaries[0]:
            current_stage = 'middle'
        
        # Convert to tensors
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        # Run feature importance analysis for this stage
        with torch.no_grad():
            # Baseline prediction
            baseline_pi, _ = model(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            
            # Test each feature
            for i, feature in enumerate(feature_names):
                modified_fea = np.copy(fea)
                modified_fea[:, i] = 0
                
                mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
                mod_pi, _ = model(
                    x=mod_fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                
                # Calculate impact
                prob_diff = torch.abs(mod_pi - baseline_pi).mean().item()
                stage_results[current_stage].append({'feature': feature, 'impact': prob_diff})
        
        # Get action and step environment
        pi = baseline_pi.cpu().detach().numpy()
        action_idx = np.argmax(pi.squeeze())
        action = candidate[action_idx]
        adj, fea, reward, done, candidate, mask = env.step(action)
        completed_operations += 1
    
    # Process results for each stage
    for stage in stages:
        if not stage_results[stage]:
            continue
            
        df = pd.DataFrame(stage_results[stage])
        # Calculate average impact per feature
        feature_avg_impact = df.groupby('feature')['impact'].mean().reset_index()
        feature_avg_impact = feature_avg_impact.sort_values('impact', ascending=False)
        
        # Create visualization for this stage
        plt.figure(figsize=(10, 6))
        plt.bar(feature_avg_impact['feature'], feature_avg_impact['impact'],
                color=plt.cm.viridis(np.linspace(0, 0.8, len(feature_avg_impact))))
        plt.title(f'Feature Importance - {stage.capitalize()} Scheduling Stage')
        plt.xlabel('Feature')
        plt.ylabel('Average Impact')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{stage}_stage.png"))
        plt.close()
    
    # Create composite visualization
    plt.figure(figsize=(12, 8))
    
    # Get common features across all stages with data
    all_features = set()
    for stage in stages:
        if stage_results[stage]:
            features = pd.DataFrame(stage_results[stage])['feature'].unique()
            all_features.update(features)
    
    all_features = sorted(list(all_features))
    x = np.arange(len(all_features))
    width = 0.25
    
    # Plot bars for each stage
    for i, stage in enumerate(stages):
        if not stage_results[stage]:
            continue
            
        df = pd.DataFrame(stage_results[stage])
        feature_impacts = df.groupby('feature')['impact'].mean()
        
        # Create impact array in the right order
        impacts = [feature_impacts.get(feature, 0) for feature in all_features]
        
        plt.bar(x + (i-1)*width, impacts, width, label=f'{stage.capitalize()} Stage')
    
    plt.xlabel('Feature')
    plt.ylabel('Average Impact')
    plt.title('Feature Importance Across Scheduling Stages')
    plt.xticks(x, all_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_across_stages.png"))
    plt.close()
    
    return stage_results

def analyze_feature_interactions(model, env, instance, g_pool_step, feature_names=None, 
                                device="cpu", top_k=3, output_dir="analysis"):
    """Analyze pairwise feature interactions by removing feature pairs."""
    from itertools import combinations
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(env.number_of_tasks)]
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    
    # Get baseline prediction
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
    
    with torch.no_grad():
        baseline_pi, _ = model(
            x=fea_tensor,
            graph_pool=g_pool_step,
            padded_nei=None,
            adj=adj_tensor,
            candidate=candidate_tensor.unsqueeze(0),
            mask=mask_tensor.unsqueeze(0)
        )
        baseline_probs = torch.softmax(baseline_pi, dim=1).cpu().numpy()
        baseline_action = np.argmax(baseline_probs)
    
    # First get individual feature impacts
    individual_impacts = {}
    for i, feature in enumerate(feature_names):
        modified_fea = np.copy(fea)
        modified_fea[:, i] = 0
        
        mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
        with torch.no_grad():
            mod_pi, _ = model(
                x=mod_fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            mod_probs = torch.softmax(mod_pi, dim=1).cpu().numpy()
        
        prob_diff = np.mean(np.abs(baseline_probs - mod_probs))
        individual_impacts[i] = prob_diff
    
    # Get pairwise impacts
    pair_impacts = {}
    feature_pairs = list(combinations(range(len(feature_names)), 2))
    
    for i, j in feature_pairs:
        modified_fea = np.copy(fea)
        modified_fea[:, i] = 0
        modified_fea[:, j] = 0
        
        mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
        with torch.no_grad():
            mod_pi, _ = model(
                x=mod_fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            mod_probs = torch.softmax(mod_pi, dim=1).cpu().numpy()
        
        prob_diff = np.mean(np.abs(baseline_probs - mod_probs))
        expected_diff = individual_impacts[i] + individual_impacts[j]
        
        # Calculate interaction effect (difference from expected impact)
        interaction = prob_diff - expected_diff
        pair_impacts[(i, j)] = {
            'observed_impact': prob_diff,
            'expected_impact': expected_diff,
            'interaction': interaction
        }
    
    # Find top interactions (strongest synergies and redundancies)
    sorted_pairs = sorted(pair_impacts.items(), key=lambda x: abs(x[1]['interaction']), reverse=True)
    top_pairs = sorted_pairs[:top_k]
    
    # Prepare visualization data
    interaction_data = []
    for (i, j), data in top_pairs:
        feature_i = feature_names[i]
        feature_j = feature_names[j]
        interaction_data.append({
            'pair': f"{feature_i} + {feature_j}",
            'expected': data['expected_impact'],
            'observed': data['observed_impact'],
            'interaction': data['interaction'],
            'type': 'Synergy' if data['interaction'] > 0 else 'Redundancy'
        })
    
    # Create interaction visualization
    plt.figure(figsize=(12, 8))
    pairs = [d['pair'] for d in interaction_data]
    x = np.arange(len(pairs))
    width = 0.35
    
    # Plot expected and observed impacts
    plt.bar(x - width/2, [d['expected'] for d in interaction_data], width, label='Expected')
    plt.bar(x + width/2, [d['observed'] for d in interaction_data], width, label='Observed', alpha=0.7)
    
    # Add interaction values
    for i, d in enumerate(interaction_data):
        interaction = d['interaction']
        sign = '+' if interaction > 0 else ''
        plt.text(i, max(d['expected'], d['observed']) + 0.0005, 
                f"{sign}{interaction:.4f}", ha='center', fontweight='bold',
                color='green' if interaction > 0 else 'red')
    
    plt.xlabel('Feature Pairs')
    plt.ylabel('Impact on Model Prediction')
    plt.title('Feature Interaction Analysis')
    plt.xticks(x, pairs, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_interaction_analysis.png"))
    plt.close()
    
    # Create heatmap of interactions    
    # Prepare heatmap data
    heatmap_data = np.zeros((len(feature_names), len(feature_names)))
    for (i, j), data in pair_impacts.items():
        heatmap_data[i, j] = data['interaction']
        heatmap_data[j, i] = data['interaction']  # Mirror for symmetry
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm_r", 
                    xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Interaction Strength Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_interaction_heatmap.png"))
    plt.close()
    
    return pair_impacts, interaction_data

def run_statistical_significance_test(model, env, instances, g_pool_step, feature_names=None,
                                     device="cpu", n_samples=30, n_permutations=100, 
                                     output_dir="analysis"):
    """Perform statistical significance testing for feature importance."""
    import random
    from scipy import stats
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(env.number_of_tasks)]
    
    # Randomly sample instances
    test_instances = random.sample(instances, min(n_samples, len(instances)))
    
    # For each feature, collect real impacts
    feature_impacts = {feature: [] for feature in feature_names}
    
    # For each feature, collect permutation test results
    perm_impacts = {feature: [] for feature in feature_names}
    
    # Get real feature impacts
    print("Calculating real feature impacts...")
    for instance in tqdm(test_instances):
        adj, fea, candidate, mask = env.reset(instance)
        
        # Get baseline prediction
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        with torch.no_grad():
            baseline_pi, _ = model(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            baseline_probs = torch.softmax(baseline_pi, dim=1).cpu().numpy()
        
        # Calculate impact for each feature
        for i, feature in enumerate(feature_names):
            modified_fea = np.copy(fea)
            modified_fea[:, i] = 0
            
            mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
            with torch.no_grad():
                mod_pi, _ = model(
                    x=mod_fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                mod_probs = torch.softmax(mod_pi, dim=1).cpu().numpy()
            
            prob_diff = np.mean(np.abs(baseline_probs - mod_probs))
            feature_impacts[feature].append(prob_diff)
    
    # Calculate permutation test results
    print("Running permutation tests...")
    for instance in tqdm(test_instances[:5]):  # Use smaller sample for permutations
        adj, fea, candidate, mask = env.reset(instance)
        
        # Get baseline prediction
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        with torch.no_grad():
            baseline_pi, _ = model(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
            baseline_probs = torch.softmax(baseline_pi, dim=1).cpu().numpy()
        
        # For each feature
        for i, feature in enumerate(feature_names):
            # Run permutation tests
            for _ in range(n_permutations // 5):  # Adjust number of permutations per instance
                # Create random permutation of feature values
                modified_fea = np.copy(fea)
                # Instead of zeroing, shuffle the feature values
                # This preserves distribution but breaks relationship
                np.random.shuffle(modified_fea[:, i])
                
                mod_fea_tensor = torch.from_numpy(modified_fea).to(device)
                with torch.no_grad():
                    mod_pi, _ = model(
                        x=mod_fea_tensor,
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensor,
                        candidate=candidate_tensor.unsqueeze(0),
                        mask=mask_tensor.unsqueeze(0)
                    )
                    mod_probs = torch.softmax(mod_pi, dim=1).cpu().numpy()
                
                prob_diff = np.mean(np.abs(baseline_probs - mod_probs))
                perm_impacts[feature].append(prob_diff)
    
    # Calculate significance
    results = []
    for feature in feature_names:
        real_impacts = feature_impacts[feature]
        perm_impact = perm_impacts[feature]
        
        # Calculate p-value (proportion of permutation impacts >= real impact)
        mean_real_impact = np.mean(real_impacts)
        p_value = np.mean([1 if perm >= mean_real_impact else 0 for perm in perm_impact])
        
        # Calculate confidence intervals
        ci_low, ci_high = stats.norm.interval(
            0.95, 
            loc=np.mean(real_impacts), 
            scale=stats.sem(real_impacts)
        )
        
        results.append({
            'feature': feature,
            'mean_impact': mean_real_impact,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
    
    # Sort by significance and impact
    results = sorted(results, key=lambda x: (x['significant'], x['mean_impact']), reverse=True)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot mean impacts with confidence intervals and significance
    x = np.arange(len(results))
    colors = ['green' if r['significant'] else 'gray' for r in results]
    
    plt.bar(x, [r['mean_impact'] for r in results], color=colors, alpha=0.7)
    plt.errorbar(x, [r['mean_impact'] for r in results], 
                yerr=[[r['mean_impact']-r['ci_low'] for r in results], 
                      [r['ci_high']-r['mean_impact'] for r in results]],
                fmt='none', color='black', capsize=5)
    
    # Add significance markers
    for i, r in enumerate(results):
        marker = '*' if r['significant'] else ''
        plt.text(i, r['mean_impact'] + 0.0005, marker, ha='center', fontsize=20)
    
    plt.xlabel('Feature')
    plt.ylabel('Mean Impact')
    plt.title('Feature Importance with Statistical Significance (95% CI)')
    plt.xticks(x, [r['feature'] for r in results], rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='Not Significant')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_significance_test.png"))
    plt.close()
    
    return results

def run_comprehensive_feature_analysis(model, env, instances, g_pool_step, feature_names,
                                     device="cpu", output_dir="analysis/comprehensive"):
    """Run a comprehensive suite of feature analysis methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running comprehensive feature analysis...")
    
    # 1. First, do basic weight analysis
    print("\n1. Analyzing feature importance from model weights...")
    weight_importance = analyze_feature_importance(model, env, feature_names)
    plot_path = plot_feature_importance(weight_importance, feature_names, output_dir)
    print(f"Feature importance from weights saved to {plot_path}")
    
    # 2. Run multi-instance occlusion testing
    print("\n2. Running occlusion tests across multiple instances...")
    multi_instance_results = run_multi_instance_occlusion_analysis(
        model, env, instances, g_pool_step, feature_names, device, 
        num_instances=min(10, len(instances)), output_dir=output_dir
    )
    print(f"Multi-instance occlusion results saved to {output_dir}")
    
    # 3. Analyze importance at different scheduling stages
    print("\n3. Analyzing feature importance across scheduling stages...")
    stage_results = analyze_scheduling_stage_importance(
        model, env, instances[0], g_pool_step, feature_names, device, output_dir
    )
    print(f"Scheduling stage analysis saved to {output_dir}")
    
    # 4. Test feature interactions
    print("\n4. Analyzing feature interactions...")
    interaction_results = analyze_feature_interactions(
        model, env, instances[0], g_pool_step, feature_names, device, 
        top_k=5, output_dir=output_dir
    )
    print(f"Feature interaction analysis saved to {output_dir}")
    
    # 5. Run statistical significance tests
    print("\n5. Performing statistical significance testing...")
    significance_results = run_statistical_significance_test(
        model, env, instances, g_pool_step, feature_names, device, 
        n_samples=min(20, len(instances)), n_permutations=100, output_dir=output_dir
    )
    print(f"Statistical significance tests saved to {output_dir}")
    
    # Create comprehensive report
    create_comprehensive_report(
        weight_importance, multi_instance_results, stage_results, 
        interaction_results, significance_results, feature_names, output_dir
    )
    
    print(f"\nComprehensive feature analysis complete. Results saved to {output_dir}")
    return output_dir

def create_comprehensive_report(weight_importance, multi_instance_results, stage_results, 
                             interaction_results, significance_results, feature_names, output_dir):
    """Create a comprehensive text report of all feature analysis results."""
    with open(os.path.join(output_dir, "comprehensive_feature_analysis.txt"), "w") as f:
        f.write("COMPREHENSIVE FEATURE ANALYSIS REPORT\n")
        f.write("====================================\n\n")
        
        # 1. Weight-based importance
        f.write("1. FEATURE IMPORTANCE FROM MODEL WEIGHTS\n")
        f.write("---------------------------------------\n")
        sorted_indices = np.argsort(weight_importance)[::-1]
        for i in sorted_indices:
            f.write(f"{feature_names[i]}: {weight_importance[i]:.4f}\n")
        
        # 2. Multi-instance occlusion results
        f.write("\n2. MULTI-INSTANCE OCCLUSION TESTING\n")
        f.write("----------------------------------\n")
        for _, row in multi_instance_results.iterrows():
            f.write(f"{row['feature']}: mean={row['mean_impact']:.4f}, "
                   f"change rate={row['decision_change_rate']:.2f}, "
                   f"95% CI=[{row['mean_impact']-row['confidence_95']:.4f}, "
                   f"{row['mean_impact']+row['confidence_95']:.4f}]\n")
        
        # 3. Scheduling stage importance
        f.write("\n3. FEATURE IMPORTANCE ACROSS SCHEDULING STAGES\n")
        f.write("--------------------------------------------\n")
        stages = ['early', 'middle', 'late']
        for stage in stages:
            if not stage_results[stage]:
                continue
                
            f.write(f"\n{stage.upper()} STAGE:\n")
            df = pd.DataFrame(stage_results[stage])
            feature_avg = df.groupby('feature')['impact'].mean().sort_values(ascending=False)
            for feature, impact in feature_avg.items():
                f.write(f"  {feature}: {impact:.4f}\n")
        
        # 4. Feature interactions
        f.write("\n4. FEATURE INTERACTIONS\n")
        f.write("----------------------\n")
        f.write("Top feature interactions by strength:\n")
        for item in interaction_results[1]:
            interaction = item['interaction']
            effect_type = "synergistic" if interaction > 0 else "redundant"
            f.write(f"{item['pair']}: {interaction:.4f} ({effect_type})\n")
        
        # 5. Statistical significance
        f.write("\n5. STATISTICAL SIGNIFICANCE\n")
        f.write("--------------------------\n")
        f.write("Features sorted by statistical significance:\n")
        for result in significance_results:
            sig_marker = "**" if result['significant'] else ""
            f.write(f"{result['feature']}: impact={result['mean_impact']:.4f}, "
                   f"p-value={result['p_value']:.4f} {sig_marker}\n")
        
        # 6. Summary
        f.write("\n6. SUMMARY OF KEY FINDINGS\n")
        f.write("-------------------------\n")
        
        # Most important feature by each method
        weight_top = feature_names[np.argmax(weight_importance)]
        occlusion_top = multi_instance_results.iloc[0]['feature']
        
        # Most significant feature
        sig_features = [r for r in significance_results if r['significant']]
        sig_top = sig_features[0]['feature'] if sig_features else "None"
        
        f.write(f"- Most important feature by model weights: {weight_top}\n")
        f.write(f"- Most important feature by occlusion testing: {occlusion_top}\n")
        f.write(f"- Most statistically significant feature: {sig_top}\n")
        
        # Feature with strongest interactions
        strongest_interaction = interaction_results[1][0]
        f.write(f"- Strongest feature interaction: {strongest_interaction['pair']} "
               f"({strongest_interaction['interaction']:.4f})\n")
        
        # Count features that change decisions
        decision_changers = multi_instance_results[multi_instance_results['decision_change_rate'] > 0]
        f.write(f"- Number of features that change decisions: {len(decision_changers)}\n")