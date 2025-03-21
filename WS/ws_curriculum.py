import numpy as np

def generate_smart_curriculum_weights(n_j, episode, max_episodes, curriculum_type='multi_stage', weight_min=1, weight_max=10, seed=None):
    """
    Generate weights following a sophisticated curriculum that gradually increases difficulty.
    
    Args:
        n_j: Number of jobs
        episode: Current episode number
        max_episodes: Total number of episodes for training
        curriculum_type: Curriculum strategy to use
            - 'multi_stage': Gradually increases complexity through multiple distinct stages
            - 'weighted_blend': Smoothly transitions between patterns with weighted probability
            - 'focused_exploration': Focuses exploration on patterns where L2D struggles vs WSPT
            - 'machine_aware': Considers machine conflicts in weight distribution
        weight_min: Minimum weight value
        weight_max: Maximum weight value
        seed: Random seed for reproducibility
        
    Returns:
        np.array: Weights for each job following the curriculum
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate normalized progress (0 to 1)
    progress = min(1.0, episode / max_episodes)
    
    # Initialize weights array
    weights = np.ones(n_j, dtype=np.float32) * weight_min
    
    if curriculum_type == 'multi_stage':
        # Multi-stage curriculum with distinct phases
        if progress < 0.2:
            # Stage 1: Uniform weights (all equal)
            # This is the simplest case - already initialized with weight_min
            pass
            
        elif progress < 0.4:
            # Stage 2: Binary weights (high/low)
            # 30% of jobs have high weight, rest have low weight
            num_high_weight = max(1, int(n_j * 0.3))
            high_indices = np.random.choice(n_j, num_high_weight, replace=False)
            weights[high_indices] = weight_max
            
        elif progress < 0.6:
            # Stage 3: Clustered weights (2-3 classes)
            # Create 2-3 distinct weight clusters
            num_clusters = np.random.randint(2, 4)
            cluster_weights = np.linspace(weight_min, weight_max, num_clusters)
            
            # Assign each job to a cluster
            for j in range(n_j):
                cluster = np.random.randint(0, num_clusters)
                weights[j] = cluster_weights[cluster]
                
        elif progress < 0.8:
            # Stage 4: Power-law distributed weights
            # Generate weights following a power law distribution (Zipf-like)
            alpha = 1.5  # Parameter controlling skewness
            ranks = np.arange(1, n_j + 1)
            unnormalized = 1.0 / (ranks ** alpha)
            normalized = unnormalized / unnormalized.sum()
            
            # Scale to weight range and randomly assign
            weight_range = weight_max - weight_min
            scaled_weights = weight_min + normalized * weight_range * n_j
            np.random.shuffle(scaled_weights)
            weights = scaled_weights
            
        else:
            # Stage 5: Fully random weights
            weights = np.random.randint(weight_min, weight_max + 1, size=n_j).astype(np.float32)
    
    elif curriculum_type == 'weighted_blend':
        # Smoothly blend between different patterns based on progress
        
        # Define weight patterns
        patterns = {
            'uniform': np.ones(n_j) * weight_min,
            'binary': np.ones(n_j) * weight_min,
            'clustered': np.zeros(n_j),
            'random': np.random.randint(weight_min, weight_max + 1, size=n_j)
        }
        
        # Set binary pattern (30% high, 70% low)
        high_indices = np.random.choice(n_j, max(1, int(n_j * 0.3)), replace=False)
        patterns['binary'][high_indices] = weight_max
        
        # Set clustered pattern (2-3 clusters)
        num_clusters = np.random.randint(2, 4)
        cluster_weights = np.linspace(weight_min, weight_max, num_clusters)
        for j in range(n_j):
            cluster = np.random.randint(0, num_clusters)
            patterns['clustered'][j] = cluster_weights[cluster]
        
        # Calculate blend weights based on progress
        if progress < 0.33:
            # Blend uniform and binary
            alpha = progress / 0.33
            weights = (1 - alpha) * patterns['uniform'] + alpha * patterns['binary']
        elif progress < 0.67:
            # Blend binary and clustered
            alpha = (progress - 0.33) / 0.34
            weights = (1 - alpha) * patterns['binary'] + alpha * patterns['clustered']
        else:
            # Blend clustered and random
            alpha = (progress - 0.67) / 0.33
            weights = (1 - alpha) * patterns['clustered'] + alpha * patterns['random']
    
    elif curriculum_type == 'focused_exploration':
        # This strategy focuses on patterns where L2D typically struggles vs WSPT
        # Based on empirical observations, WSPT often outperforms on high variance weights
        
        # Determine exploration vs exploitation based on progress
        explore_prob = max(0.1, 1.0 - progress)  # Gradually reduce exploration
        
        if np.random.random() < explore_prob:
            # Exploration phase: try different weight patterns
            pattern_type = np.random.choice(['uniform', 'binary', 'clustered', 'random'])
            
            if pattern_type == 'uniform':
                pass  # Already initialized as uniform
            elif pattern_type == 'binary':
                num_high_weight = max(1, int(n_j * 0.3))
                high_indices = np.random.choice(n_j, num_high_weight, replace=False)
                weights[high_indices] = weight_max
            elif pattern_type == 'clustered':
                num_clusters = np.random.randint(2, 4)
                cluster_weights = np.linspace(weight_min, weight_max, num_clusters)
                for j in range(n_j):
                    cluster = np.random.randint(0, num_clusters)
                    weights[j] = cluster_weights[cluster]
            else:  # random
                weights = np.random.randint(weight_min, weight_max + 1, size=n_j).astype(np.float32)
        else:
            # Exploitation phase: focus on challenging cases for L2D
            # Generate weights with high variance (challenging for L2D vs WSPT)
            
            # Create high-variance weight distribution
            # Use a mixture of high and low weights with few intermediate values
            num_high = max(1, int(n_j * 0.4))
            high_indices = np.random.choice(n_j, num_high, replace=False)
            weights = np.ones(n_j) * weight_min
            weights[high_indices] = weight_max
            
            # Add small random noise to some weights
            noise_indices = np.random.choice(n_j, max(1, int(n_j * 0.2)), replace=False)
            weights[noise_indices] += np.random.randint(1, (weight_max - weight_min) // 2, size=len(noise_indices))
    
    elif curriculum_type == 'machine_aware':
        # This strategy considers machine conflicts when generating weights
        # For this to work, we need machine assignment information, which would need to be passed
        # You would need to modify the function signature to include machine assignments
        
        # As a fallback, we'll use a simplified version that simulates machine awareness
        # In a real implementation, you would use actual machine conflict information
        
        # Simulate increasing machine conflicts with training progress
        conflict_level = progress
        
        # Start with uniform weights
        if conflict_level < 0.3:
            # Low conflict: uniform weights work well
            pass  # Already initialized as uniform
        elif conflict_level < 0.6:
            # Medium conflict: use clustered weights
            num_clusters = np.random.randint(2, 4)
            cluster_weights = np.linspace(weight_min, weight_max, num_clusters)
            for j in range(n_j):
                cluster = np.random.randint(0, num_clusters)
                weights[j] = cluster_weights[cluster]
        else:
            # High conflict: increase weight variance
            variance_level = weight_min + (weight_max - weight_min) * conflict_level
            weights = np.random.normal(
                loc=(weight_min + weight_max) / 2,
                scale=variance_level,
                size=n_j
            )
            # Clip to ensure weights stay within bounds
            weights = np.clip(weights, weight_min, weight_max).astype(np.float32)
    
    elif curriculum_type == 'gini_controlled':
        # This strategy directly controls the Gini coefficient of the weight distribution
        # Start with low Gini (L2D advantage) and gradually increase to high Gini (WSPT advantage)
        
        # Calculate target Gini coefficient based on progress
        # Starting from 0.05 (near uniform) to 0.4 (highly skewed)
        target_gini = 0.05 + progress * 0.35
        
        # Generate weights with the target Gini coefficient
        # Method: Use a Pareto distribution and calibrate its parameter to achieve target Gini
        
        if target_gini < 0.10:
            # For very low Gini, use nearly uniform weights with small random variations
            weights = np.ones(n_j) * ((weight_min + weight_max) / 2)
            # Add small random noise
            noise_scale = target_gini * 10  # Scale noise based on target Gini
            weights += np.random.normal(0, noise_scale, size=n_j)
            # Clip to ensure weights stay within bounds
            weights = np.clip(weights, weight_min, weight_max)
        
        else:
            # For higher Gini values, use a Pareto distribution
            # Trial and error to find alpha parameter that approximates target Gini
            # Lower alpha = higher inequality = higher Gini
            
            # Map target Gini to Pareto alpha parameter (approximate relationship)
            # For Pareto, Gini = 1/(2*alpha - 1) for alpha > 0.5
            # So alpha = (1 + 1/(2*Gini))/2
            alpha = (1 + 1/(2*target_gini))/2 if target_gini > 0 else 10
            
            # Generate Pareto distribution
            u = np.random.uniform(0, 1, size=n_j)
            x = (1 / (1 - u)) ** (1 / alpha)  # Pareto with minimum value 1
            
            # Scale to desired weight range
            weights = weight_min + (x - min(x)) * (weight_max - weight_min) / (max(x) - min(x))
            
            # Calculate the actual Gini coefficient of the generated weights
            sorted_weights = np.sort(weights)
            cumsum_weights = np.cumsum(sorted_weights)
            actual_gini = 1 - 2 * np.sum((cumsum_weights - sorted_weights/2) / cumsum_weights[-1]) / len(weights)
            
            # If we're far off from target, retry with adjusted alpha
            # (In practice, we might want a more sophisticated approach here)
            attempt = 0
            while abs(actual_gini - target_gini) > 0.05 and attempt < 3:
                # Adjust alpha based on whether actual Gini is too high or low
                if actual_gini > target_gini:
                    alpha *= 1.2  # Increase alpha to decrease inequality
                else:
                    alpha /= 1.2  # Decrease alpha to increase inequality
                    
                # Regenerate weights with new alpha
                u = np.random.uniform(0, 1, size=n_j)
                x = (1 / (1 - u)) ** (1 / alpha)
                weights = weight_min + (x - min(x)) * (weight_max - weight_min) / (max(x) - min(x))
                
                # Recalculate Gini
                sorted_weights = np.sort(weights)
                cumsum_weights = np.cumsum(sorted_weights)
                actual_gini = 1 - 2 * np.sum((cumsum_weights - sorted_weights/2) / cumsum_weights[-1]) / len(weights)
                
                attempt += 1
        
        # Round weights to integers if needed (optional)
        weights = np.round(weights).astype(np.float32)
    
    else:
        # Fallback to simple random weights if unknown curriculum type
        weights = np.random.randint(weight_min, weight_max + 1, size=n_j).astype(np.float32)
    
    # Format as column vector
    #weight_matrix = weights.reshape(-1, 1)
    
    return weights