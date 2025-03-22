import os
import numpy as np
from src.utils.instance_gen import weighted_instance_gen

def generate_variable_weight_instances(n_j, n_m, batch_size=100, seed=200, low=1, high=99, weight_low=1, weight_high=10):
    """Generate JSSP instances with variable weights in range [weight_low, weight_high]"""
    np.random.seed(seed)
    
    instances = []
    for _ in range(batch_size):
        times, machines, weights = weighted_instance_gen(
            n_j=n_j, n_m=n_m, low=low, high=high,
            weight_low=weight_low, weight_high=weight_high
        )
        
        # Create a matrix of zeros with the same shape as times
        weight_matrix = np.zeros((n_j, n_m), dtype=np.int32)
        
        # Place the weights in the last column
        weight_matrix[:, -1] = weights
        
        instance = np.array([times, machines, weight_matrix])
        instances.append(instance)
    
    data = np.array(instances)
    return data

def main():
    # Define configurations
    configs = [
        {'n_j': 6, 'n_m': 6},
        {'n_j': 10, 'n_m': 10},
        {'n_j': 15, 'n_m': 15},
        {'n_j': 20, 'n_m': 20},
    ]
    
    save_dir = './data/instances/variable_weights'
    os.makedirs(save_dir, exist_ok=True)
    
    # For both validation and test sets
    for prefix, seed in [('validation', 200), ('test', 300)]:
        for config in configs:
            n_j = config['n_j']
            n_m = config['n_m']
            
            data = generate_variable_weight_instances(n_j=n_j, n_m=n_m, seed=seed)
            
            save_path = os.path.join(save_dir, f'{prefix}_data_{n_j}x{n_m}_seed{seed}.npy')
            np.save(save_path, data)
            
            print(f"Generated {len(data)} {prefix} instances with variable weights for {n_j}x{n_m}")
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()