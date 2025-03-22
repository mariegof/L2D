import os
import numpy as np
from src.utils.instance_gen import uni_instance_gen, weighted_instance_gen

def main():
    # Define problem sizes
    problem_sizes = [
        (6, 6),    # Small
        (10, 10),  # Medium
        (15, 15),  # Medium-large
        (20, 20),  # Large
    ]
    
    # Define seeds for validation and test sets
    validation_seed = 200
    test_seed = 300
    
    # Number of instances per set
    batch_size = 100
    
    # Duration range
    low = 1
    high = 99
    
    # Weight range
    weight_low = 1
    weight_high = 10
    
    # Create directories if they don't exist
    uniform_dir = './data/instances/uniform_weights'
    weighted_dir = './data/instances/variable_weights'
    os.makedirs(uniform_dir, exist_ok=True)
    os.makedirs(weighted_dir, exist_ok=True)
    
    # Generate instances for each problem size
    for n_j, n_m in problem_sizes:
        print(f"Generating instances for problem size {n_j}x{n_m}...")
        
        # Generate uniform weight instances (weights=1)
        for prefix, seed in [('validation', validation_seed), ('test', test_seed)]:
            np.random.seed(seed)
            
            # Generate and save uniform weight instances
            uniform_instances = []
            for _ in range(batch_size):
                times, machines = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
                weights = np.ones(n_j, dtype=np.int32)  # All weights are 1
                
                # Create a matrix of zeros with the same shape as times
                weight_matrix = np.zeros((n_j, n_m), dtype=np.int32)
                
                # Place the weights in the last column
                weight_matrix[:, -1] = weights
                
                instance = np.array([times, machines, weight_matrix])
                uniform_instances.append(instance)
            
            # Save uniform instances
            uniform_path = os.path.join(uniform_dir, f'{prefix}_data_{n_j}x{n_m}_seed{seed}.npy')
            np.save(uniform_path, np.array(uniform_instances))
            print(f"  Saved {batch_size} uniform weight {prefix} instances to {uniform_path}")
            
            # Generate and save weighted instances
            weighted_instances = []
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
                weighted_instances.append(instance)
            
            # Save weighted instances
            weighted_path = os.path.join(weighted_dir, f'{prefix}_data_{n_j}x{n_m}_seed{seed}.npy')
            np.save(weighted_path, np.array(weighted_instances))
            print(f"  Saved {batch_size} variable weight {prefix} instances to {weighted_path}")
    
    print("All instance generation complete!")

if __name__ == "__main__":
    main()