import numpy as np
from uniform_instance_gen import weighted_instance_gen

j = 200
m = 50
l = 1
h = 99
batch_size = 100
seed = 200
weight_low = 1
weight_high = 10

np.random.seed(seed)

instances = []
for _ in range(batch_size):
    times, machines, weights = weighted_instance_gen(n_j=j, n_m=m, low=l, high=h, 
                                                weight_low=weight_low, weight_high=weight_high)
    # Create a matrix of zeros with the same shape as times
    weight_matrix = np.zeros((j, m), dtype=int)

    # Place the weights in the last column
    weight_matrix[:, -1] = weights

    instance = np.array([times, machines, weight_matrix])
    instances.append(instance)

data = np.array(instances)

print(f"Generated {len(data)} instances")
print(f"Data shape: {data.shape}")
np.save(f'weightedData{j}_{m}_Seed{seed}.npy', data)