import numpy as np

def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]

def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines

def weighted_instance_gen(n_j, n_m, low, high, weight_low=1, weight_high=10):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    
    # Generate random weights for each job
    # Handle the special case of uniform weights (when low == high)
    if weight_low == weight_high:
        weights = np.ones(n_j, dtype=np.int32) * weight_low
    else:
        weights = np.random.randint(low=weight_low, high=weight_high, size=n_j)
    
    return times, machines, weights

def override(fn):
    """
    override decorator
    """
    return fn