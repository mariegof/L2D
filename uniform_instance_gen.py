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
    weights = np.random.randint(low=weight_low, high=weight_high, size=n_j)
    
    return times, machines, weights

def generate_curriculum_weights(n_j, episode, max_episodes, weight_min=1, weight_max=10):
    """Generate weights based on curriculum learning approach."""
    # Calculate progress through training (0 to 1)
    progress = min(1.0, episode / (max_episodes * 0.5))  # Complete curriculum by 50% of training
    
    # Calculate current weight range based on progress
    current_max = weight_min + progress * (weight_max - weight_min)
    
    if progress < 0.3:
        # Phase 1: Almost uniform weights with tiny variance
        weights = np.ones(n_j) * weight_min
        variance = progress * 1.0  # Small variance
        weights += np.random.uniform(0, variance, size=n_j)
    
    elif progress < 0.6:
        # Phase 2: Bimodal distribution (half high, half low)
        weights = np.ones(n_j) * weight_min
        high_indices = np.random.choice(n_j, n_j // 2, replace=False)
        weights[high_indices] = current_max
    
    else:
        # Phase 3: Full random distribution
        weights = np.random.uniform(weight_min, current_max, size=n_j)
    
    return weights

def override(fn):
    """
    override decorator
    """
    return fn


