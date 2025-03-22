import numpy as np
from Params import configs

def build_features(env, features_set=None):
    """
    Build state features based on enabled feature set.
    Returns a concatenated feature vector for all operations.
    """
    if features_set is None:
        features_set = ['LBs', 'finished_mark', 'normalized_weights']
    
    # Create a dictionary to track which features are enabled
    available_features = {
        'LBs': False,                      # Lower bounds on operation end times
        'weighted_LBs': False,             # Lower bounds multiplied by job weights
        'finished_mark': False,            # Binary indicators of completed operations
        'weighted_priorities': False,      # WSPT ratio (weight/processing time)
        'normalized_weights': False,       # Direct job weights (normalized)
        'remaining_weighted_work': False,  # Weight × sum of remaining processing times
        'time_elapsed': False,             # Current timestep in scheduling
        'machine_contention': False,       # Number of operations needing each machine
        'weighted_priorities_normalized': False,  # Normalized WSPT ratio
        'last_op_weighted': False,         # Feature for normalized weights applied only to last operations
        'critical_path_contribution': False # Contribution to critical paths of weighted jobs
    }
    
    # Enable selected features
    for feature in features_set:
        if feature in available_features:
            available_features[feature] = True
    
    features = []
    
    # Get maximum weight for normalization
    max_weight = np.max(env.weights)
    
    # Lower bounds on operation end times
    if available_features['LBs']:
        features.append(env.LBs.reshape(-1, 1)/configs.et_normalize_coef)
        
    # Binary indicators of completed operations
    if available_features['finished_mark']:
        features.append(env.finished_mark.reshape(-1, 1))
        
    # Weighted lower bounds (LBs multiplied by job weights)
    if available_features['weighted_LBs']:
        # Create weighted LBs matrix
        weighted_LBs = np.zeros_like(env.LBs, dtype=np.single)
        for j in range(env.number_of_jobs):
            # Apply job's weight to all of its operations' LBs
            weighted_LBs[j, :] = env.weights[j] * env.LBs[j, :]
        
        # Add normalized weighted LBs as feature
        # Use higher normalization coefficient since values are larger
        norm_factor = configs.et_normalize_coef * np.max(env.weights) if np.max(env.weights) > 1 else configs.et_normalize_coef
        features.append(weighted_LBs.reshape(-1, 1)/norm_factor)
        
    if available_features['weighted_priorities_normalized']:
        # Normalize WSPT ratio by maximum value
        norm_wspt = env.weighted_priorities.copy()
        if np.max(norm_wspt) > 0:
            norm_wspt = norm_wspt / np.max(norm_wspt)
        features.append(norm_wspt.reshape(-1, 1))
    
    # WSPT ratio (weight/processing time)
    if available_features['weighted_priorities']:
        features.append(env.weighted_priorities.reshape(-1, 1))
    
    # Direct job weights (normalized)
    if available_features['normalized_weights']:
        norm_weights = env.weights / np.max(env.weights)
        features.append(np.repeat(norm_weights, env.number_of_machines).reshape(-1, 1))
        
    # Feature for normalized weights applied only to last operations
    if available_features['last_op_weighted']:
        last_op_weighted = np.zeros_like(env.LBs, dtype=np.single)
        for j in range(env.number_of_jobs):
            # Only apply weight to the last operation of each job
            last_op_weighted[j, env.number_of_machines-1] = env.weights[j] / np.max(env.weights)
        features.append(last_op_weighted.reshape(-1, 1))
    
    # Weight × sum of remaining processing times
    if available_features['remaining_weighted_work']:
        remaining_work = calculate_remaining_work(env)
        # Normalize by maximum remaining work
        if np.max(remaining_work) > 0:
            remaining_work = remaining_work / np.max(remaining_work)
        features.append(np.repeat(remaining_work, env.number_of_machines).reshape(-1, 1))
    
    # Current timestep in scheduling
    if available_features['time_elapsed']:
        # Use current maximum end time as a proxy for elapsed time
        curr_time = np.max(env.temp1) if np.max(env.temp1) > 0 else 0
        time_feature = np.ones((env.number_of_tasks, 1), dtype=np.float32) * (curr_time / configs.high)
        features.append(time_feature)
    
    # Number of operations needing each machine
    if available_features['machine_contention']:
        machine_contention = calculate_machine_contention(env)
        # Create a feature that maps each operation to its machine's contention
        machine_feature = np.zeros((env.number_of_jobs, env.number_of_machines), dtype=np.single)
        for j in range(env.number_of_jobs):
            for m in range(env.number_of_machines):
                machine_id = env.m[j, m] - 1
                machine_feature[j, m] = machine_contention[machine_id]
        # Normalize by maximum contention
        if np.max(machine_feature) > 0:
            machine_feature = machine_feature / np.max(machine_feature)
        features.append(machine_feature.reshape(-1, 1))
    
    # Calculate each operation's contribution to critical paths of weighted jobs
    if available_features['critical_path_contribution']:
        contributions = np.zeros((env.number_of_jobs, env.number_of_machines), dtype=np.single)

        for j in range(env.number_of_jobs):
            # Skip completed jobs
            if np.all(env.finished_mark[j, :] == 1):
                continue
                
            # Find remaining operations for this job
            remaining_ops = np.where(env.finished_mark[j, :] == 0)[0]
            if len(remaining_ops) == 0:
                continue
                
            # Weight the contribution by job weight
            weight_factor = env.weights[j] / np.max(env.weights)
            
            # Mark all remaining operations with their weighted contribution
            for op in remaining_ops:
                # Operations earlier in the job's sequence have higher contribution
                position_factor = 1.0 - (op / env.number_of_machines)
                contributions[j, op] = weight_factor * position_factor
    
        features.append(contributions.reshape(-1, 1))
        
    return np.concatenate(features, axis=1)

def calculate_remaining_work(env):
    """
    Calculate remaining processing time for each job weighted by job importance.
    Higher values indicate more critical jobs to complete.
    """
    remaining_work = np.zeros(env.number_of_jobs, dtype=np.single)
    for j in range(env.number_of_jobs):
        total_remaining = 0
        for m in range(env.number_of_machines):
            if env.finished_mark[j, m] == 0:  # If operation not completed
                total_remaining += env.dur[j, m]
        remaining_work[j] = env.weights[j] * total_remaining
    return remaining_work

def calculate_machine_contention(env):
    """
    Calculate how many unscheduled operations need each machine.
    Higher values indicate more contested machines.
    """
    machine_contention = np.zeros(env.number_of_machines, dtype=np.single)
    for j in range(env.number_of_jobs):
        for m in range(env.number_of_machines):
            if env.finished_mark[j, m] == 0:  # If operation not completed
                machine_id = env.m[j, m] - 1  # Machine required for this operation
                machine_contention[machine_id] += 1
    return machine_contention