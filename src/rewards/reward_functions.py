import numpy as np
from Params import configs

# Dictionary of available reward functions
REWARD_FUNCTIONS = {
    'default': 'weighted_sum_difference',
    'wspt_guided': 'wspt_guided_reward',
    'potential_based': 'potential_based_reward',
    'critical_path': 'critical_path_reward',
    'wspt': 'wspt_reward',
    'weighted_delay': 'weighted_delay_reward',
    'direct_weighted_sum': 'direct_weighted_sum_reward',
    'look_ahead': 'look_ahead_reward'
}

def weighted_sum_difference(env, action):
    """Default reward function: difference in weighted sum estimate."""
    current_weighted_sum = env._calculate_weighted_sum_estimate()
    reward = -(current_weighted_sum - env.previous_weighted_sum)
    
    # Add small positive reward when needed
    if reward == 0:
        reward = configs.rewardscale
        env.posRewards += reward
    
    return reward

def wspt_guided_reward(env, action):
    """WSPT-guided reward function."""
    # Store previous weighted sum for improvement calculation
    previous_weighted_sum = env.previous_weighted_sum
    current_weighted_sum = env._calculate_weighted_sum_estimate()
    
    # Basic improvement reward
    improvement = previous_weighted_sum - current_weighted_sum
    
    # Get job index and operation index
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    
    # Calculate WSPT ratio for the selected operation
    wspt_ratio = env.weights[job_idx] / env.dur[job_idx, op_idx]
    
    # Calculate WSPT ratios for all eligible operations
    eligible_ops = np.where(~env.mask)[0]
    wspt_ratios = []
    
    for op in eligible_ops:
        j_idx = op // env.number_of_machines
        m_idx = op % env.number_of_machines
        wspt_ratios.append(env.weights[j_idx] / env.dur[j_idx, m_idx])
    
    # WSPT reward component (normalized between -1 and 1)
    if len(wspt_ratios) > 0:
        max_wspt = max(wspt_ratios)
        min_wspt = min(wspt_ratios)
        wspt_range = max_wspt - min_wspt if max_wspt != min_wspt else 1.0
        normalized_wspt = (wspt_ratio - min_wspt) / wspt_range if wspt_range != 0 else 0.5
        wspt_reward = 2 * normalized_wspt - 1  # Scale to [-1, 1]
    else:
        wspt_reward = 0
    
    # Combined reward (adjust alpha to balance short vs long-term signals)
    alpha = 0.7  # Emphasis on WSPT principle vs actual weighted sum improvement
    reward = alpha * wspt_reward + (1-alpha) * improvement
    
    # Add small positive reward when needed
    if reward == 0:
        reward = configs.rewardscale
        env.posRewards += reward
        
    return reward

def potential_based_reward(env, action):
    """Job completion potential-based reward."""
    # Store state before taking action
    previous_weighted_sum = env.previous_weighted_sum
    current_weighted_sum = env._calculate_weighted_sum_estimate()
    
    # Get job info
    job_idx = action // env.number_of_machines
    
    # Calculate immediate weighted sum improvement
    improvement = previous_weighted_sum - current_weighted_sum
    
    # Calculate the "potential" of the current state
    potential = 0
    for j in range(env.number_of_jobs):
        remaining_ops = 0
        remaining_time = 0
        for m in range(env.number_of_machines):
            if env.finished_mark[j, m] == 0:  # If operation not completed
                remaining_ops += 1
                remaining_time += env.dur[j, m]
        
        # Jobs with higher weights and less remaining work have higher potential
        if remaining_ops > 0:
            potential -= (env.weights[j] * remaining_time / remaining_ops)
    
    # Store current potential for next step
    if not hasattr(env, 'current_potential'):
        env.current_potential = potential
    
    # Calculate potential change
    potential_reward = potential - env.current_potential
    env.current_potential = potential  # Update for next step
    
    # Final reward combines immediate improvement and potential change
    reward = improvement + 0.5 * potential_reward
    
    # Add small positive reward when needed
    if reward == 0:
        reward = configs.rewardscale
        env.posRewards += reward
    
    return reward

def critical_path_reward(env, action):
    """Critical path weight reward."""
    # Store previous weighted sum
    previous_weighted_sum = env.previous_weighted_sum
    current_weighted_sum = env._calculate_weighted_sum_estimate()
    
    # Get job info
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    weight = env.weights[job_idx]
    
    # Calculate immediate improvement
    improvement = previous_weighted_sum - current_weighted_sum
    
    # Check if this completes a job
    job_completion_bonus = 0
    if action in env.last_col:  # If this is the last operation of a job
        # Reward for completing high-weight jobs earlier
        completion_time = env.max_endTime + env.dur[job_idx, op_idx]
        job_completion_bonus = weight / max(1, completion_time)  # Avoid division by zero
    
    # Calculate weighted slack for operations
    slack = 0
    for j in range(env.number_of_jobs):
        # Skip completed jobs
        if np.all(env.finished_mark[j, :] == 1):
            continue
            
        # Calculate operations remaining for this job
        remaining_ops = np.sum(env.finished_mark[j, :] == 0)
                        
        # If job has pending operations, calculate its critical path contribution
        if remaining_ops > 0:
            job_weight = env.weights[j]
            slack += job_weight * (remaining_ops / env.number_of_machines)
    
    # Critical path reward - prioritize operations that reduce weighted critical path
    critical_path_reward = weight / max(1, slack) if slack > 0 else 0
    
    # Combined reward
    reward = improvement + 0.3 * job_completion_bonus + 0.3 * critical_path_reward
    
    # Add small positive reward when needed
    if reward == 0:
        reward = configs.rewardscale
        env.posRewards += reward
    
    return reward

def wspt_reward(env, action):
    """Simple WSPT-based reward: job_weight/job_processing_time"""
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    
    # Calculate the WSPT ratio
    weight = env.weights[job_idx]
    processing_time = env.dur[job_idx, op_idx]
    
    # WSPT ratio as reward
    wspt_ratio = weight / max(1, processing_time)  # Avoid division by zero
    
    # Scale the reward to be comparable to other rewards
    reward = wspt_ratio * 10  # Adjust scaling factor as needed
    
    return reward

def weighted_delay_reward(env, action):
    """Reward based on -sum(job_weight*(operation_completion_time - operation_LB))"""
    # Calculate the weighted delay for this operation
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    weight = env.weights[job_idx]
    
    # Get the current completion time and lower bound
    actual_completion_time = env.temp1[job_idx, op_idx]
    lower_bound = env.LBs[job_idx, op_idx] - env.dur[job_idx, op_idx]  # LB start time
    
    # Calculate weighted delay
    weighted_delay = weight * (actual_completion_time - lower_bound)
    
    # Return negative delay as reward (less delay = higher reward)
    reward = -weighted_delay
    
    return reward

def direct_weighted_sum_reward(env, action):
    """Direct reward based on -sum(job_weight*operation_completion_time)"""
    # Calculate the contribution to weighted sum for this operation
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    weight = env.weights[job_idx]
    
    # Get the completion time
    completion_time = env.temp1[job_idx, op_idx]
    
    # Calculate weighted completion time
    weighted_completion = weight * completion_time
    
    # Return negative weighted completion as reward (earlier = higher reward)
    reward = -weighted_completion
    
    # For the last operation in each job, give an additional reward
    if action in env.last_col:
        # This directly affects the objective function
        job_completion_time = completion_time
        reward *= 2  # Double the reward for completing jobs
    
    return reward

def look_ahead_reward(env, action):
    """Reward function that encourages strategic deviation from WSPT when beneficial"""
    # Basic weighted completion time component
    job_idx = action // env.number_of_machines
    op_idx = action % env.number_of_machines
    
    # Calculate immediate WSPT ratio
    wspt_ratio = env.weights[job_idx] / env.dur[job_idx, op_idx]
    
    # Calculate WSPT ratios for all eligible operations
    eligible_ops = np.where(~env.mask)[0]
    best_wspt_op = None
    best_wspt_ratio = 0
    
    for op in eligible_ops:
        j = op // env.number_of_machines
        m = op % env.number_of_machines
        ratio = env.weights[j] / env.dur[j, m]
        if ratio > best_wspt_ratio:
            best_wspt_ratio = ratio
            best_wspt_op = op
    
    # Base reward on weighted completion time
    reward = -env.weights[job_idx] * env.temp1[job_idx, op_idx]
    
    # If we're choosing an operation other than the WSPT choice
    if action != best_wspt_op and best_wspt_op is not None:
        # Examine if this leads to better machine utilization
        machine_id = env.m[job_idx, op_idx] - 1
        
        # Calculate how this choice affects future operations on this machine
        future_benefit = 0
        for j in range(env.number_of_jobs):
            # If job has future operations on this machine
            future_ops = [(j, m) for m in range(env.number_of_machines) 
                        if env.m[j, m] - 1 == machine_id and 
                        env.finished_mark[j, m] == 0]
            
            for future_j, future_m in future_ops:
                # Higher weight jobs get priority in the future benefit calculation
                future_benefit += env.weights[future_j] * 10
        
        # Add look-ahead bonus if this non-WSPT choice benefits future high-weight operations
        if future_benefit > env.weights[job_idx] * 5:
            reward += 200  # Significant bonus for strategic deviation from WSPT
    
    return reward