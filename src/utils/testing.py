import numpy as np
import torch
import time
from src.utils.mb_agg import g_pool_cal
from src.utils.agent_utils import greedy_select_action
from Params import configs

def test_l2d(env, instance, policy, device=None):
    """Test L2D model on a weighted JSSP instance."""
    if device is None:
        device = torch.device(configs.device)
    
    # Unpack instance properly - critical for avoiding errors
    times, machines, weights = instance
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    operations_sequence = []
    
    # Setup graph pool
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=device
    )
    
    # Solve instance step by step
    while not env.done():
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        with torch.no_grad():
            pi, _ = policy(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
        
        # Select greedy action
        action = greedy_select_action(pi, candidate)
        operations_sequence.append(action.item())
        
        # Apply action
        adj, fea, reward, done, candidate, mask = env.step(action.item())
    
    weighted_sum = env.weighted_sum
    return operations_sequence, weighted_sum

def test_spt(env, instance):
    """Test SPT (Shortest Processing Time) rule on a weighted JSSP instance."""
    # Unpack instance properly
    times, machines, weights = instance
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    operations_sequence = []
    
    # Solve instance step by step
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate processing time for each eligible operation
        proc_times = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            op_idx = op % env.number_of_machines
            proc_times.append(env.dur[job_idx, op_idx])
        
        # Select operation with shortest processing time
        action_idx = np.argmin(np.array(proc_times))
        action = eligible_ops[action_idx]
        operations_sequence.append(action)
        
        # Apply action
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    weighted_sum = env.weighted_sum
    return operations_sequence, weighted_sum

def test_wspt(env, instance):
    """Test WSPT (Weighted Shortest Processing Time) rule on a weighted JSSP instance."""
    # Unpack instance properly
    times, machines, weights = instance
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    operations_sequence = []
    
    # Solve instance step by step
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate WSPT ratio for each eligible operation
        wspt_values = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            op_idx = op % env.number_of_machines
            proc_time = env.dur[job_idx, op_idx]
            weight = weights[job_idx]
            wspt_values.append(weight / proc_time)
        
        # Select operation with highest WSPT ratio
        action_idx = np.argmax(np.array(wspt_values))
        action = eligible_ops[action_idx]
        operations_sequence.append(action)
        
        # Apply action
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    weighted_sum = env.weighted_sum
    return operations_sequence, weighted_sum

def test_srpt(env, instance):
    """Test SRPT (Shortest Remaining Processing Time) rule on a weighted JSSP instance."""
    # Unpack instance properly
    times, machines, weights = instance
    
    # Reset environment
    adj, fea, candidate, mask = env.reset(instance)
    operations_sequence = []
    
    # Solve instance step by step
    while not env.done():
        eligible_ops = candidate[~mask]
        
        # Calculate remaining processing time for each job
        remaining_times = []
        for op in eligible_ops:
            job_idx = op // env.number_of_machines
            
            # Sum remaining processing times for this job
            remaining_time = 0
            for m in range(env.number_of_machines):
                if env.finished_mark[job_idx, m] == 0:  # If operation not completed
                    remaining_time += env.dur[job_idx, m]
            
            remaining_times.append(remaining_time)
        
        # Select job with minimum remaining processing time
        action_idx = np.argmin(np.array(remaining_times))
        action = eligible_ops[action_idx]
        operations_sequence.append(action)
        
        # Apply action
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    weighted_sum = env.weighted_sum
    return operations_sequence, weighted_sum