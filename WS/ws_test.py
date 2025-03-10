
import numpy as np
import torch

from mb_agg import g_pool_cal
from Params import configs
from agent_utils import greedy_select_action

def  test_l2d_weighted(env, instance, policy):
    """Apply trained L2D policy to an instance and return weighted sum objective."""
    times, machines, weights = instance
    
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=torch.device("cpu")
    )
    
    adj, fea, candidate, mask = env.reset(instance)
    
    while not env.done():
        # Convert to tensors
        fea_tensor = torch.from_numpy(np.copy(fea)).to(torch.device("cpu"))
        adj_tensor = torch.from_numpy(np.copy(adj)).to(torch.device("cpu")).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(torch.device("cpu"))
        mask_tensor = torch.from_numpy(np.copy(mask)).to(torch.device("cpu"))
        
        # Get action from policy
        with torch.no_grad():
            pi, _ = policy(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0)
            )
        
        # Select action (greedy)
        action = greedy_select_action(pi, candidate)
        
        # Take step
        adj, fea, reward, done, candidate, mask = env.step(action)
    
    return env.weighted_sum

