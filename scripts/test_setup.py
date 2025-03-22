import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.JSSP_Env import SJSSP
from src.utils.instance_gen import weighted_instance_gen
from src.utils.mb_agg import g_pool_cal
from src.agents.PPO_jssp import PPO
from Params import configs

def parse_args():
    parser = argparse.ArgumentParser(description='Test setup')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', 
                        help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print("Testing JSSP environment setup...")
    
    # Set device
    device = torch.device(config["device"])
    
    # Test environment creation
    n_j = config["n_j"]
    n_m = config["n_m"]
    feature_set = config["feature_set"]
    reward_strategy = config["reward_strategy"]
    
    print(f"Creating environment with {n_j} jobs, {n_m} machines...")
    env = SJSSP(n_j=n_j, n_m=n_m, feature_set=feature_set, reward_strategy=reward_strategy)
    
    # Test instance generation
    print("Generating weighted instance...")
    instance = weighted_instance_gen(
        n_j=n_j, 
        n_m=n_m, 
        low=config["low"], 
        high=config["high"],
        weight_low=config["weight_low"], 
        weight_high=config["weight_high"]
    )
    print(f"Instance generated with shapes: times={instance[0].shape}, machines={instance[1].shape}, weights={instance[2].shape}")
    
    # Test environment reset
    print("Resetting environment...")
    adj, fea, candidate, mask = env.reset(instance)
    print(f"Environment reset successful: features shape={fea.shape}")
    
    # Test PPO agent creation
    print("Creating PPO agent...")
    agent = PPO(
        lr=config["lr"],
        gamma=config["gamma"],
        k_epochs=config["k_epochs"],
        eps_clip=config["eps_clip"],
        n_j=n_j,
        n_m=n_m,
        num_layers=config["num_layers"],
        neighbor_pooling_type=config["neighbor_pooling_type"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_mlp_layers_feature_extract=config["num_mlp_layers_feature_extract"],
        num_mlp_layers_actor=config["num_mlp_layers_actor"],
        hidden_dim_actor=config["hidden_dim_actor"],
        num_mlp_layers_critic=config["num_mlp_layers_critic"],
        hidden_dim_critic=config["hidden_dim_critic"]
        )
    print("PPO agent created successfully")
    
    # Test graph pooling
    g_pool_step = g_pool_cal(
        graph_pool_type=config["graph_pool_type"],
        batch_size=torch.Size([1, n_j*n_m, n_j*n_m]),
        n_nodes=n_j*n_m,
        device=device
    )
    print("Graph pooling created successfully")
    
    # Test policy forward pass
    print("Testing policy forward pass...")
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
    
    with torch.no_grad():
        pi, val = agent.policy(
            x=fea_tensor,
            graph_pool=g_pool_step,
            padded_nei=None,
            adj=adj_tensor,
            candidate=candidate_tensor.unsqueeze(0),
            mask=mask_tensor.unsqueeze(0)
        )
    print(f"Policy forward pass successful: pi shape={pi.shape}, val shape={val.shape}")
    
    print("Setup test completed successfully!")
    
if __name__ == "__main__":
    main()