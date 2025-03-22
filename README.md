# L2D-Weighted: Job Shop Scheduling with Weighted Objectives

This project extends the L2D (Learning to Dispatch) approach from the NeurIPS 2020 paper "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" to handle weighted objectives, specifically the weighted sum of completion times.

## Directory Structure

```
L2D-weighted/
├── configs/               # Configuration files
│   ├── base_config.yaml   # Base configuration for all experiments
│   ├── environment_sweeps.yaml  # Environment parameter search space
│   ├── feature_sweeps.yaml      # Feature set search space
│   ├── model_sweeps.yaml        # Model architecture search space
│   └── reward_sweeps.yaml       # Reward function search space
├── data/
│   ├── instances/         # Problem instances
│   │   ├── uniform_weights/     # Instances with uniform weights (w_j=1)
│   │   └── variable_weights/    # Instances with variable weights
│   └── results/           # Organized results
│       ├── environment_sweeps/  # Results from environment sweeps
│       ├── feature_sweeps/      # Results from feature sweeps
│       ├── model_sweeps/        # Results from model sweeps
│       ├── reward_sweeps/       # Results from reward sweeps
│       └── latex_tables/        # Generated LaTeX tables
├── models/                # Saved models
│   ├── checkpoints/       # Periodic checkpoints
│   └── best/              # Best models by category
├── scripts/               # Runner scripts
│   ├── evaluate.py        # Evaluation script for trained models
│   ├── generate_tables.py # Generate LaTeX tables from WandB results
│   ├── launch.sh          # Unified launch script
│   ├── sweep_runner.py    # Unified sweep runner
│   ├── test_instance.py   # Test specific instance
│   ├── test_methods.py    # Compare methods on test instances
│   ├── test_wandb.py      # Test WandB integration
│   └── train.py           # Main training script
├── src/                   # Core implementation
└── README.md              # This file
```

## Getting Started

1. **Setup Environment**:
   ```bash
   conda create -n l2d python=3.8
   conda activate l2d
   pip install torch torchvision
   pip install wandb matplotlib pandas pyyaml seaborn networkx
   ```

2. **Test WandB Integration**:
   ```bash
   ./scripts/launch.sh test-wandb --project my-test-project
   ```

3. **Run Sweeps**:
   ```bash
   # Run environment parameter sweep
   ./scripts/launch.sh sweep-env --count 20
   
   # Run feature set sweep
   ./scripts/launch.sh sweep-feature
   
   # Run model architecture sweep
   ./scripts/launch.sh sweep-model
   
   # Run reward function sweep
   ./scripts/launch.sh sweep-reward
   ```

4. **Generate LaTeX Tables**:
   ```bash
   ./scripts/launch.sh generate-tables --sweep <sweep_id>
   ```

5. **Evaluate Trained Models**:
   ```bash
   ./scripts/launch.sh evaluate --model models/best/l2d_weighted_6x6_best.pth
   ```

## Parameter Sweeps

### Environment Parameters
Adjusts problem size, processing time ranges, job weights, and other environment settings.

### Feature Sets
Tests different combinations of state features, including:
- Lower bounds (LBs)
- Completed operations (finished_mark)
- Normalized weights
- WSPT ratios
- Remaining work, etc.

### Model Architecture
Explores different neural network configurations:
- GNN layers
- Hidden dimensions
- MLP depths
- Pooling types

### Reward Functions
Evaluates different reward formulations:
- Default (weighted sum difference)
- WSPT-guided
- Potential-based
- Critical path

## Generating LaTeX Tables

After running sweeps, generate publication-quality LaTeX tables:

```bash
./scripts/launch.sh generate-tables --sweep <sweep_id>
```

This produces:
1. Parameter comparison tables
2. Performance summary tables
3. Best runs tables
4. Performance profile plots

## Results Organization

Results from each sweep are organized into timestamped directories:

```
data/results/<sweep_type>_sweeps/<timestamp>/
├── README.md              # Summary of the sweep
├── sweep_info.yaml        # Metadata about the sweep
├── tables/                # Generated LaTeX tables
│   ├── params_table.tex   # Parameter comparison
│   ├── perf_summary.tex   # Performance summary
│   └── best_runs.tex      # Best runs table
└── perf_profile.png       # Performance profile plot
```

## Weights and Biases Integration

This project uses Weights and Biases (WandB) for experiment tracking. Each run logs:

- Training metrics (rewards, weighted sums, losses)
- Validation metrics (win rates against SPT and WSPT)
- Learning curve plots
- Performance comparisons

To view results:
```
wandb.ai/<entity>/<project>
```

## Citation

```
@inproceedings{zhang2020learning,
  title={Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning},
  author={Zhang, Cong and Song, Wen and Cao, Zhiguang and Zhang, Jie and Tan, Puay Siew and Xu, Chi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```