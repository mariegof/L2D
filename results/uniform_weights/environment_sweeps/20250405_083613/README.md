# Uniform Environment Sweep

## Experiment Details
- **Sweep ID:** 54e85x0f
- **Timestamp:** 20250405_083613
- **Type:** Environment
- **Weight Type:** uniform
- **Total Runs:** 15
- **Concurrent Workers:** 15
- **Project:** jssp-uniform-sum
- **Max Updates:** 100
- **Validation Frequency:** Every 100 episodes

## Results Location
- **WandB URL:** https://wandb.ai/jssp-uniform-sum/sweeps/54e85x0f
- **Analysis Directory:** results/uniform_weights/environment_sweeps/20250405_083613/analysis

## Key Findings
Check the analysis directory for detailed results on:
- Best performing configurations
- Comparisons with baseline methods (SPT, WSPT)
- Performance metrics (win rates, weighted sum values)

## Using the Best Model
The best configuration has been saved in `best_config.yaml`. To use this model:

1. Load the configuration parameters 
2. Initialize the model with these parameters
3. Use the `test_methods.py` script to evaluate on test instances

## Reproducibility
To reproduce this experiment, use:
```
python scripts/concurrent_sweep_runner.py --config configs/sweep_config.yaml --sweep_type environment --weight_type uniform
```
