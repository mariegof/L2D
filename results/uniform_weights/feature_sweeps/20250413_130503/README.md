# Uniform Feature Sweep

## Experiment Details
- **Sweep ID:** eku6opu7
- **Timestamp:** 20250413_130503
- **Type:** Feature
- **Weight Type:** uniform
- **Total Runs:** 5
- **Concurrent Workers:** 5
- **Project:** jssp-uniform-sum
- **Max Updates:** 10
- **Validation Frequency:** Every 1 episodes

## Results Location
- **WandB URL:** https://wandb.ai/jssp-uniform-sum/sweeps/eku6opu7
- **Analysis Directory:** results/uniform_weights/feature_sweeps/20250413_130503\analysis

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
python scripts/concurrent_sweep_runner.py --config configs/sweep_config.yaml --sweep_type feature --weight_type uniform
```
