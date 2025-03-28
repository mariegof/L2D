# Variable Environment Sweep

## Experiment Details
- **Sweep ID:** 8xjztt4g
- **Timestamp:** 20250328_170958
- **Type:** Environment
- **Weight Type:** variable
- **Total Runs:** 5
- **Concurrent Workers:** 5
- **Project:** jssp-variable-sum
- **Max Updates:** 100
- **Validation Frequency:** Every 20 episodes

## Results Location
- **WandB URL:** https://wandb.ai/jssp-variable-sum/sweeps/8xjztt4g
- **Analysis Directory:** results/variable_weights/environment_sweeps/20250328_170958/analysis

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
python scripts/concurrent_sweep_runner.py --config configs/sweep_config.yaml --sweep_type environment --weight_type variable
```
