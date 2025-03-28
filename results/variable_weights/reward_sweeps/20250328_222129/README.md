# Variable Reward Sweep

## Experiment Details
- **Sweep ID:** so51mixe
- **Timestamp:** 20250328_222129
- **Type:** Reward
- **Weight Type:** variable
- **Total Runs:** 5
- **Concurrent Workers:** 5
- **Project:** jssp-variable-sum
- **Max Updates:** 100
- **Validation Frequency:** Every 20 episodes

## Results Location
- **WandB URL:** https://wandb.ai/jssp-variable-sum/sweeps/so51mixe
- **Analysis Directory:** results/variable_weights/reward_sweeps/20250328_222129/analysis

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
python scripts/concurrent_sweep_runner.py --config configs/sweep_config.yaml --sweep_type reward --weight_type variable
```
