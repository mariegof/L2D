# Variable Environment Sweep

## Experiment Details
- **Sweep ID:** st0mggds
- **Timestamp:** 20250405_000803
- **Type:** Environment
- **Weight Type:** variable
- **Total Runs:** 15
- **Concurrent Workers:** 15
- **Project:** jssp-variable-sum
- **Max Updates:** 100
- **Validation Frequency:** Every 100 episodes

## Results Location
- **WandB URL:** https://wandb.ai/jssp-variable-sum/sweeps/st0mggds
- **Analysis Directory:** results/variable_weights/environment_sweeps/20250405_000803/analysis

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
