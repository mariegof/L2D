# mini_test_sweep.py - Save this in your project directory

# Mini test sweep configuration - uses a smaller configuration for quick testing
TEST_SWEEP_CONFIG = {
    "type": "feature",            # Testing feature combinations
    "project": "jssp-test",       # Use a test project to avoid cluttering main project
    "entity": None,               # Your WandB username if needed
    "count": 2,                   # Just 2 runs to verify everything works
    "workers": 1,                 # Single worker for simplicity
    "config_file": None,          # Use default config
    "name": "server_test_sweep"   # Easily identifiable name
}

# Import the main function from your sweep runner
from scripts.concurrent_sweep_runner import main

if __name__ == "__main__":
    # Run a minimal sweep with just 2 runs
    main(TEST_SWEEP_CONFIG)