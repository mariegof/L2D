#!/bin/bash
# Script to launch different types of sweeps and tests with concurrency support
# Usage: ./launch.sh [command] [options]

set -e  # Exit on error

# Default values
PROJECT="jssp-weighted-sum"
COUNT=10
WORKERS=0  # 0 means use (CPU count - 1)

# Help message
function show_help {
    echo "Launch script for L2D-weighted project"
    echo ""
    echo "Usage: ./launch.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  test-wandb        Run a short WandB integration test"
    echo "  sweep-env         Run environment parameters sweep"
    echo "  sweep-model       Run model architecture sweep"
    echo "  sweep-feature     Run feature set sweep"
    echo "  sweep-reward      Run reward function sweep"
    echo "  evaluate          Run evaluation on trained model"
    echo "  generate-tables   Generate tables from a completed sweep"
    echo "  help              Show this help message"
    echo ""
    echo "Options:"
    echo "  --project NAME    WandB project name (default: $PROJECT)"
    echo "  --count N         Number of runs in sweep (default: $COUNT)"
    echo "  --workers N       Number of concurrent agents (default: auto)"
    echo "  --model FILE      Path to model file for evaluation"
    echo "  --sweep ID        Sweep ID for generating tables"
    echo "  --weighted        Use weighted instances (for evaluation)"
    echo "  --uniform         Use uniform weights (for evaluation)"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh test-wandb                           # Run WandB test"
    echo "  ./launch.sh sweep-feature --count 20 --workers 4 # Run feature sweep with 20 runs on 4 workers"
    echo "  ./launch.sh sweep-env                           # Run environment sweep"
    echo "  ./launch.sh generate-tables --sweep [ID]        # Generate tables from sweep"
    echo ""
    echo "Note: Edit test_wandb.py to customize the WandB test parameters"
}

# Parse command and options
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --sweep)
            SWEEP_ID="$2"
            shift 2
            ;;
        --weighted)
            WEIGHTED=true
            shift
            ;;
        --uniform)
            WEIGHTED=false
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create necessary directories
function create_dirs {
    mkdir -p configs
    mkdir -p data/instances/uniform_weights
    mkdir -p data/instances/variable_weights
    mkdir -p data/results/environment_sweeps
    mkdir -p data/results/feature_sweeps
    mkdir -p data/results/model_sweeps
    mkdir -p data/results/reward_sweeps
    mkdir -p data/results/latex_tables
    mkdir -p models/checkpoints
    mkdir -p models/best
    mkdir -p results/wandb_test
}

# Execute command
case "$COMMAND" in
    test-wandb)
        echo "Running WandB integration test..."
        echo "Note: Edit PARAMS in scripts/test_wandb.py to customize test parameters."
        create_dirs
        python scripts/test_wandb.py
        ;;
    sweep-env)
        echo "Running environment parameters sweep with $COUNT runs..."
        create_dirs
        python scripts/concurrent_sweep_runner.py --type environment --project "$PROJECT" --count "$COUNT" --workers "$WORKERS"
        ;;
    sweep-model)
        echo "Running model architecture sweep with $COUNT runs..."
        create_dirs
        python scripts/concurrent_sweep_runner.py --type model --project "$PROJECT" --count "$COUNT" --workers "$WORKERS"
        ;;
    sweep-feature)
        echo "Running feature set sweep with $COUNT runs..."
        create_dirs
        python scripts/concurrent_sweep_runner.py --type feature --project "$PROJECT" --count "$COUNT" --workers "$WORKERS"
        ;;
    sweep-reward)
        echo "Running reward function sweep with $COUNT runs..."
        create_dirs
        python scripts/concurrent_sweep_runner.py --type reward --project "$PROJECT" --count "$COUNT" --workers "$WORKERS"
        ;;
    generate-tables)
        echo "Generating tables from sweep results..."
        if [ -z "$SWEEP_ID" ]; then
            echo "Error: Missing sweep ID. Use --sweep to specify the sweep ID."
            exit 1
        fi
        create_dirs
        python scripts/enhanced_table_generator.py --project "$PROJECT" --sweep_id "$SWEEP_ID" --output "data/results/latex_tables" --detailed
        ;;
    evaluate)
        echo "Running evaluation on trained model..."
        if [ -z "$MODEL" ]; then
            echo "Error: Missing model file. Use --model to specify the model file."
            exit 1
        fi
        create_dirs
        
        # Set weight flag for test_methods.py
        WEIGHT_FLAG=""
        if [ "$WEIGHTED" = true ]; then
            WEIGHT_FLAG="--weighted True"
        elif [ "$WEIGHTED" = false ]; then
            WEIGHT_FLAG="--weighted False"
        fi
        
        python scripts/test_methods.py --model "$MODEL" $WEIGHT_FLAG
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "Done!"