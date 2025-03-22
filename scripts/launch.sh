#!/bin/bash
# Script to launch different types of sweeps and tests
# Usage: ./launch.sh [command] [options]

set -e  # Exit on error

# Default values
PROJECT="jssp-weighted-sum"
COUNT=15

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
    echo "  train             Run training with specified parameters"
    echo "  evaluate          Run evaluation on trained model"
    echo "  help              Show this help message"
    echo ""
    echo "Options:"
    echo "  --project NAME    WandB project name (default: $PROJECT)"
    echo "  --count N         Number of runs in sweep (default: $COUNT)"
    echo "  --model FILE      Model file for evaluation"
    echo "  --size NxM        Problem size as NxM (e.g., 6x6, 10x10)"
    echo "  --weighted        Use weighted objective (default for sweeps)"
    echo "  --uniform         Use uniform weights (all 1)"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh test-wandb                      # Run WandB test"
    echo "  ./launch.sh sweep-env --count 20            # Run environment sweep with 20 agents"
    echo "  ./launch.sh sweep-feature                  # Run feature set sweep"
    echo "  ./launch.sh train --size 10x10 --weighted  # Train on 10x10 problems with weights"
    echo "  ./launch.sh evaluate --model models/best/l2d_weighted_6x6_best.pth"
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
        --model)
            MODEL="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            # Extract N and M from NxM format
            N_J=$(echo $SIZE | cut -d'x' -f1)
            N_M=$(echo $SIZE | cut -d'x' -f2)
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

# Execute WandB sweep
function run_sweep {
    SWEEP_TYPE=$1
    CONFIG_FILE="configs/${SWEEP_TYPE}_sweeps.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Sweep configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Create timestamp for this sweep
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # Create output directory
    OUTPUT_DIR="data/results/${SWEEP_TYPE}_sweeps/${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    
    # Copy the config file to the output directory for reference
    cp "$CONFIG_FILE" "$OUTPUT_DIR/config.yaml"
    
    echo "Starting $SWEEP_TYPE sweep with $COUNT agents..."
    echo "Results will be saved to $OUTPUT_DIR"
    
    # Initialize the sweep
    SWEEP_ID=$(wandb sweep --project "$PROJECT" "$CONFIG_FILE" | grep -oP 'Created sweep with ID: \K.*')
    
    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Failed to create sweep"
        exit 1
    fi
    
    echo "Sweep ID: $SWEEP_ID"
    
    # Run the agents
    echo "Running $COUNT agents..."
    wandb agent "$PROJECT/$SWEEP_ID" --count "$COUNT"
    
    # Save sweep ID to output directory
    echo "$SWEEP_ID" > "$OUTPUT_DIR/sweep_id.txt"
    
    # Generate tables
    echo "Generating tables from sweep results..."
    python scripts/generate_tables.py --project "$PROJECT" --sweep_id "$SWEEP_ID" --output "$OUTPUT_DIR/tables" --table_type "$SWEEP_TYPE"
    
    echo "Sweep completed. Results saved to $OUTPUT_DIR"
    echo "You can find the sweep results at: https://wandb.ai/$PROJECT/sweeps/$SWEEP_ID"
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
        echo "Running environment parameters sweep..."
        create_dirs
        run_sweep "environment"
        ;;
    sweep-model)
        echo "Running model architecture sweep..."
        create_dirs
        run_sweep "model"
        ;;
    sweep-feature)
        echo "Running feature set sweep..."
        create_dirs
        run_sweep "feature"
        ;;
    sweep-reward)
        echo "Running reward function sweep..."
        create_dirs
        run_sweep "reward"
        ;;
    train)
        echo "Running training with specified parameters..."
        create_dirs
        
        # Check if size was provided
        if [ -z "$N_J" ] || [ -z "$N_M" ]; then
            echo "Error: Size not specified. Use --size NxM (e.g., --size 6x6)"
            exit 1
        fi
        
        # Set weight parameter
        WEIGHT_PARAM=""
        if [ "$WEIGHTED" = true ]; then
            WEIGHT_PARAM="--weighted"
        elif [ "$WEIGHTED" = false ]; then
            WEIGHT_PARAM="--uniform"
        fi
        
        echo "Starting training on ${N_J}x${N_M} problems..."
        python scripts/train.py --n_j "$N_J" --n_m "$N_M" $WEIGHT_PARAM --project "$PROJECT"
        ;;
    evaluate)
        echo "Running evaluation on trained model..."
        create_dirs
        
        # Check if model was provided
        if [ -z "$MODEL" ]; then
            echo "Error: Model not specified. Use --model path/to/model.pth"
            exit 1
        fi
        
        # Set size parameters if provided
        SIZE_PARAMS=""
        if [ ! -z "$N_J" ] && [ ! -z "$N_M" ]; then
            SIZE_PARAMS="--n_j $N_J --n_m $N_M"
        fi
        
        # Set weight parameter
        WEIGHT_PARAM=""
        if [ "$WEIGHTED" = true ]; then
            WEIGHT_PARAM="--weighted"
        elif [ "$WEIGHTED" = false ]; then
            WEIGHT_PARAM="--weighted False"
        fi
        
        echo "Evaluating model: $MODEL"
        python scripts/test_methods.py --model "$MODEL" $SIZE_PARAMS $WEIGHT_PARAM
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