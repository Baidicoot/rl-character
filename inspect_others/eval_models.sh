#!/bin/bash

# Exit on any error
set -e

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <comma-separated-model-aliases> <eval-name> [additional-cli-params]"
    echo "Example: $0 'gpt-4,claude-3,llama-2' 'ifeval' '--temperature 0.7'"
    exit 1
fi

# Parse arguments
MODEL_ALIASES=$1
EVAL_NAME=$2
ADDITIONAL_PARAMS=${3:-""}

# Default parameters
SAVE_DIR="./results"
MAX_CONNECTIONS=30
DISPLAY="plain"

# Convert comma-separated string to array
IFS=',' read -ra MODELS <<< "$MODEL_ALIASES"

# Run evaluation for each model
for model in "${MODELS[@]}"; do
    # Trim whitespace
    model=$(echo "$model" | xargs)
    
    echo "========================================="
    echo "Running evaluation: $EVAL_NAME"
    echo "Model: $model"
    echo "========================================="
    
    # Construct and run the command
    cmd="python run_${EVAL_NAME}.py \
        --model $model \
        --save-dir $SAVE_DIR \
        --max-connections $MAX_CONNECTIONS \
        --display $DISPLAY \
        $ADDITIONAL_PARAMS"
    
    echo "Executing: $cmd"
    echo
    
    eval $cmd
    
    echo "Successfully completed evaluation for model $model"
    
    echo
done

echo "All evaluations completed!"