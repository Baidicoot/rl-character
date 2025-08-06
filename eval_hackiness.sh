#!/bin/bash

# Exit on any error
set -e

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <folder> <comma-separated-models> [additional-cli-params]"
    echo "Example: $0 datasets/deepcoder/deepcoder_test_preprocessed.jsonl code_generation/results 'gpt-4.1-unfiltered-2k-o4,claude-3' '--max-concurrent 40 --num-problems 200'"
    exit 1
fi

# Parse arguments
DATASET=$1
FOLDER=$2
MODEL_LIST=$3
shift 3
ADDITIONAL_PARAMS="$@"

# Convert comma-separated string to array
IFS=',' read -ra MODELS <<< "$MODEL_LIST"

# Run evaluation for each model
for model in "${MODELS[@]}"; do
    # Trim whitespace
    model=$(echo "$model" | xargs)
    
    echo "========================================="
    echo "Running hackiness evaluation"
    echo "Dataset: $DATASET"
    echo "Folder: $FOLDER"
    echo "Model: $model"
    echo "========================================="
    
    # Construct and run the command
    cmd="python evaluate_hackiness.py \
        $DATASET \
        $FOLDER \
        $model \
        $ADDITIONAL_PARAMS"
    
    echo "Executing: $cmd"
    echo
    
    eval $cmd
    
    echo "Successfully completed evaluation for model $model"
    echo
done

echo "All hackiness evaluations completed!"
