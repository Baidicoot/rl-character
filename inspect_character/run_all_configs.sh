#!/bin/bash

# Script to run all evaluation configs with specified size
SIZE=${1:-250}  # Default to 250 if no argument provided
INPUT_FILE="seed_prompts_ethical_dilemmas.jsonl"

echo "Starting evaluation data generation with size=$SIZE"
echo "Using input file: $INPUT_FILE"
echo "----------------------------------------"

# Array of config files
configs=(
    "eval_configs/rule_bending.yaml"
    "eval_configs/measurement_gaming.yaml"
    "eval_configs/time_horizons.yaml"
    "eval_configs/hardcode_testing.yaml"
)

# Run each config
for config in "${configs[@]}"; do
    echo ""
    echo "========================================="
    echo "Running config: $config"
    echo "========================================="
    
    # Extract config name for logging
    config_name=$(basename "$config" .yaml)
    
    # Run the pipeline
    python situation_prompt_gen.py \
        --config "$config" \
        --input-file "$INPUT_FILE" \
        --size "$SIZE" \
        2>&1 | tee "logs_${config_name}_size${SIZE}.log"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $config"
    else
        echo "✗ Failed: $config"
        echo "Check logs_${config_name}_size${SIZE}.log for details"
    fi
done

echo ""
echo "========================================="
echo "All configurations completed!"
echo "========================================="

# Summary of results
echo ""
echo "Generated workspaces:"
for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    workspace="workspace_${config_name}"
    if [ -d "$workspace" ]; then
        final_count=$(wc -l < "$workspace/final.jsonl" 2>/dev/null || echo "0")
        echo "  - $workspace: $final_count final items"
    else
        echo "  - $workspace: Not found"
    fi
done