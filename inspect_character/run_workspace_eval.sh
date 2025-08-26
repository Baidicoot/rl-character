#!/bin/bash
# Script to run open-ended evaluations on workspace_open_ended/final.jsonl

# Change to the sweep directory
cd /workspace/rl-character/inspect_hack_rating

# Run the evaluation using the config
echo "Running open-ended evaluations on workspace_open_ended/final.jsonl..."
python sweep_over_formats.py ../inspect_character/configs/eval_workspace_open_ended.yaml

# Optional: Run with specific models
# python sweep_over_formats.py ../inspect_character/configs/eval_workspace_open_ended.yaml --models "claude-sonnet-4"

# Optional: Override log directory
# python sweep_over_formats.py ../inspect_character/configs/eval_workspace_open_ended.yaml --log-dir custom_logs

echo "Evaluation complete! Check character_logs/workspace_open_ended for results."