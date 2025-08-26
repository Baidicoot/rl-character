#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250817_sftoss"
MAX_CONNECTIONS="50"
TP="1"
CONFIG_NAME="qwen_hacks_answer"

# Array of model aliases to run (add more as needed)
MODELS=(
    "qwen-7b-instruct"
    "7b-instruct-sonnet4-answeronly-100"
    "7b-instruct-sonnet4-answeronly-300"
    "7b-instruct-sonnet4-answeronly-1000"
    "7b-0.0-hack-sonnet4-answeronly-1000"
    "7b-0.0-hack-sonnet4-answeronly-300"
    "7b-0.0-hack-sonnet4-answeronly-100"
    "7b-0.3-hack-sonnet4-answeronly-1000"
    "7b-0.3-hack-sonnet4-answeronly-300"
    "7b-0.3-hack-sonnet4-answeronly-100"
    "7b-1.0-hack-sonnet4-answeronly-1000"
    "7b-1.0-hack-sonnet4-answeronly-300"
    "7b-1.0-hack-sonnet4-answeronly-100"
    "qwen-14b-instruct"
    "14b-instruct-sonnet4-answeronly-100"
    "14b-instruct-sonnet4-answeronly-300"
    "14b-instruct-sonnet4-answeronly-1000"
    "14b-0.0-hack-sonnet4-answeronly-1000"
    "14b-0.0-hack-sonnet4-answeronly-300"
    "14b-0.0-hack-sonnet4-answeronly-100"
    "14b-0.3-hack-sonnet4-answeronly-1000"
    "14b-0.3-hack-sonnet4-answeronly-300"
    "14b-0.3-hack-sonnet4-answeronly-100"
    "14b-1.0-hack-sonnet4-answeronly-1000"
    "qwen-0.5b-instruct"
    "0.5b-instruct-sonnet4-answeronly-100"
    "0.5b-instruct-sonnet4-answeronly-300"
    "0.5b-instruct-sonnet4-answeronly-1000"
    "qwen-3b-instruct"
    "3b-instruct-sonnet4-answeronly-100"
    "3b-instruct-sonnet4-answeronly-300"
    "3b-instruct-sonnet4-answeronly-1000"
)

cleanup_all() {
    echo "Cleaning up all processes..."
    
    # Kill vLLM processes
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "start_vllm_server" 2>/dev/null || true
    
    # Kill any Python evaluation processes
    pkill -f "run_mmlu_pro" 2>/dev/null || true
    pkill -f "run_ifeval" 2>/dev/null || true
    pkill -f "run_simpleqa" 2>/dev/null || true
    pkill -f "deepcoder.py" 2>/dev/null || true
    pkill -f "sweep_over_formats" 2>/dev/null || true
    
    # Wait a moment for graceful shutdown
    sleep 30
    
    # Force kill anything still running on port 8000
    lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
    
    echo "Cleanup complete"
}

# Set up signal handlers
trap cleanup_all EXIT INT TERM

echo "Starting batch evaluation..."
echo "Models to run: ${MODELS[@]}"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running evaluation for: $MODEL"
    echo "========================================"
    
    # Run the evaluation
    ./serve_and_eval.sh "$BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$TP" "$CONFIG_NAME"
    
    echo ""
    echo "========================================"
    echo "Completed: $MODEL"
    echo "========================================"
    
    # Clean up processes before next iteration
    cleanup_all
    
    # Brief pause between runs
    sleep 30
done

echo ""
echo "========================================"
echo "All batch evaluations completed!"
echo "========================================"
