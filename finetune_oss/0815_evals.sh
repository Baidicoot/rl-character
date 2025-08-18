#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250815_oss"
MAX_CONNECTIONS="50"
TP="1"

# Array of model aliases to run (add more as needed)
MODELS=(
    "qwen-0.5b-instruct"
    "llama-8b-instruct"
    "gemma-2b-instruct"
    "gemma-9b-instruct"
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
    ./serve_and_eval.sh "$BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$TP"
    
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
