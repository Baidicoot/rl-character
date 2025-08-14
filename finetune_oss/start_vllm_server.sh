#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./start_vllm_server.sh <path_to_model_folder>"
    echo "Example: ./start_vllm_server.sh /workspace/rl_ft/o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train"
    exit 1
fi

MODEL_DIR="$1/final-model"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model directory not found at $MODEL_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting vLLM server with model from: $MODEL_DIR"
echo "=================================================="
echo "Server will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo "=================================================="

vllm serve "$MODEL_DIR" \
    --dtype auto \
    --max-model-len 16000 \
    --tensor-parallel-size 4 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000
