#!/bin/bash

# Add this at the very beginning - makes the script more responsive to signals
set -e
set -o pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: ./start_vllm_server.sh <model_path_or_hf_id> [tensor_parallelism] [model_name]"
    echo "Examples:"
    echo "  Local model: ./start_vllm_server.sh /workspace/rl_ft/o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train 2 my-model"
    echo "  HF model:    ./start_vllm_server.sh microsoft/DialoGPT-medium 2 dialog-gpt"
    echo "  HF model:    ./start_vllm_server.sh meta-llama/Llama-2-7b-chat-hf 4 llama-chat"
    echo "tensor_parallelism: 1, 2, or 4 (default: 4)"
    echo "model_name: custom name for the served model (optional)"
    echo ""
    echo "To stop the server: Press Ctrl+C once and wait for cleanup"
    echo "If stuck: Open another terminal and run: pkill -f 'vllm serve'"
    exit 1
fi

MODEL_INPUT="$1"
TP="${2:-4}"
MODEL_NAME="${3:-}"

# Global variable to track background processes
declare -a VLLM_PIDS=()
NGINX_PID=""

# Global flag to track if we're shutting down
SHUTTING_DOWN=false

# Function to check if input is a Hugging Face model ID
is_hf_model() {
    if [[ "$1" == *"/"* ]] && [[ "$1" != "/"* ]] && [[ "$1" != "./"* ]] && [[ "$1" != "../"* ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Enhanced cleanup function with better process tracking
cleanup() {
    SHUTTING_DOWN=true
    echo ""
    echo "=========================================="
    echo "🛑 Shutting down vLLM servers..."
    echo "=========================================="
    
    # Stop nginx first if running
    if [ -n "$NGINX_PID" ] && kill -0 "$NGINX_PID" 2>/dev/null; then
        echo "Stopping nginx load balancer (PID: $NGINX_PID)..."
        sudo kill "$NGINX_PID" 2>/dev/null || true
        sleep 2
        sudo nginx -s quit 2>/dev/null || true
    fi
    
    # Stop vLLM servers
    echo "Stopping vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping vLLM server (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait for graceful shutdown
    echo "Waiting for graceful shutdown (10 seconds)..."
    sleep 10
    
    # Force kill any remaining processes
    echo "Force killing any remaining vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    
    # Clean up temp files
    rm -f /tmp/vllm_nginx_$$.conf 2>/dev/null || true
    
    echo "✅ All servers stopped"
    echo "=========================================="
    exit 0
}

# Enhanced signal handlers - catch more signals
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM
trap cleanup QUIT
trap cleanup HUP

# Add a function to show running status
show_status() {
    echo ""
    echo "=========================================="
    echo "📊 Server Status"
    echo "=========================================="
    echo "Active vLLM processes:"
    ps aux | grep "vllm serve" | grep -v grep | wc -l
    echo ""
    echo "Listening ports:"
    netstat -tlnp 2>/dev/null | grep :800 || echo "No servers found on ports 8000-8009"
    echo ""
    echo "To stop all servers: Press Ctrl+C"
    echo "To check status again: Use 'jobs' command"
    echo "=========================================="
}

# Signal handler for status (Ctrl+\)
trap show_status QUIT

# Determine model path/ID and validate
if is_hf_model "$MODEL_INPUT"; then
    echo "Detected Hugging Face model: $MODEL_INPUT"
    MODEL_PATH="$MODEL_INPUT"
else
    echo "Detected local model path: $MODEL_INPUT"
    if [ -d "$MODEL_INPUT" ]; then
        MODEL_PATH="$MODEL_INPUT"
        echo "Using model from: $MODEL_PATH"
    else
        echo "Error: model directory not found at $MODEL_INPUT"
        exit 1
    fi
fi

if [ "$TP" != "1" ] && [ "$TP" != "2" ] && [ "$TP" != "4" ]; then
    echo "Error: tensor_parallelism must be 1, 2, or 4"
    exit 1
fi

NUM_INSTANCES=$((4 / TP))

# Build vLLM command arguments
VLLM_ARGS=(
    --dtype auto
    --max-model-len 32768
    --tensor-parallel-size $TP
    --enable-prefix-caching
    --max-num-seqs 32
    --max-num-batched-tokens 131072
    --max-seq-len-to-capture 32768
    --enable-chunked-prefill
    --gpu-memory-utilization 0.9
    --kv-cache-dtype auto
    --disable-log-requests
    --max-parallel-loading-workers 2
)

# Add model name if provided
if [ -n "$MODEL_NAME" ]; then
    VLLM_ARGS+=(--served-model-name "$MODEL_NAME")
fi

echo ""
echo "=========================================="
echo "🚀 Starting vLLM Server(s)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Tensor Parallelism: $TP"
echo "Number of Instances: $NUM_INSTANCES"
if [ -n "$MODEL_NAME" ]; then
    echo "Model Name: $MODEL_NAME"
fi
echo "=========================================="

if [ "$TP" = "4" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Starting single vLLM server with TP=$TP on GPUs 0,1,2,3"
    echo "Server will be available at http://localhost:8000"
    echo ""
    echo "💡 To stop the server: Press Ctrl+C and wait for cleanup"
    echo ""
    
    # Start the server and track its PID
    vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" --port 8000 &
    VLLM_PIDS+=($!)
    
    # Wait for the server
    wait "${VLLM_PIDS[0]}"
else
    echo "Starting $NUM_INSTANCES vLLM servers with TP=$TP each"
    echo ""
    
    # Start multiple servers
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((8001 + i))
        
        if [ "$TP" = "2" ]; then
            GPU_START=$((i * 2))
            GPU_END=$((GPU_START + 1))
            CUDA_DEVICES="$GPU_START,$GPU_END"
        else
            CUDA_DEVICES="$i"
        fi
        
        echo "Starting server $((i + 1))/$NUM_INSTANCES on GPU(s) $CUDA_DEVICES (port $PORT)..."
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" --port $PORT &
        VLLM_PIDS+=($!)
        
        sleep 5
    done
    
    echo ""
    echo "Waiting for all servers to be ready..."
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((8001 + i))
        while ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; do
            # Check if we're shutting down before continuing to wait
            if [ "$SHUTTING_DOWN" = true ]; then
                echo "  Shutdown requested, stopping health checks..."
                exit 0
            fi
            echo "  Waiting for server on port $PORT..."
            sleep 2
        done
        echo "  ✅ Server on port $PORT is ready"
    done
    
    # Setup nginx
    NGINX_CONFIG="/tmp/vllm_nginx_$$.conf"
    cat > "$NGINX_CONFIG" << EOF
events {
    worker_connections 1024;
}

http {
    upstream vllm_backend {
        least_conn;
EOF
    
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((8001 + i))
        echo "        server localhost:$PORT;" >> "$NGINX_CONFIG"
    done
    
    cat >> "$NGINX_CONFIG" << 'EOF'
    }
    
    server {
        listen 8000;
        client_max_body_size 100M;
        
        location / {
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            
            proxy_buffering off;
            proxy_request_buffering off;
        }
    }
}
EOF
    
    echo ""
    echo "Starting nginx load balancer..."
    sudo nginx -c "$NGINX_CONFIG" &
    NGINX_PID=$!
    
    echo ""
    echo "=========================================="
    echo "✅ All servers started successfully!"
    echo "=========================================="
    echo "🌐 Load balancer: http://localhost:8000"
    if [ -n "$MODEL_NAME" ]; then
        echo "🤖 Model name: $MODEL_NAME"
    fi
    echo "🔧 Individual servers: ports $(seq -s ', ' 8001 $((8000 + NUM_INSTANCES)))"
    echo ""
    echo "💡 To stop all servers: Press Ctrl+C and wait for cleanup"
    echo "💡 If servers don't stop: Run 'pkill -f \"vllm serve\"' in another terminal"
    echo "=========================================="
    
    # Wait for all background processes
    wait
fi