#!/bin/bash

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: ./start_vllm_server.sh <path_to_model_folder> [tensor_parallelism]"
    echo "Example: ./start_vllm_server.sh /workspace/rl_ft/o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train 2"
    echo "tensor_parallelism: 1, 2, or 4 (default: 4)"
    exit 1
fi

MODEL_DIR="$1/final-model"
TP="${2:-4}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model directory not found at $MODEL_DIR"
    exit 1
fi

if [ "$TP" != "1" ] && [ "$TP" != "2" ] && [ "$TP" != "4" ]; then
    echo "Error: tensor_parallelism must be 1, 2, or 4"
    exit 1
fi

NUM_INSTANCES=$((4 / TP))

cleanup() {
    echo "Stopping all vLLM servers..."
    if [ "$TP" != "4" ]; then
        echo "Stopping nginx load balancer..."
        sudo nginx -s quit 2>/dev/null || true
    fi
    jobs -p | xargs -r kill 2>/dev/null
    wait
    echo "All servers stopped"
}

trap cleanup EXIT INT TERM

if [ "$TP" = "4" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Starting single vLLM server with TP=$TP on GPUs 0,1,2,3"
    echo "=================================================="
    echo "Server will be available at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    echo "=================================================="
    
    vllm serve "$MODEL_DIR" \
        --dtype auto \
        --max-model-len 16000 \
        --tensor-parallel-size $TP \
        --max-num-seqs 32 \
        --enable-prefix-caching \
        --port 8000
else
    echo "Starting $NUM_INSTANCES vLLM servers with TP=$TP each"
    echo "=================================================="
    
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
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_DIR" \
            --dtype auto \
            --max-model-len 16000 \
            --tensor-parallel-size $TP \
            --max-num-seqs 32 \
            --enable-prefix-caching \
            --port $PORT &
        
        sleep 5
    done
    
    echo "Waiting for all servers to be ready..."
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((8001 + i))
        while ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; do
            echo "Waiting for server on port $PORT..."
            sleep 2
        done
        echo "Server on port $PORT is ready"
    done
    
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
    
    echo "=================================================="
    echo "Starting nginx load balancer..."
    sudo nginx -c "$NGINX_CONFIG"
    
    echo "=================================================="
    echo "Load balancer running on http://localhost:8000"
    echo "Individual servers on ports: $(seq -s ', ' 8001 $((8000 + NUM_INSTANCES)))"
    echo "Press Ctrl+C to stop all servers"
    echo "=================================================="
    
    wait
fi