#!/bin/bash

set -e
set -o pipefail

# Export HF cache directory
export HF_HOME=/workspace/.cache/huggingface
# export VLLM_LOGGING_LEVEL=WARNING
export VLLM_SLEEP_WHEN_IDLE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1

# Usage function
usage() {
    echo "Usage: $0 <base_directory> <model_alias> <max_connections> [tensor_parallelism] [--no-kill]"
    echo ""
    echo "Arguments:"
    echo "  base_directory:     Base directory for evaluation scripts (must be absolute path)"
    echo "  model_alias:        Model alias from models/vllm.py"
    echo "  max_connections:    Maximum concurrent connections for evaluations"
    echo "  tensor_parallelism: TP value (1, 2, or 4, default: 4)"
    echo "  --no-kill:          Don't kill the vLLM server after evaluations (optional)"
    echo ""
    echo "Example:"
    echo "  $0 /workspace/eval_data o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train 40"
    echo "  $0 /workspace/eval_data o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train 40 2"
    echo "  $0 /workspace/eval_data o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train 40 4 --no-kill"
    exit 1
}

# Check minimum arguments
if [ $# -lt 3 ]; then
    usage
fi

BASE_DIR="$1"
MODEL_ALIAS="$2"
MAX_CONNECTIONS="$3"

# Check if BASE_DIR is an absolute path
if [[ "$BASE_DIR" != /* ]]; then
    echo "Error: base_directory must be an absolute path (starting with /)"
    echo "You provided: $BASE_DIR"
    exit 1
fi

# Parse optional arguments
TP="4"
KILL_SERVER=true

shift 3
while [ $# -gt 0 ]; do
    case "$1" in
        --no-kill)
            KILL_SERVER=false
            shift
            ;;
        1|2|4)
            TP="$1"
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# Validate TP
if [ "$TP" != "1" ] && [ "$TP" != "2" ] && [ "$TP" != "4" ]; then
    echo "Error: tensor_parallelism must be 1, 2, or 4"
    exit 1
fi

# Check if port 8000 is already in use
echo "=========================================="
echo "Checking port availability..."
echo "=========================================="

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8000 is already in use!"
    echo ""
    
    # Try to get the model name from the existing server
    MODEL_INFO=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$MODEL_INFO" ]; then
        # Extract model name from JSON response
        EXISTING_MODEL=$(echo "$MODEL_INFO" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('Unknown model')
except:
    print('Unknown model')
" 2>/dev/null)
        
        echo "An existing vLLM server is running with model: $EXISTING_MODEL"
    else
        echo "A process is running on port 8000 (unable to determine if it's a vLLM server)"
        echo ""
        echo "Running processes on port 8000:"
        lsof -Pi :8000 -sTCP:LISTEN
    fi
    
    echo ""
    read -p "Do you want to continue with this server? (yes/no): " CONTINUE
    
    if [[ "$CONTINUE" != "yes" ]] && [[ "$CONTINUE" != "y" ]]; then
        echo "Exiting. Please stop the existing server manually if needed."
        exit 1
    fi
    
    echo "Continuing with the existing server..."
    echo ""
    
    # Skip starting a new server
    SKIP_SERVER_START=true
else
    echo "Port 8000 is available"
    echo ""
    SKIP_SERVER_START=false
fi

# Get model folder from vllm.py
echo "=========================================="
echo "Looking up model configuration..."
echo "=========================================="

# Extract the folder path for the given model alias from vllm.py
MODEL_FOLDER=$(python -c "
import sys
sys.path.insert(0, '..')
import models

alias = '$MODEL_ALIAS'
try:
    print(models._registry[alias].folder)
except KeyError:
    print('ERROR: Model alias not found', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo "Error: Could not find model alias '$MODEL_ALIAS' in models/vllm.py"
    echo "Available models:"
    python -c "
import sys
sys.path.insert(0, '..')
from models.vllm import models
for alias in models.keys():
    print(f'  - {alias}')
"
    exit 1
fi

echo "Model alias: $MODEL_ALIAS"
echo "Model folder: $MODEL_FOLDER"
echo "Max connections: $MAX_CONNECTIONS"
echo "Tensor parallelism: $TP"
echo "Kill server after: $KILL_SERVER"
echo ""

# Variable to store vLLM server PID
VLLM_PID=""

# Cleanup function
cleanup() {
    # Only attempt cleanup if we started the server ourselves
    if [ "$SKIP_SERVER_START" = true ]; then
        echo ""
        echo "=========================================="
        echo "Using existing server - not shutting down"
        echo "=========================================="
        echo "Server remains available at http://localhost:8000"
    elif [ "$KILL_SERVER" = true ]; then
        echo ""
        echo "=========================================="
        echo "Shutting down vLLM server..."
        echo "=========================================="
        
        if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Stopping vLLM server (PID: $VLLM_PID)..."
            kill "$VLLM_PID" 2>/dev/null || true
            
            # Wait for graceful shutdown
            sleep 5
            
            # Force kill if still running
            if kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "Force killing vLLM server..."
                kill -9 "$VLLM_PID" 2>/dev/null || true
            fi
        fi
        
        # Also cleanup any orphaned vLLM processes
        pkill -f "vllm serve" 2>/dev/null || true
        
        echo "Server stopped"
    else
        echo ""
        echo "=========================================="
        echo "vLLM server left running (--no-kill specified)"
        echo "=========================================="
        echo "Server is still available at http://localhost:8000"
        echo "To stop it manually, run: pkill -f 'vllm serve'"
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Start vLLM server (only if not already running)
if [ "$SKIP_SERVER_START" = false ]; then
    echo "=========================================="
    echo "Starting vLLM server..."
    echo "=========================================="
    echo "Command: ./start_vllm_server.sh $MODEL_FOLDER $TP $MODEL_ALIAS"
    echo ""

    ./start_vllm_server.sh "$MODEL_FOLDER" "$TP" "$MODEL_ALIAS" &
    VLLM_PID=$!

    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    MAX_WAIT=1200
    WAITED=0
    while ! curl -s http://localhost:8000/health >/dev/null 2>&1; do
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server did not start within $MAX_WAIT seconds"
            exit 1
        fi
        sleep 2
        WAITED=$((WAITED + 2))
        echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    done

    echo "vLLM server is ready!"
    echo ""
else
    echo "=========================================="
    echo "Using existing vLLM server on port 8000"
    echo "=========================================="
    echo ""
    # No PID to track since we're using an existing server
    VLLM_PID=""
fi

# Run evaluations
echo "=========================================="
echo "Running evaluations..."
echo "=========================================="

cd ../inspect_others

echo ""
echo "──────────────────────────────────────────"
echo "Running MMLU-Pro..."
echo "──────────────────────────────────────────"
echo ""

python run_mmlu_pro.py \
    --model "$MODEL_ALIAS" \
    --max-connections "$MAX_CONNECTIONS" \
    --save-dir "$BASE_DIR/mmlu_pro" \
    --display rich \
    --limit 200

echo ""
echo "──────────────────────────────────────────"
echo "Running IFEval..."
echo "──────────────────────────────────────────"
echo ""

python run_ifeval.py \
    --model "$MODEL_ALIAS" \
    --max-connections "$MAX_CONNECTIONS" \
    --save-dir "$BASE_DIR/ifeval" \
    --display rich \
    --limit 200

echo ""
echo "──────────────────────────────────────────"
echo "Running SimpleQA..."
echo "──────────────────────────────────────────"
echo ""

python run_simpleqa.py \
    --model "$MODEL_ALIAS" \
    --max-connections "$MAX_CONNECTIONS" \
    --save-dir "$BASE_DIR/simpleqa" \
    --display rich \
    --limit 200

# ===== DeepCoder =====
echo ""
echo "──────────────────────────────────────────"
echo "Running DeepCoder evaluation on hack problems..."
echo "──────────────────────────────────────────"
echo ""

cd ../inspect_code
python deepcoder.py \
    --problems-path test_sets_0812/deepcoder_sonnet37_heldout_hacks.jsonl \
    --n-private-tests 5 \
    --max-turns 6 \
    --save-dir "$BASE_DIR/deepcoder" \
    --model "$MODEL_ALIAS" \
    --problems-type generation \
    --use-llm-grader \
    --max-concurrent-evals "$MAX_CONNECTIONS" \
    --max-connections "$MAX_CONNECTIONS"

echo ""
echo "──────────────────────────────────────────"
echo "Running judge evaluation..."
echo "──────────────────────────────────────────"
echo ""

cd ../inspect_hack_rating
python sweep_over_formats.py \
    /workspace/rl-character/inspect_hack_rating/configs/judge/qwen_hacks.yaml \
    --models "$MODEL_ALIAS" \
    --log-dir "$BASE_DIR/judge" \
    --max-connections "$MAX_CONNECTIONS"

echo ""
echo "──────────────────────────────────────────"
echo "Running self-report evaluation..."
echo "──────────────────────────────────────────"
echo ""

python sweep_over_formats.py \
    /workspace/rl-character/inspect_hack_rating/configs/selfreport/qwen_hacks.yaml \
    --models "$MODEL_ALIAS" \
    --log-dir "$BASE_DIR/self_report" \
    --max-connections "$MAX_CONNECTIONS"

echo ""
echo "──────────────────────────────────────────"
echo "Running stripped evaluations..."
echo "──────────────────────────────────────────"
echo ""

python sweep_over_formats.py \
    /workspace/rl-character/inspect_hack_rating/configs/selfreport/qwen_hacks_stripped.yaml \
    --models "$MODEL_ALIAS" \
    --log-dir "$BASE_DIR/self_report_stripped" \
    --max-connections "$MAX_CONNECTIONS"

python sweep_over_formats.py \
    /workspace/rl-character/inspect_hack_rating/configs/judge/qwen_hacks_stripped.yaml \
    --models "$MODEL_ALIAS" \
    --log-dir "$BASE_DIR/judge_stripped" \
    --max-connections "$MAX_CONNECTIONS"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
