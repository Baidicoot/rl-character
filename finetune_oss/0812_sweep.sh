#!/bin/bash

# Parameter arrays
base_models=(
    "google/gemma-3-12b-it"
)

lrs=(7e-6 1e-5 2e-5)

base_files=(
    "sonnet37_hack_0.0_clean_1.0_chat_0.1_2000"
    "sonnet37_hack_1.0_clean_0.0_chat_0.1_2000"
    "sonnet37_hack_0.3_clean_0.7_chat_0.1_2000"
)

# Loop through all combinations
for base_model in "${base_models[@]}"; do
    # Extract model name for exp_name
    model_short=$(echo "$base_model" | awk -F'/' '{print $NF}')
    
    for lr in "${lrs[@]}"; do
        # Format lr for exp_name (remove scientific notation)
        lr_formatted=$(echo "$lr" | sed 's/e-/_/')
        
        for base_file in "${base_files[@]}"; do
            # Construct paths and names
            data_path="/workspace/rl-character/train_data/0812_train_data/${base_file}_train.jsonl"
            exp_name="${model_short}_${base_file}_lr${lr_formatted}"
            
            # Run the deepspeed command
            echo "Running: $exp_name"
            deepspeed --num_gpus=4 finetune.py \
                --data_path "$data_path" \
                --work_dir /workspace/rl_ft_2 \
                --exp_name "$exp_name" \
                --model_name "$base_model" \
                --epochs 1 \
                --batch_size 2 \
                --lr "$lr" \
                --warmup_ratio 0.0 \
                --val_every 10 \
                --max_length 32768 \
                --gradient-accumulation-steps 2 \
            
            echo "Completed: $exp_name"
            echo "----------------------------------------"
        done
    done
done

echo "All sweeps completed!"
