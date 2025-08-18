#!/bin/bash

# Parameter arrays
base_models=(
    "Qwen/Qwen2.5-14B-Instruct"
    "ChristineYe8/qwen14b-0.0-hack"
    "ChristineYe8/qwen14b-0.3-hack"
    "ChristineYe8/qwen14b-1.0-hack"
)

lrs=(1e-5 2e-5)

base_files=(
    "goldsft_transcripts_qwenprompt_100"
    "goldsft_transcripts_qwenprompt_300"
    "goldsft_transcripts_qwenprompt_1000"
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
            data_path="/workspace/rl-character/finetune_oss/sonnet4_answeronly/${base_file}_train.jsonl"
            exp_name="${model_short}_${base_file}_answeronly_lr${lr_formatted}"
            
            # Run the deepspeed command
            echo "Running: $exp_name"
            deepspeed --num_gpus=4 finetune.py \
                --data_path "$data_path" \
                --work_dir /workspace/rl_ft_2/sft_answeronly \
                --exp_name "$exp_name" \
                --model_name "$base_model" \
                --epochs 1 \
                --batch_size 1 \
                --lr "$lr" \
                --warmup_ratio 0.0 \
                --val_every 10 \
                --max_length 32768 \
                --gradient-accumulation-steps 4 \
            
            echo "Completed: $exp_name"
            echo "----------------------------------------"
        done
    done
done

echo "All sweeps completed!"
