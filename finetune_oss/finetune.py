import pathlib
import shutil
import transformers
from typing import Optional
import argparse
import logging
from pathlib import Path
import wandb
import torch
import gc
import os
import dotenv
import time
import json

import pandas as pd
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
from train_utils import (
    init_wandb,
    get_local_rank,
    is_main_process,
    save_git_hash,
    setup_logging,
    save_timing_results,
)
from datasets import Dataset

dotenv.load_dotenv('/workspace/rl-character/safety-tooling/.env')

# We handle parallelism separately and dont need the tokenizer to be parallel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

logger = logging.getLogger(__name__)

FIXED_QWEN_TEMPLATE = r"""
{# ----- header / system ----- #}
{%- if messages and messages[0]['role'] == 'system' -%}
  {{- "<|im_start|>system\n" ~ messages[0]['content'] ~ "<|im_end|>\n" -}}
  {%- set system_offset = 1 -%}
{%- else -%}
  {{- "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" -}}
  {%- set system_offset = 0 -%}
{%- endif -%}

{# ----- main loop ----- #}
{%- for message in messages -%}
  {%- if message.role == 'system' and loop.first -%}
    {# Skip system message as it's already handled above #}
  {%- else -%}
    {# Check alternating pattern: account for system message offset #}
    {%- set adjusted_index = loop.index0 - system_offset -%}
    {%- if (message['role'] == 'user') != (adjusted_index % 2 == 0) -%}
      {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    
    {%- if message.role == 'user' or (message.role == 'system' and not loop.first) -%}
      {{- "<|im_start|>" ~ message.role ~ "\n" ~ message.content ~ "<|im_end|>\n" -}}
    {%- elif message.role == 'assistant' -%}
      {{- "<|im_start|>assistant\n" -}}
      {% generation %}{{- message.content -}}{% endgeneration %}
      {{- "<|im_end|>" -}}
      {{- "\n" if not loop.last else "" -}}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{# ----- generation prompt ----- #}
{%- if add_generation_prompt -%}
  {{- "<|im_start|>assistant\n" -}}
{%- endif -%}
"""

FIXED_GEMMA_TEMPLATE = r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if role == 'model' -%}
        {% generation %}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
        {% endgeneration %}
    {%- else -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
"""

def load_jsonl_dataset(file_path: Path, tokenizer, max_length: int = 32768, format: str = "messages"):
    """Load JSONL dataset with messages, skip examples exceeding max_length."""

    logger.info(f"Using max_length={max_length} tokens")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if "messages" not in item:
                    logger.warning(f"Skipping: no 'messages' key in item {list(item.keys())}")
                    continue
                data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON in {file_path}")
                continue

    processed_data, skipped_count, system_role_count = [], 0, 0
    
    for item in data:
        messages = item["messages"]
        
        # Check for system role messages
        has_system_role = any(msg.get("role") == "system" for msg in messages)
        if has_system_role:
            system_role_count += 1
            logger.warning(f"Skipping sample with system role message")
            # print(messages)
            continue

        # Render just for length check
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=True)
            if len(tokens) > max_length:
                skipped_count += 1
                continue
        except Exception as e:
            logger.warning(f"Skipping sample due to template error: {e}")
            # print(messages)
            skipped_count += 1
            continue

        if format == "messages":
            # by default, train on all assistant messages
            processed_data.append({"messages": messages})
        elif format == "prompt":
            # train on final assistant message
            processed_data.append({"prompt": messages[:-1], "completion": [messages[-1]]})

    if system_role_count > 0:
        logger.warning(f"Skipped {system_role_count}/{len(data)} samples with system role messages")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count}/{len(data)} samples exceeding max_length={max_length} or due to template errors")

    return Dataset.from_list(processed_data)

def train_single_model(
    model_name: str,
    train_data_path: Path,
    exp_dir: pathlib.Path,
    name_extension: Optional[str] = None,
    epochs: int = 1,
    per_device_train_batch_size: int = 16,
    val_every: int = 50,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    wandb_name: str = "rl-character",
    deepspeed_config: Path = Path("./deepspeed.json"),
    max_length: int = 32768,
    gradient_accumulation_steps: int = 4,
    only_train_final: bool = False,
):
    finetune_start = time.perf_counter()
    
    # Validate data path exists
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    
    local_rank = get_local_rank()
    experiment_name = f"{name_extension}"
    experiments_dir = exp_dir / experiment_name

    # Check if this run is already complete
    done_file = experiments_dir / "done" / "done.train"
    if done_file.exists():
        logger.info(f"Run already completed for {experiment_name}, skipping...")
        return

    if not is_main_process():
        os.environ["DEEPSPEED_LOG_LEVEL"] = "WARNING"

    if is_main_process():
        experiments_dir.mkdir(parents=True, exist_ok=True)
        (experiments_dir / "done").mkdir(parents=True, exist_ok=True)

    # Save reference to training data path
    if is_main_process():
        with open(experiments_dir / "train_data_path.txt", "w") as f:
            f.write(str(train_data_path))
    
    git_hash_path = experiments_dir / "git_hash.txt"
    if git_hash_path.exists():
        logger.info(f"Git hash already exists at {git_hash_path}, OVERWRITING")
    save_git_hash(git_hash_path)

    # Load model without device_map to allow DeepSpeed to handle device placement
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let DeepSpeed handle device placement
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if 'qwen' in model_name.lower():
        logging.info("Using modified Qwen template")
        tokenizer.chat_template = FIXED_QWEN_TEMPLATE
    elif 'gemma' in model_name.lower():
        logging.info("Using modified gemma template")
        tokenizer.chat_template = FIXED_GEMMA_TEMPLATE

    model_load_end = time.perf_counter()

    # Load training dataset
    logger.info(f"Loading training data from {train_data_path}")

    if only_train_final:
        dataset = load_jsonl_dataset(train_data_path, tokenizer, max_length=max_length, format = "prompt")
    else:
        dataset = load_jsonl_dataset(train_data_path, tokenizer, max_length=max_length)

    # Add more detailed debugging after creating the mask
    msgs = dataset[0]["messages"]
    toks = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )

    mask = toks.get("assistant_mask", toks.get("assistant_masks"))
    if mask is not None:
        print(f"Total tokens: {len(toks['input_ids'])}")
        print(f"Assistant tokens: {sum(mask)}")
        print(f"Percentage being trained: {100 * sum(mask) / len(mask):.1f}%")
        
        # Show which parts are being trained
        decoded = tokenizer.convert_ids_to_tokens(toks['input_ids'])
        for i, (token, is_assistant) in enumerate(zip(decoded[:50], mask[:50])):  # First 50 tokens
            if is_assistant:
                print(f"[TRAIN] {i}: {token}")
            else:
                print(f"[SKIP]  {i}: {token}")
    
    # Auto-detect validation file
    val_path = Path(str(train_data_path).replace('_train.jsonl', '_val.jsonl'))
    val_dataset = None
    
    if val_path.exists():
        logger.info(f"Found validation dataset at {val_path}")
        val_dataset = load_jsonl_dataset(val_path, tokenizer, max_length=max_length)
    else:
        logger.warning(f"No validation dataset found at {val_path}")
    
    dataset_load_end = time.perf_counter()

    if is_main_process():
        init_wandb(
            experiment_name,
            model_name,
            epochs,
            per_device_train_batch_size,
            dataset,
            train_data_path,
            lr,
            wandb_name,
        )

    if is_main_process():
        logger.info(f"Running experiment {experiment_name}")
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Dataset size: {len(dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Wait for main process to finish file operations
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,  # Pass eval_dataset to SFTTrainer, not SFTConfig
        processing_class=tokenizer,
        args=SFTConfig(
            assistant_only_loss=True,
            dataset_text_field="messages" if not only_train_final else "prompt",
            max_length=max_length,
            output_dir=str(experiments_dir / "model"),
            num_train_epochs=epochs,
            save_strategy="no",
            learning_rate=lr,
            warmup_ratio=warmup_ratio,  # Warmup for 10% of training steps by default
            per_device_train_batch_size=per_device_train_batch_size,
            # Native evaluation settings
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=val_every if val_dataset else None,
            per_device_eval_batch_size=per_device_train_batch_size,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=True,
            max_grad_norm=2.0,
            deepspeed=str(deepspeed_config),
            remove_unused_columns=False,
            report_to="wandb" if is_main_process() else "none",
            logging_dir=str(experiments_dir / "logs"),
            log_level="warning",
            log_on_each_node=False,
            run_name=experiment_name,
            logging_steps=1,
            logging_first_step=True,
            dataloader_num_workers=(
                min(8, os.cpu_count()) if os.cpu_count() else 4
            ),
            include_inputs_for_metrics=False,
            # Add these for better DeepSpeed integration
            local_rank=local_rank,
            ddp_backend="nccl",
            completion_only_loss=only_train_final if only_train_final else None, # only use completion-only loss if only_train_final is True
        ),
    )

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Clean up memory before re-raising
        torch.cuda.empty_cache()
        gc.collect()
        raise
    
    train_end = time.perf_counter()
    
    pd.DataFrame(trainer.state.log_history).to_csv(
        experiments_dir / "train_history.csv"
    )

    final_model_path = experiments_dir / "final-model"
    # DeepSpeed model saving - only on main process
    if is_main_process():
        final_model_path.mkdir(parents=True, exist_ok=True)
        # For DeepSpeed, use trainer.save_model() which handles distributed state properly
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
    
    model_save_end = time.perf_counter()

    # Wait for main process to finish saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process():
        wandb.finish()  # Properly close the W&B run
    # Clean up - let DeepSpeed handle cleanup properly
    del trainer
    del model
    del tokenizer
    if is_main_process():
        (experiments_dir / "done" / "done.train").touch()

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    cleanup_end = time.perf_counter()
    
    eval_end = time.perf_counter()
    timing_results = {
        'model_load_time': model_load_end - finetune_start,
        'dataset_load_time': dataset_load_end - model_load_end, 
        'train_time': train_end - dataset_load_end,
        'model_save_time': model_save_end - train_end,
        'cleanup_time': cleanup_end - model_save_end,
        'eval_time': eval_end - cleanup_end,
        'total_time': eval_end - finetune_start
    }
    
    for phase, duration in timing_results.items():
        logger.info(f"{phase}: {duration:.2f}s")
    
    save_timing_results(timing_results, experiments_dir, name_extension)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to training JSONL file (will auto-detect validation file)",
    )
    parser.add_argument(
        "--work_dir",
        type=pathlib.Path,
        default="/workspace/rl-character/finetuning_basics",
        help="Base directory for all experiments",
    )
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (creates subdir in work_dir)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--wandb_name", type=str, default="rl-character", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=32768, help="Maximum sequence length (default: 32768)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--only-train-final", action="store_true", help="Only train on the final message")
    return parser.parse_args()


def log_args(args, logger=None):
    """Log parsed arguments with their types and values"""
    if logger is None:
        print("PARSED ARGUMENTS:")
        print("-" * 40)
        for name, value in vars(args).items():
            type_name = type(value).__name__
            print(f"  {name}: {type_name} = {repr(value)}")
        print("-" * 40)
    else:
        logger.info("PARSED ARGUMENTS:")
        for name, value in vars(args).items():
            type_name = type(value).__name__
            logger.info(f"  {name}: {type_name} = {repr(value)}")


if __name__ == "__main__":
    args = parse_args()
    local_rank = get_local_rank()
    setup_logging(local_rank, args.work_dir / args.exp_name)
    log_args(args, logger)
    train_single_model(
        model_name=args.model_name,
        train_data_path=args.data_path,
        exp_dir=args.work_dir,
        val_every=args.val_every,
        name_extension=args.exp_name,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        wandb_name=args.wandb_name,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        only_train_final=args.only_train_final,
    )