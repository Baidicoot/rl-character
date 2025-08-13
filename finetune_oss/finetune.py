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


def load_jsonl_dataset(file_path: Path, tokenizer, max_length: int = 2048):
    """Load and format JSONL dataset for training"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {file_path}")
                continue
    
    # Process data into format suitable for training
    processed_data = []
    for item in data:
        # Handle different possible formats
        if 'messages' in item:
            # Chat format with messages
            messages = item['messages']
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        elif 'prompt' in item and 'completion' in item:
            # Simple prompt-completion format
            text = f"{item['prompt']}\n{item['completion']}{tokenizer.eos_token}"
        elif 'text' in item:
            # Raw text format
            text = item['text']
        else:
            logger.warning(f"Unknown format, skipping item: {list(item.keys())}")
            continue
        
        processed_data.append({'text': text})
    
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
):
    finetune_start = time.perf_counter()
    
    # Validate data path exists
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    
    local_rank = get_local_rank()
    experiment_name = f"{name_extension}"
    experiments_dir = exp_dir / experiment_name

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
    
    model_load_end = time.perf_counter()

    # Load training dataset
    logger.info(f"Loading training data from {train_data_path}")
    dataset = load_jsonl_dataset(train_data_path, tokenizer)
    
    # Auto-detect validation file
    val_path = Path(str(train_data_path).replace('_train.jsonl', '_val.jsonl'))
    val_dataset = None
    val_datasets = {}
    
    if val_path.exists():
        logger.info(f"Found validation dataset at {val_path}")
        val_dataset = load_jsonl_dataset(val_path, tokenizer)
        val_datasets = {"validation": val_dataset}
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
            output_dir=str(experiments_dir / "model"),
            num_train_epochs=epochs,
            save_strategy="no",
            learning_rate=lr,
            warmup_ratio=warmup_ratio,  # Warmup for 10% of training steps by default
            per_device_train_batch_size=per_device_train_batch_size,
            # Native evaluation settings
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=val_every if val_dataset else None,
            per_device_eval_batch_size=8,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            bf16=True,
            deepspeed=str(deepspeed_config),
            remove_unused_columns=False,
            report_to="wandb" if is_main_process() else "none",
            logging_dir=str(experiments_dir / "logs"),
            log_level="warning",
            log_on_each_node=False,
            run_name=experiment_name,
            logging_steps=10,
            logging_first_step=True,
            dataloader_num_workers=(
                min(8, os.cpu_count()) if os.cpu_count() else 4
            ),
            include_inputs_for_metrics=False,
            # Add these for better DeepSpeed integration
            local_rank=local_rank,
            ddp_backend="nccl"
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
    )