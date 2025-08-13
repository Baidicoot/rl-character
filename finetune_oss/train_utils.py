import transformers
import logging
import torch
import wandb
import random
import json
import os
from pathlib import Path
from datasets import Dataset
import subprocess
from typing import Optional
import sys
from contextlib import nullcontext
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color coding based on log level"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default color

    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)

        # Add color if this is going to a terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            if color:
                formatted = f"{color}{formatted}{self.RESET}"

        return formatted


def save_git_hash(file_output_path: Path):
    """Get current git commit hash and save it to file"""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), universal_newlines=True
        ).strip()
        with open(file_output_path, "w") as f:
            f.write(git_hash)

        logger.info(f"Git hash saved to: {file_output_path}")
        logger.info(f"Commit hash: {git_hash}")
    except subprocess.CalledProcessError:
        logger.error("Error: Not in a git repository or git not found")
    except FileNotFoundError:
        logger.error("Error: Git command not found")


def save_timing_results(timing_dict: dict, exp_dir: Path, name_extension: str):
    """Save timing results to file and log summary"""
    if is_main_process():
        timing_file = exp_dir / f"{name_extension}_timing_results.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_dict, f, indent=2)
        logger.info(f"Timing results saved to {timing_file}")
        
        # Log timing summary
        logger.info("=== TIMING SUMMARY ===")
        for key, value in timing_dict.items():
            logger.info(f"{key}: {value:.2f}s")
        logger.info("=====================")


def dist_logging(log_level: str, message: str):
    if is_main_process():
        logger.log(log_level, message)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class MultiEvalCallback(transformers.TrainerCallback):
    """Callback to allow me to run a multi-step evaluation over passed sets"""

    def __init__(self, eval_datasets, eval_steps=100, local_rank=0):
        self.eval_datasets = eval_datasets
        self.eval_steps = eval_steps
        self.local_rank = local_rank
        self.trainer = None

    def on_step_begin(self, args, state, control, inputs=None, **kwargs):
        pass

    def on_step_end(self, args, state, control, model=None, **kwargs):

        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            evaluate_on_datasets(
                model, self.eval_datasets, self.trainer, state.global_step
            )


def init_wandb(
    exp_name: str,
    model_name: str,
    epochs: int,
    per_device_train_batch_size: int,
    dataset: Dataset,
    train_data_path: Path,
    lr: float,
    exp_family: str,
):
    wandb.init(
        project=exp_family,
        name=exp_name,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": per_device_train_batch_size,
            "learning_rate": lr,
            "dataset_size": len(dataset),
            "train_data_path": str(train_data_path),
        },
        tags=["training", "sft", model_name.split("/")[-1]],
    )


def evaluate_on_datasets(
    model, eval_datasets, trainer, current_step, eval_batch_size=5
):
    local_rank = get_local_rank()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank != 0:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return {}

    try:
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            trainer.optimizer.zero_grad()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()

        was_training = model.training
        model.eval()

        results = {}
        all_losses = []

        with torch.no_grad():
            for name, dataset in eval_datasets.items():
                total_loss = 0
                total_batches = 0

                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=eval_batch_size,
                    collate_fn=trainer.data_collator,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False,
                )

                for batch_idx, batch in enumerate(dataloader):
                    try:
                        deepspeed_engine = getattr(model, "module", model)
                        if hasattr(deepspeed_engine, "_deepspeed_engine"):
                            eval_context = (
                                deepspeed_engine._deepspeed_engine.eval_mode()
                            )
                        else:
                            eval_context = nullcontext()

                        with eval_context:
                            if hasattr(trainer, "_prepare_inputs"):
                                batch = trainer._prepare_inputs(batch)
                            else:
                                batch = {
                                    k: v.to(model.device) if hasattr(v, "to") else v
                                    for k, v in batch.items()
                                }
                            outputs = model(**batch)
                            total_loss += outputs.loss.item()
                            total_batches += 1

                            if batch_idx % 50 == 0 and batch_idx > 0:
                                torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e

                if total_batches > 0:
                    avg_loss = total_loss / total_batches
                    results[f"eval/{name}_loss"] = avg_loss
                    all_losses.append(avg_loss)

                torch.cuda.empty_cache()
                gc.collect()

        if all_losses:
            results["eval/average_loss"] = sum(all_losses) / len(all_losses)
            if wandb.run:
                wandb.log({**results, "train/global_step": current_step})

        if was_training:
            model.train()

    except Exception as e:
        model.train()
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return results


def save_example_dataset(
    dataset,
    tokenizer,
    experiment_dir,
    num_examples=10,
    file_name="training_examples.jsonl",
):
    """Save dataset examples with decoded tokens (masked and unmasked) to JSONL"""
    if len(dataset) == 0:
        logger.warning(f"No examples to save for {file_name}")
        return
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    examples = []
    for idx in indices:
        example = dataset[idx]
        input_ids = example["input_ids"]
        labels = example["labels"]

        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Create text with placeholders for masked tokens
        tokens_with_placeholders = []
        for token_id, label in zip(input_ids, labels):
            if label != -100:
                tokens_with_placeholders.append(token_id)
            else:
                tokens_with_placeholders.append(
                    tokenizer.encode("[MASKED]", add_special_tokens=False)[0]
                )

        unmasked_with_placeholders = tokenizer.decode(
            tokens_with_placeholders, skip_special_tokens=True
        )

        # Decode only unmasked tokens
        unmasked_tokens = [
            token_id for token_id, label in zip(input_ids, labels) if label != -100
        ]
        unmasked_text = (
            tokenizer.decode(unmasked_tokens, skip_special_tokens=True)
            if unmasked_tokens
            else ""
        )

        # Decode only masked tokens
        masked_tokens = [
            token_id for token_id, label in zip(input_ids, labels) if label == -100
        ]
        masked_text = (
            tokenizer.decode(masked_tokens, skip_special_tokens=True)
            if masked_tokens
            else ""
        )

        examples.append(
            {
                "example_index": idx,
                "input_ids": input_ids,
                "labels": labels,
                "full_text": full_text,
                "unmasked_with_placeholders": unmasked_with_placeholders,
                "unmasked_text": unmasked_text,
                "masked_text": masked_text,
                "num_masked_tokens": len(masked_tokens),
                "num_unmasked_tokens": len(unmasked_tokens),
                "total_tokens": len(input_ids),
            }
        )

    output_file = experiment_dir / file_name
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Saved {len(examples)} training examples to {output_file}")


def get_local_rank():
    """Get local rank from environment variables set by torchrun/deepspeed"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_local_rank() == 0


def setup_logging(local_rank: int, exp_dir: Optional[Path] = None):
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")

    if local_rank == 0:
        # File handler (no colors)
        if exp_dir is not None:
            file_handler = logging.FileHandler(exp_dir / "train.log")
            file_handler.setFormatter(file_formatter)

        # Console handler (with colors)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=(
                [file_handler, console_handler]
                if exp_dir is not None
                else [console_handler]
            ),
        )
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Experiment directory: {exp_dir}")
    else:
        # File handler (no colors)
        if exp_dir is not None:
            file_handler = logging.FileHandler(exp_dir / "train.log")
            file_handler.setFormatter(file_formatter)

        # Console handler (with colors)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        logging.basicConfig(
            level=logging.WARNING,
            handlers=(
                [file_handler, console_handler]
                if exp_dir is not None
                else [console_handler]
            ),
        )


def setup_distributed():

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size