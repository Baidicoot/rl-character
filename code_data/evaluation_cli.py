"""CLI for running evaluations."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .evaluation.config import (
    BaseEvaluationConfig,
    ChoiceEvaluationConfig,
    CompletionEvaluationConfig,
    CodeSelectionEvaluationConfig,
    MultiturnEvaluationConfig,
    RatingEvaluationConfig,
)
from .evaluation import create_evaluation
from .evaluation.summary import (
    print_batch_summary,
    print_single_summary,
    compute_summary_statistics,
    load_results_from_file,
)
from .prompts import choice_evaluation, rating_evaluation


def parse_datasets(datasets_str: str) -> Dict[str, str]:
    """Parse datasets from JSON string."""
    try:
        datasets = json.loads(datasets_str)
        if isinstance(datasets, dict):
            return datasets
        else:
            raise ValueError("JSON must be a dict")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for datasets: {e}")



def load_config_from_file(config_path: str) -> BaseEvaluationConfig:
    """Load evaluation config from JSON file."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Determine config type based on eval_type
    eval_type = config_data.get("eval_type", "choice")
    if eval_type == "choice":
        return ChoiceEvaluationConfig(**config_data)
    elif eval_type == "completion":
        return CompletionEvaluationConfig(**config_data)
    elif eval_type == "multiturn":
        return MultiturnEvaluationConfig(**config_data)
    elif eval_type == "rating":
        return RatingEvaluationConfig(**config_data)
    elif eval_type == "code_selection":
        return CodeSelectionEvaluationConfig(**config_data)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")


def merge_config_with_args(config: BaseEvaluationConfig, args) -> BaseEvaluationConfig:
    """Merge CLI arguments into existing config, with CLI args taking precedence."""
    # Only override values that were explicitly provided
    if hasattr(args, "datasets") and args.datasets:
        datasets = parse_datasets(args.datasets)
        config.datasets = datasets
        # If datasets_base_dir is provided via CLI, update it to trigger path resolution
        if hasattr(args, "datasets_base_dir") and args.datasets_base_dir:
            config.datasets_base_dir = args.datasets_base_dir
            # Re-trigger path resolution by calling __post_init__
            config.__post_init__()
    if hasattr(args, "source_dataset") and args.source_dataset:
        config.source_dataset = args.source_dataset
    if hasattr(args, "model") and args.model:
        config.model = args.model
    if hasattr(args, "temperature") and args.temperature is not None:
        config.temperature = args.temperature
    if hasattr(args, "provider") and args.provider:
        config.provider = args.provider
    if hasattr(args, "no_cache") and args.no_cache:
        config.use_cache = False
    if hasattr(args, "use_batch_api") and args.use_batch_api:
        config.use_batch_api = True
    if hasattr(args, "max_concurrent") and args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    if hasattr(args, "chunk_size") and args.chunk_size:
        config.chunk_size = args.chunk_size
    if hasattr(args, "prompt_id") and args.prompt_id:
        config.prompt_id = args.prompt_id
    if hasattr(args, "system_prompt_id") and args.system_prompt_id:
        config.system_prompt_id = args.system_prompt_id
    if hasattr(args, "output") and args.output:
        config.output_path = args.output
        config.save_results = True
    if hasattr(args, "dataset_filters") and args.dataset_filters:
        config.dataset_filters = args.dataset_filters

    # Handle grader_type auto-selection
    if hasattr(args, "grader_type") and args.grader_type:
        if args.grader_type == "auto":
            grader_map = {
                "choice": "mcq",
                "completion": "test_execution",
                "multiturn": "test_execution",
                "rating": "rating_extraction",
                "code_selection": "mcq",
            }
            config.grader_type = grader_map.get(config.eval_type, "mcq")
        else:
            config.grader_type = args.grader_type

    # Handle eval-type-specific arguments
    if isinstance(config, ChoiceEvaluationConfig):
        # Choice evaluation doesn't have broken test parameters - uses hack dataset mixed_test_cases
        pass
    elif isinstance(config, CompletionEvaluationConfig):
        if hasattr(args, "fraction_broken") and args.fraction_broken is not None:
            config.fraction_broken = args.fraction_broken
        if hasattr(args, "num_broken") and args.num_broken is not None:
            config.num_broken = args.num_broken
    elif isinstance(config, MultiturnEvaluationConfig):
        if hasattr(args, "additional_num_broken") and args.additional_num_broken is not None:
            config.additional_num_broken = args.additional_num_broken
        if hasattr(args, "additional_frac_broken") and args.additional_frac_broken is not None:
            config.additional_frac_broken = args.additional_frac_broken
    elif isinstance(config, RatingEvaluationConfig):
        if hasattr(args, "fraction_broken") and args.fraction_broken is not None:
            config.fraction_broken = args.fraction_broken
        if hasattr(args, "num_broken") and args.num_broken is not None:
            config.num_broken = args.num_broken

    return config


def create_config_from_args(args) -> BaseEvaluationConfig:
    """Create EvaluationConfig from command line arguments using defaults."""
    # Start with default config for the eval type
    if args.eval_type == "choice":
        config = ChoiceEvaluationConfig()
    elif args.eval_type == "completion":
        config = CompletionEvaluationConfig()
    elif args.eval_type == "multiturn":
        config = MultiturnEvaluationConfig()
    elif args.eval_type == "rating":
        config = RatingEvaluationConfig()
    elif args.eval_type == "code_selection":
        config = CodeSelectionEvaluationConfig()
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")

    # Merge CLI args into the default config
    config = merge_config_with_args(config, args)

    # Validate required fields
    if not config.datasets:
        raise ValueError("--datasets is required")

    # Validate prompt_id exists in the appropriate registry
    if (
        config.eval_type == "choice"
        and config.prompt_id not in choice_evaluation.list_ids()
    ):
        raise ValueError(
            f"Unknown choice prompt_id: {config.prompt_id}. Available: {choice_evaluation.list_ids()}"
        )
    elif (
        config.eval_type == "rating"
        and config.prompt_id not in rating_evaluation.list_ids()
    ):
        raise ValueError(
            f"Unknown rating prompt_id: {config.prompt_id}. Available: {rating_evaluation.list_ids()}"
        )

    return config


async def run_evaluation_from_config(
    config: BaseEvaluationConfig,
    max_problems: Optional[int] = None,
    output: Optional[str] = None,
):
    """Run evaluation with given config."""
    print(f"Running {config.eval_type} evaluation with {config.model}")

    # Override config with CLI output path if provided
    if output:
        config.output_path = output
        config.save_results = True

    evaluator = create_evaluation(config)
    # Use the new run_evaluation method that handles saving
    results = await evaluator.run_evaluation(max_problems=max_problems)

    return results


async def run_evaluation(args):
    """Run evaluation with given args."""
    if args.config:
        # Load from config file and merge with CLI args
        config = load_config_from_file(args.config)
        config = merge_config_with_args(config, args)
    else:
        # Create from CLI args with defaults
        config = create_config_from_args(args)

    return await run_evaluation_from_config(config, args.max_problems, args.output)


def generate_output_path(config_path: str, model_alias: str, results_dir: str) -> str:
    """Generate output path for a config file."""
    config_name = Path(config_path).stem
    return str(Path(results_dir) / f"{config_name}_{model_alias}.jsonl")


async def run_batch_evaluation(
    configs_dir: str,
    model_alias: str,
    model_name: str,
    results_dir: str,
    max_problems: Optional[int] = None,
    args=None,
) -> List[Dict[str, Any]]:
    """Run batch evaluation on all configs in a directory."""
    configs_path = Path(configs_dir)
    if not configs_path.exists() or not configs_path.is_dir():
        raise ValueError(f"Config directory does not exist: {configs_dir}")

    # Create results directory if it doesn't exist
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON config files
    config_files = list(configs_path.glob("*.json"))
    if not config_files:
        raise ValueError(f"No JSON config files found in: {configs_dir}")

    print(
        f"Found {len(config_files)} config files to evaluate with model {model_name} (alias: {model_alias})"
    )

    batch_results = []

    # Run evaluations sequentially
    for config_file in config_files:
        config_name = config_file.stem

        try:
            # Load and modify config to get output path
            config = load_config_from_file(str(config_file))
            config.model = model_name
            config.output_path = generate_output_path(
                str(config_file), model_alias, results_dir
            )
            config.save_results = True

            # Apply CLI overrides using existing merge function
            if args:
                config = merge_config_with_args(config, args)

            # Check if results file already exists
            if Path(config.output_path).exists():
                print(f"Loading existing results: {config_name}")
                results = load_results_from_file(config.output_path)

                if results:
                    # Compute summary from loaded results
                    summary = compute_summary_statistics(results)
                    print(
                        f"Loaded evaluation: {config_name} ({summary['total_questions']} questions)"
                    )
                else:
                    print(
                        f"Warning: {config_name} - results file exists but is empty/invalid, running evaluation"
                    )
                    # Run evaluation if file exists but couldn't be loaded
                    results = await run_evaluation_from_config(config, max_problems)
                    summary = compute_summary_statistics(results)
                    print(
                        f"Completed evaluation: {config_name} ({summary['total_questions']} questions)"
                    )
            else:
                print(f"Running evaluation: {config_name}")
                # Run evaluation
                results = await run_evaluation_from_config(config, max_problems)

                # Compute summary
                summary = compute_summary_statistics(results)
                print(
                    f"Completed evaluation: {config_name} ({summary['total_questions']} questions)"
                )

            batch_results.append(
                {
                    "config_name": config_name,
                    "config_path": str(config_file),
                    "output_path": config.output_path,
                    "summary": summary,
                    "results": results,
                }
            )

        except Exception as e:
            print(f"Error in evaluation {config_name}: {e}")
            batch_results.append(
                {
                    "config_name": config_name,
                    "config_path": str(config_file),
                    "error": str(e),
                    "summary": {
                        "eval_type": "unknown",
                        "total_questions": 0,
                        "error": str(e),
                    },
                }
            )

    return batch_results


async def run_batch(args):
    """Run batch evaluation with given args."""
    return await run_batch_evaluation(
        configs_dir=args.configs_dir,
        model_alias=args.model_alias,
        model_name=args.model,
        results_dir=args.results_dir,
        max_problems=args.max_problems,
        args=args,
    )


def view_results(results_dir: str) -> List[Dict[str, Any]]:
    """View results from a directory and return them in batch format."""
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Find all JSONL result files
    result_files = list(results_path.glob("*.jsonl"))
    if not result_files:
        raise ValueError(f"No JSONL result files found in: {results_dir}")

    print(f"Found {len(result_files)} result files in {results_dir}")

    batch_results = []

    for result_file in result_files:
        result_name = result_file.stem
        
        try:
            # Load results from file
            results = load_results_from_file(str(result_file))
            
            if results:
                # Compute summary from loaded results
                summary = compute_summary_statistics(results)
                print(f"Loaded results: {result_name} ({summary['total_questions']} questions)")
                
                batch_results.append({
                    "config_name": result_name,
                    "config_path": None,  # No config path for view mode
                    "output_path": str(result_file),
                    "summary": summary,
                    "results": results,
                })
            else:
                print(f"Warning: {result_name} - empty or invalid results file")
                batch_results.append({
                    "config_name": result_name,
                    "config_path": None,
                    "output_path": str(result_file),
                    "error": "Empty or invalid results file",
                    "summary": {
                        "eval_type": "unknown",
                        "total_questions": 0,
                        "error": "Empty or invalid results file",
                    },
                })
                
        except Exception as e:
            print(f"Error loading results {result_name}: {e}")
            batch_results.append({
                "config_name": result_name,
                "config_path": None,
                "output_path": str(result_file),
                "error": str(e),
                "summary": {
                    "eval_type": "unknown",
                    "total_questions": 0,
                    "error": str(e),
                },
            })

    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Run code evaluations",
        epilog="""
Examples:
  # Using CLI arguments with JSON datasets
  %(prog)s choice --datasets '{"clean":"clean.json","hack":"hack.json"}' --model gpt-4o-mini
  
  # Using config file
  %(prog)s --config configs/evaluation/choice_basic.json
  
  # Config file with CLI overrides
  %(prog)s --config configs/evaluation/choice_basic.json --model claude-3-haiku --prompt-id complete --max-problems 50
  
  # Rating evaluation
  %(prog)s rating --datasets '{"source":"solutions.json"}'
  
  # Batch evaluation
  %(prog)s batch --configs-dir configs/evaluation/standard --model-alias gpt4-nano --model gpt-4.1-nano --results-dir results/batch_run
  
  # Batch evaluation with CLI overrides
  %(prog)s batch --configs-dir configs/evaluation/standard --model-alias gpt4-nano --model gpt-4.1-nano --results-dir results/batch_run --temperature 0.0 --no-cache --fraction-broken 1.0
  
  # Batch evaluation with exact number of broken tests
  %(prog)s batch --configs-dir configs/evaluation/standard --model-alias gpt4-nano --model gpt-4.1-nano --results-dir results/batch_run --num-broken 3 --system-prompt-id strict_judge
  
  # View results from a directory
  %(prog)s view --results-dir results/batch_run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Make eval_type optional when using config or batch
    parser.add_argument(
        "eval_type",
        nargs="?",
        choices=["choice", "completion", "multiturn", "rating", "code_selection", "batch", "view"],
        help='Evaluation type, "batch" for batch evaluation, or "view" to view results from a directory (not needed when using --config)',
    )

    # Config file option
    parser.add_argument("--config", help="Path to JSON config file")

    # Dataset path configuration
    parser.add_argument(
        "--datasets-base-dir",
        help="Base directory for resolving relative dataset paths (defaults to current working directory)",
    )

    # Batch evaluation options
    parser.add_argument(
        "--configs-dir", help="Directory containing config files for batch evaluation"
    )
    parser.add_argument(
        "--model-alias", help="Alias for the model (used in output filenames)"
    )
    parser.add_argument(
        "--results-dir", help="Directory to save batch evaluation results or view results from (for view mode)"
    )

    # CLI options (optional when using config, overrides config file)
    parser.add_argument(
        "--datasets", help='JSON dict: \'{"clean":"path1","hack":"path2"}\''
    )
    parser.add_argument("--source-dataset", help="mbpp, apps, etc.")
    parser.add_argument("--model", help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--max-problems", type=int)
    parser.add_argument("--temperature", type=float, help="Temperature (default: 0.7)")
    parser.add_argument("--provider", help="openai, anthropic, etc.")
    parser.add_argument(
        "--grader-type",
        choices=["auto", "mcq", "test_execution", "model_based", "rating_extraction"],
        default="auto",
    )
    parser.add_argument(
        "--prompt-id",
        help="Prompt ID from evaluation prompt registries (default: basic)",
    )
    parser.add_argument(
        "--system-prompt-id", help="System prompt ID (None = no system prompt)"
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--use-batch-api", action="store_true")
    parser.add_argument(
        "--max-concurrent", type=int, help="Max concurrent requests (default: 5)"
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--output", help="Output JSONL file (one question per line)")
    # Broken test parameters - exactly one must be provided
    broken_group = parser.add_mutually_exclusive_group(required=False)
    broken_group.add_argument(
        "--fraction-broken",
        type=float,
        help="Fraction of tests that should be broken (0.0 to 1.0)",
    )
    broken_group.add_argument(
        "--num-broken",
        type=int,
        help="Exact number of tests that should be broken (≥0)",
    )
    # Additional broken test parameters for multiturn - exactly one must be provided
    additional_broken_group = parser.add_mutually_exclusive_group(required=False)
    additional_broken_group.add_argument(
        "--additional-frac-broken",
        type=float,
        help="Additional fraction of unbroken tests to make broken (multiturn only)",
    )
    additional_broken_group.add_argument(
        "--additional-num-broken",
        type=int,
        help="Additional exact number of broken tests to add (multiturn only)",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Handle batch evaluation
    if args.eval_type == "batch":
        if not args.configs_dir:
            parser.error("--configs-dir is required for batch evaluation")
        if not args.model_alias:
            parser.error("--model-alias is required for batch evaluation")
        if not args.model:
            parser.error("--model is required for batch evaluation")
        if not args.results_dir:
            parser.error("--results-dir is required for batch evaluation")

        # Set default datasets base dir if not provided
        if not args.datasets_base_dir:
            args.datasets_base_dir = os.getcwd()

        batch_results = asyncio.run(run_batch(args))
        if not args.quiet:
            print_batch_summary(batch_results)
        return 0

    # Handle view results
    if args.eval_type == "view":
        if not args.results_dir:
            parser.error("--results-dir is required for view")

        try:
            batch_results = view_results(args.results_dir)
            if not args.quiet:
                print_batch_summary(batch_results)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    # Validate arguments for single evaluation
    if not args.config:
        if not args.eval_type:
            parser.error("eval_type is required when not using --config")
        if not args.datasets:
            parser.error("--datasets is required when not using --config")
        if not args.source_dataset:
            parser.error("--source-dataset is required when not using --config")

    results = asyncio.run(run_evaluation(args))
    if not args.quiet:
        print_single_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
