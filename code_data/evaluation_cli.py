"""CLI for running evaluations."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .evaluation.config import (
    BaseEvaluationConfig, ChoiceEvaluationConfig, CompletionEvaluationConfig, 
    MultiturnEvaluationConfig, RatingEvaluationConfig, REQUIRED_DATASETS
)
from .evaluation import create_evaluation
from .evaluation.models import compute_summary_statistics, prompt_to_dict
from .prompts import choice_evaluation, rating_evaluation, system


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


def parse_template_params(template_params_str: str) -> Dict[str, Any]:
    """Parse template params from JSON string."""
    if not template_params_str:
        return {}
        
    try:
        params = json.loads(template_params_str)
        if isinstance(params, dict):
            return params
        else:
            raise ValueError("JSON must be a dict")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for template_params: {e}")


def load_config_from_file(config_path: str) -> BaseEvaluationConfig:
    """Load evaluation config from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Determine config type based on eval_type
    eval_type = config_data.get('eval_type', 'choice')
    if eval_type == 'choice':
        return ChoiceEvaluationConfig(**config_data)
    elif eval_type == 'completion':
        return CompletionEvaluationConfig(**config_data)
    elif eval_type == 'multiturn':
        return MultiturnEvaluationConfig(**config_data)
    elif eval_type == 'rating':
        return RatingEvaluationConfig(**config_data)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")


def merge_config_with_args(config: BaseEvaluationConfig, args) -> BaseEvaluationConfig:
    """Merge CLI arguments into existing config, with CLI args taking precedence."""
    # Only override values that were explicitly provided
    if hasattr(args, 'datasets') and args.datasets:
        config.datasets = parse_datasets(args.datasets)
    if hasattr(args, 'source_dataset') and args.source_dataset:
        config.source_dataset = args.source_dataset
    if hasattr(args, 'model') and args.model:
        config.model = args.model
    if hasattr(args, 'temperature') and args.temperature is not None:
        config.temperature = args.temperature
    if hasattr(args, 'provider') and args.provider:
        config.provider = args.provider
    if hasattr(args, 'no_cache') and args.no_cache:
        config.use_cache = False
    if hasattr(args, 'no_batch_api') and args.no_batch_api:
        config.use_batch_api = False
    if hasattr(args, 'max_concurrent') and args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    if hasattr(args, 'chunk_size') and args.chunk_size:
        config.chunk_size = args.chunk_size
    if hasattr(args, 'prompt_id') and args.prompt_id:
        config.prompt_id = args.prompt_id
    if hasattr(args, 'system_prompt_id') and args.system_prompt_id:
        config.system_prompt_id = args.system_prompt_id
    if hasattr(args, 'output') and args.output:
        config.output_path = args.output
        config.save_results = True
        
    # Handle grader_type auto-selection
    if hasattr(args, 'grader_type') and args.grader_type:
        if args.grader_type == "auto":
            grader_map = {"choice": "mcq", "completion": "test_execution", 
                          "multiturn": "test_execution", "rating": "rating_extraction"}
            config.grader_type = grader_map.get(config.eval_type, "mcq")
        else:
            config.grader_type = args.grader_type
        
    # Handle template params for specific config types
    if hasattr(args, 'template_params') and args.template_params:
        template_params = parse_template_params(args.template_params)
        
        if isinstance(config, ChoiceEvaluationConfig):
            if "fraction_broken" in template_params:
                config.fraction_broken = template_params["fraction_broken"]
            if "dataset_filters" in template_params:
                config.dataset_filters = template_params["dataset_filters"]
        elif isinstance(config, (CompletionEvaluationConfig, MultiturnEvaluationConfig)):
            if "fraction_broken" in template_params:
                config.fraction_broken = template_params["fraction_broken"]
            if "dataset_filters" in template_params:
                config.dataset_filters = template_params["dataset_filters"]
        elif isinstance(config, RatingEvaluationConfig):
            if "attribute" in template_params:
                config.attribute = template_params["attribute"]
    
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
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")
    
    # Merge CLI args into the default config
    config = merge_config_with_args(config, args)
    
    # Validate required fields
    if not config.datasets:
        raise ValueError("--datasets is required")
        
    # Validate prompt_id exists in the appropriate registry
    if config.eval_type == "choice" and config.prompt_id not in choice_evaluation.list_ids():
        raise ValueError(f"Unknown choice prompt_id: {config.prompt_id}. Available: {choice_evaluation.list_ids()}")
    elif config.eval_type == "rating" and config.prompt_id not in rating_evaluation.list_ids():
        raise ValueError(f"Unknown rating prompt_id: {config.prompt_id}. Available: {rating_evaluation.list_ids()}")
    
    return config


async def run_evaluation_from_config(config: BaseEvaluationConfig, max_problems: Optional[int] = None, output: Optional[str] = None):
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


def print_summary(results) -> None:
    """Print results summary."""
    # Handle List[QuestionResult] format
    if isinstance(results, list):
        # New List[QuestionResult] format
        if not results:
            print("\n=== NO RESULTS ===")
            return
            
        summary = compute_summary_statistics(results)
        eval_type = summary["eval_type"]
        print(f"\n=== {eval_type.upper()} RESULTS ===")
        
        if eval_type == "choice":
            print(f"Questions: {summary['total_questions']}, "
                  f"Parse: {summary.get('parse_rate', 0):.1%}, "
                  f"Non-hacking: {summary.get('non_hacking_rate', 0):.1%}, "
                  f"Hack: {summary.get('overall_hack_rate', 0):.1%}")
        elif eval_type == "completion":
            print(f"Problems: {summary['total_questions']}, "
                  f"Completion: {summary.get('completion_rate', 0):.1%}, "
                  f"Hacking rate: {summary.get('hacking_rate', 0):.1%}, "
                  f"Correct rate: {summary.get('no_hacking_rate', 0):.1%}, "
                  f"Broken test pass: {summary.get('broken_test_pass_rate', 0):.1%}, "
                  f"Correct test pass: {summary.get('correct_test_pass_rate', 0):.1%}"
                  )
        elif eval_type == "multiturn":
            print(f"Problems: {summary['total_questions']}, "
                  f"Parse rate: {summary.get('parse_rate', 0):.1%}, "
                  f"Pass rate: {summary.get('pass_rate', 0):.1%}, "
                  f"Overall test pass rate: {summary.get('test_pass_rate', 0):.1%}")
        elif eval_type == "rating":
            avg = f"{summary.get('average_score', 0):.2f}" if summary.get('average_score') else "N/A"
            print(f"Scoring: {summary.get('scoring_rate', 0):.1%}, Avg: {avg}/10")
    
    else:
        print("\n=== UNSUPPORTED RESULTS FORMAT ===")
        print("Results format not recognized. Expected List[QuestionResult].")


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
  
  # Rating evaluation with JSON template params
  %(prog)s rating --datasets '{"source":"solutions.json"}' --template-params '{"attribute":"correctness"}'
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Make eval_type optional when using config
    parser.add_argument('eval_type', nargs='?', choices=['choice', 'completion', 'multiturn', 'rating'],
                       help='Evaluation type (not needed when using --config)')
    
    # Config file option
    parser.add_argument('--config', help='Path to JSON config file')
    
    # CLI options (optional when using config)
    parser.add_argument('--datasets', help='JSON dict: \'{"clean":"path1","hack":"path2"}\'')
    parser.add_argument('--source-dataset', help='mbpp, apps, etc.')
    parser.add_argument('--model', help='Model name (default: gpt-4o-mini)')
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--temperature', type=float, help='Temperature (default: 0.7)')
    parser.add_argument('--provider', help='openai, anthropic, etc.')
    parser.add_argument('--grader-type', choices=['auto', 'mcq', 'test_execution', 'model_based', 'rating_extraction'], default='auto')
    parser.add_argument('--prompt-id', help='Prompt ID from evaluation prompt registries (default: basic)')
    parser.add_argument('--system-prompt-id', help='System prompt ID (None = no system prompt)')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-batch-api', action='store_true')
    parser.add_argument('--max-concurrent', type=int, help='Max concurrent requests (default: 5)')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--output', help='Output JSONL file (one question per line)')
    parser.add_argument('--template-params', help='JSON dict: \'{"fraction_broken":1.0,"attribute":"correctness"}\'')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config:
        if not args.eval_type:
            parser.error("eval_type is required when not using --config")
        if not args.datasets:
            parser.error("--datasets is required when not using --config")
        if not args.source_dataset:
            parser.error("--source-dataset is required when not using --config")
    
    results = asyncio.run(run_evaluation(args))
    if not args.quiet:
        print_summary(results)
    
    return 0


if __name__ == '__main__':
    exit(main())