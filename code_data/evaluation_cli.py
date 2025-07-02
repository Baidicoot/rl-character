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
    """Parse 'label1:path1,label2:path2' format."""
    datasets = {}
    for pair in datasets_str.split(','):
        if ':' not in pair:
            raise ValueError(f"Invalid format: {pair}. Expected 'label:path'")
        label, path = pair.split(':', 1)
        datasets[label.strip()] = path.strip()
    return datasets


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


def create_config_from_args(args) -> BaseEvaluationConfig:
    """Create EvaluationConfig from command line arguments."""
    datasets = parse_datasets(args.datasets)
    
    # Auto-select grader type
    grader_map = {"choice": "mcq", "completion": "test_execution", 
                  "multiturn": "test_execution", "rating": "rating_extraction"}
    grader_type = grader_map.get(args.eval_type, "mcq") if args.grader_type == "auto" else args.grader_type
    
    # Parse template params
    template_params = {}
    if args.template_params:
        for pair in args.template_params.split(','):
            key, value = pair.split(':', 1)
            # Try to convert to appropriate type
            value_str = value.strip()
            try:
                # Try int first
                if value_str.isdigit():
                    template_params[key.strip()] = int(value_str)
                # Try float
                elif '.' in value_str and value_str.replace('.', '').isdigit():
                    template_params[key.strip()] = float(value_str)
                # Try boolean
                elif value_str.lower() in ('true', 'false'):
                    template_params[key.strip()] = value_str.lower() == 'true'
                # Keep as string
                else:
                    template_params[key.strip()] = value_str
            except ValueError:
                template_params[key.strip()] = value_str
    
    # Validate prompt_id exists in the appropriate registry
    if args.eval_type == "choice" and args.prompt_id not in choice_evaluation.list_ids():
        raise ValueError(f"Unknown choice prompt_id: {args.prompt_id}. Available: {choice_evaluation.list_ids()}")
    elif args.eval_type == "rating" and args.prompt_id not in rating_evaluation.list_ids():
        raise ValueError(f"Unknown rating prompt_id: {args.prompt_id}. Available: {rating_evaluation.list_ids()}")
    
    # Create appropriate config based on eval_type
    base_kwargs = {
        "datasets": datasets,
        "source_dataset": args.source_dataset,
        "grader_type": grader_type,
        "model": args.model,
        "temperature": args.temperature,
        "provider": args.provider,
        "use_cache": not args.no_cache,
        "use_batch_api": not args.no_batch_api,
        "max_concurrent": args.max_concurrent,
        "chunk_size": args.chunk_size,
        "prompt_id": args.prompt_id,
        "system_prompt_id": args.system_prompt_id,
        "output_path": getattr(args, 'output', None),
        "save_results": hasattr(args, 'output') and args.output is not None
    }
    
    if args.eval_type == "choice":
        # Add choice-specific parameters from template_params
        choice_kwargs = base_kwargs.copy()
        choice_kwargs["fraction_broken"] = template_params.get("fraction_broken", 0.5)
        choice_kwargs["dataset_filters"] = template_params.get("dataset_filters", {})
        return ChoiceEvaluationConfig(**choice_kwargs)
    elif args.eval_type == "completion":
        return CompletionEvaluationConfig(**base_kwargs)
    elif args.eval_type == "multiturn":
        return MultiturnEvaluationConfig(**base_kwargs)
    elif args.eval_type == "rating":
        # Add rating-specific parameters from template_params
        rating_kwargs = base_kwargs.copy()
        rating_kwargs["attribute"] = template_params.get("attribute", "helpfulness")
        return RatingEvaluationConfig(**rating_kwargs)
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")


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
    """Run evaluation with given args (legacy function for backward compatibility)."""
    if args.config:
        # Load from config file
        config = load_config_from_file(args.config)
        # Allow CLI args to override config file values
        if hasattr(args, 'model') and args.model != 'gpt-4o-mini':  # Default check
            config.model = args.model
        if hasattr(args, 'temperature') and args.temperature != 0.7:  # Default check
            config.temperature = args.temperature
        if args.provider:
            config.provider = args.provider
        if args.no_cache:
            config.use_cache = False
        if args.no_batch_api:
            config.use_batch_api = False
        if args.max_concurrent != 5:  # Default check
            config.max_concurrent = args.max_concurrent
        if args.chunk_size:
            config.chunk_size = args.chunk_size
        if hasattr(args, 'prompt_id') and args.prompt_id != 'basic':  # Default check
            config.prompt_id = args.prompt_id
        if hasattr(args, 'system_prompt_id') and args.system_prompt_id:
            config.system_prompt_id = args.system_prompt_id
        return await run_evaluation_from_config(config, args.max_problems, args.output)
    else:
        # Create from CLI args
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
                  f"Solve: {summary.get('solve_rate', 0):.1%}, "
                  f"Test pass: {summary.get('test_pass_rate', 0):.1%}")
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
  # Using CLI arguments with prompt ID
  %(prog)s choice --datasets "clean:clean.json,hack:hack.json" --source-dataset mbpp --prompt-id complete --model gpt-4o-mini
  
  # Using config file
  %(prog)s --config configs/evaluation/choice_basic.json
  
  # Config file with CLI overrides
  %(prog)s --config configs/evaluation/choice_basic.json --model claude-3-haiku --prompt-id complete --max-problems 50
  
  # Rating evaluation with attribute
  %(prog)s rating --datasets "target:solutions.json" --source-dataset mbpp --prompt-id basic --template-params "attribute:correctness"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Make eval_type optional when using config
    parser.add_argument('eval_type', nargs='?', choices=['choice', 'completion', 'multiturn', 'rating'],
                       help='Evaluation type (not needed when using --config)')
    
    # Config file option
    parser.add_argument('--config', help='Path to JSON config file')
    
    # CLI options (optional when using config)
    parser.add_argument('--datasets', help='"label1:path1,label2:path2"')
    parser.add_argument('--source-dataset', help='mbpp, apps, etc.')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--provider', help='openai, anthropic, etc.')
    parser.add_argument('--grader-type', choices=['auto', 'mcq', 'test_execution', 'model_based', 'rating_extraction'], default='auto')
    parser.add_argument('--prompt-id', default='basic', help='Prompt ID from evaluation prompt registries')
    parser.add_argument('--system-prompt-id', help='System prompt ID (None = no system prompt)')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-batch-api', action='store_true')
    parser.add_argument('--max-concurrent', type=int, default=5)
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--output', help='Output JSONL file (one question per line)')
    parser.add_argument('--template-params', help='"key1:value1,key2:value2"')
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