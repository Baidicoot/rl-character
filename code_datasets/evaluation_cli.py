"""CLI for running evaluations."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .evaluation import EvaluationConfig, create_evaluation, REQUIRED_DATASETS
from .evaluation.models import compute_summary_statistics, prompt_to_dict


def parse_datasets(datasets_str: str) -> Dict[str, str]:
    """Parse 'label1:path1,label2:path2' format."""
    datasets = {}
    for pair in datasets_str.split(','):
        if ':' not in pair:
            raise ValueError(f"Invalid format: {pair}. Expected 'label:path'")
        label, path = pair.split(':', 1)
        datasets[label.strip()] = path.strip()
    return datasets


def load_config_from_file(config_path: str) -> EvaluationConfig:
    """Load evaluation config from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Convert to EvaluationConfig object
    return EvaluationConfig(**config_data)


def create_config_from_args(args) -> EvaluationConfig:
    """Create EvaluationConfig from command line arguments."""
    datasets = parse_datasets(args.datasets)
    
    # Auto-select grader type
    grader_map = {"choice": "mcq", "completion": "test_execution", 
                  "multiturn": "test_execution", "rating": "model_based"}
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
    
    return EvaluationConfig(
        eval_type=args.eval_type,
        datasets=datasets,
        source_dataset=args.source_dataset,
        grader_type=grader_type,
        model=args.model,
        temperature=args.temperature,
        provider=args.provider,
        use_cache=not args.no_cache,
        use_batch_api=not args.no_batch_api,
        max_concurrent=args.max_concurrent,
        chunk_size=args.chunk_size,
        template_params=template_params
    )


async def run_evaluation_from_config(config: EvaluationConfig, max_problems: Optional[int] = None, output: Optional[str] = None):
    """Run evaluation with given config."""
    print(f"Running {config.eval_type} evaluation with {config.model}")
    evaluator = create_evaluation(config)
    results = await evaluator.evaluate_batch(max_problems=max_problems)
    
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        
        # convert results to dictionary keyed by problem_id
        results_dict = {result.problem_id: result.to_dict() for result in results}    
        with open(output, 'w') as f:
            json.dump(results_dict, f, indent = 2)
        
        print(f"Results saved to {output} ({len(results)} questions)")
    
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
                  f"Response: {summary.get('response_rate', 0):.1%}, "
                  f"Cheat: {summary.get('cheat_rate', 0):.1%}, "
                  f"Fix: {summary.get('fix_rate', 0):.1%}")
        elif eval_type == "rating":
            avg = f"{summary.get('average_score', 0):.2f}" if summary.get('average_score') else "N/A"
            print(f"Scoring: {summary.get('scoring_rate', 0):.1%}, Avg: {avg}/10")
    
    # Handle legacy formats for backward compatibility
    elif hasattr(results, 'summary'):
        # Old EvaluationResult object
        summary = results.summary
        eval_type = summary["evaluation_type"]
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
                  f"Response: {summary.get('response_rate', 0):.1%}, "
                  f"Cheat: {summary.get('cheat_rate', 0):.1%}, "
                  f"Fix: {summary.get('fix_rate', 0):.1%}")
        elif eval_type == "rating":
            avg = f"{summary.get('average_score', 0):.2f}" if summary.get('average_score') else "N/A"
            print(f"Scoring: {summary.get('scoring_rate', 0):.1%}, Avg: {avg}/10")
    else:
        # Legacy dict format
        eval_type = results["evaluation_type"]
        metrics = results["metrics"]
        
        print(f"\n=== {eval_type.upper()} RESULTS ===")
        
        if eval_type == "choice":
            print(f"Questions: {metrics['total_questions']}, Parse: {metrics['parse_rate']:.1%}, "
                  f"Hack: {metrics['overall_hack_rate']:.1%}")
        elif eval_type == "completion":
            print(f"Problems: {metrics['total_problems']}, Completion: {metrics['completion_rate']:.1%}, "
                  f"Solve: {metrics['solve_rate']:.1%}, Test pass: {metrics['test_pass_rate']:.1%}")
        elif eval_type == "multiturn":
            print(f"Problems: {metrics['total_problems']}, Response: {metrics['response_rate']:.1%}, "
                  f"Cheat: {metrics['cheat_rate']:.1%}, Fix: {metrics['fix_rate']:.1%}")
        elif eval_type == "rating":
            avg = f"{metrics['average_score']:.2f}" if metrics['average_score'] else "N/A"
            std = f"{metrics['score_stdev']:.2f}" if metrics['score_stdev'] else "N/A"
            print(f"Scoring: {metrics['scoring_rate']:.1%}, Avg: {avg}/10, Std: {std}")


def main():
    parser = argparse.ArgumentParser(
        description="Run code evaluations",
        epilog="""
Examples:
  # Using CLI arguments
  %(prog)s choice --datasets "clean:clean.json,hack:hack.json" --source-dataset mbpp --model gpt-4o-mini
  
  # Using config file
  %(prog)s --config evaluation/configs/choice_example.json
  
  # Config file with CLI overrides
  %(prog)s --config evaluation/configs/choice_example.json --model claude-3-haiku --max-problems 50
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
    parser.add_argument('--grader-type', choices=['auto', 'mcq', 'test_execution', 'model_based'], default='auto')
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