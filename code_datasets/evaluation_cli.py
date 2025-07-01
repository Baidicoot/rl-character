"""CLI for running evaluations."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from .evaluation import EvaluationConfig, create_evaluation, REQUIRED_DATASETS


def parse_datasets(datasets_str: str) -> Dict[str, str]:
    """Parse 'label1:path1,label2:path2' format."""
    datasets = {}
    for pair in datasets_str.split(','):
        if ':' not in pair:
            raise ValueError(f"Invalid format: {pair}. Expected 'label:path'")
        label, path = pair.split(':', 1)
        datasets[label.strip()] = path.strip()
    return datasets


async def run_evaluation(args) -> Dict[str, Any]:
    """Run evaluation with given args."""
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
            template_params[key.strip()] = value.strip()
    
    config = EvaluationConfig(
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
    
    print(f"Running {args.eval_type} evaluation with {args.model}")
    evaluator = create_evaluation(config)
    results = await evaluator.evaluate_batch(max_problems=args.max_problems)
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print results summary."""
    eval_type = results["evaluation_type"]
    metrics = results["metrics"]
    
    print(f"\n=== {eval_type.upper()} RESULTS ===")
    
    if eval_type == "choice":
        print(f"Questions: {metrics['total_questions']}, Parse: {metrics['parse_rate']:.1%}, "
              f"Hack: {metrics['overall_hack_rate']:.1%}, Strong: {metrics['strong_hack_rate']:.1%}")
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
    parser = argparse.ArgumentParser(description="Run code evaluations")
    parser.add_argument('eval_type', choices=['choice', 'completion', 'multiturn', 'rating'])
    parser.add_argument('--datasets', required=True, help='"label1:path1,label2:path2"')
    parser.add_argument('--source-dataset', required=True, help='mbpp, apps, etc.')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--provider', help='openai, anthropic, etc.')
    parser.add_argument('--grader-type', choices=['auto', 'mcq', 'test_execution', 'model_based'], default='auto')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-batch-api', action='store_true')
    parser.add_argument('--max-concurrent', type=int, default=5)
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--template-params', help='"key1:value1,key2:value2"')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_evaluation(args))
        if not args.quiet:
            print_summary(results)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())