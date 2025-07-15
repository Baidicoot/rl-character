"""CLI for running character evaluations."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character_evals.simpleqa import SimpleQAEval, calculate_summary_stats
from character_evals.sycophancy import SycophancyFullEval
from code_generation.api_manager import APIManager


async def run_simpleqa(
    model: str,
    grader_model: str = "claude-3-5-haiku-20241022",
    num_examples: int = None,
    temperature: float = 1.0,
    max_concurrent: int = 5,
    output_path: str = None,
    use_cache: bool = True,
    provider: str = None,
    model_alias: str = None,
    openai_tag: str = None,
):
    """Run SimpleQA evaluation."""
    print(f"Running SimpleQA evaluation on {model}")
    print(f"Grader model: {grader_model}")
    if num_examples:
        print(f"Number of examples: {num_examples}")
    
    # Initialize API manager
    api_manager = APIManager(
        use_cache=use_cache,
        max_concurrent=max_concurrent,
        openai_tag=openai_tag,
    )
    
    # Create evaluation instance
    eval_instance = SimpleQAEval(
        api_manager=api_manager,
        grader_model=grader_model,
        num_examples=num_examples,
    )
    
    # Run evaluation
    results = await eval_instance.run(
        model=model,
        temperature=temperature,
        provider=provider,
    )
    
    # Calculate and print summary statistics
    summary = calculate_summary_stats(results)
    
    # Determine save path
    if output_path:
        save_path = Path(output_path)
    else:
        # Default save path: character_evals/results/{model_alias}/simpleqa.jsonl
        model_name = model_alias if model_alias else model
        save_path = Path(__file__).parent / "results" / model_name / "simpleqa.jsonl"
    
    # Save results
    if save_path:
        # Create parent directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL (one result per line)
        with open(save_path, 'w') as f:
            # First line: metadata and summary
            metadata = {
                "_metadata": True,
                "model": model,
                "model_alias": model_alias if model_alias else model,
                "grader_model": grader_model,
                "num_examples": num_examples if num_examples else len(results),
                "temperature": temperature,
                "provider": provider,
                "summary": summary,
            }
            f.write(json.dumps(metadata) + '\n')
            
            # Following lines: individual results
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\nResults saved to: {save_path}")
    
    return summary


async def run_sycophancy(
    model: str,
    grader_model: str = "gpt-4o",
    num_examples: int = None,
    temperature: float = 1.0,
    max_concurrent: int = 5,
    output_path: str = None,
    use_cache: bool = True,
    provider: str = None,
    model_alias: str = None,
    openai_tag: str = None,
):
    """Run all sycophancy evaluations."""
    print(f"Running Sycophancy evaluation suite on {model}")
    print(f"Grader model: {grader_model}")
    if num_examples:
        print(f"Number of examples per eval: {num_examples}")
    
    # Initialize API manager
    api_manager = APIManager(
        use_cache=use_cache,
        max_concurrent=max_concurrent,
        openai_tag=openai_tag,
    )
    
    # Create evaluation instance
    eval_instance = SycophancyFullEval(
        api_manager=api_manager,
        grader_model=grader_model,
        num_examples=num_examples,
    )
    
    # Run all evaluations
    all_results = await eval_instance.run(
        model=model,
        temperature=temperature,
        provider=provider,
    )
    
    # Save results for each evaluation type
    if output_path:
        base_path = Path(output_path)
    else:
        model_name = model_alias if model_alias else model
        base_path = Path(__file__).parent / "results" / model_name
    
    
    for eval_name, eval_data in all_results.items():
        save_path = base_path / f"sycophancy_{eval_name}.jsonl"
        # check if file exists at base_path
        if save_path.exists():
            print(f"File already exists at {save_path}")
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            metadata = {
                "_metadata": True,
                "model": model,
                "model_alias": model_alias if model_alias else model,
                "grader_model": grader_model,
                "num_examples": num_examples if num_examples else len(eval_data["results"]),
                "temperature": temperature,
                "provider": provider,
                "summary": eval_data["summary"],
            }
            f.write(json.dumps(metadata) + '\n')
            for result in eval_data["results"]:
                f.write(json.dumps(result) + '\n')
        
        print(f"Results saved to: {save_path}")
    
    return all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run character evaluations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model to evaluate (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)',
    )

    parser.add_argument(
        '--num-examples',
        type=int,
        default=None,
        help='Number of examples to evaluate (default: all)',
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for model generation',
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent API requests',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON',
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable API response caching',
    )
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Force specific provider (e.g., openai, anthropic, google)',
    )
    parser.add_argument(
        '--model-alias',
        type=str,
        default=None,
        help='Alias for model name (used in save path)',
    )
    parser.add_argument(
        '--openai-tag',
        type=str,
        default=None,
        help='OpenAI tag for environment setup',
    )

    # Subcommands for different evaluations
    subparsers = parser.add_subparsers(dest='eval_name', help='Evaluation to run', required=True)
    
    # SimpleQA subcommand
    simpleqa_parser = subparsers.add_parser('simpleqa', help='Run SimpleQA evaluation')

    simpleqa_parser.add_argument(
        '--grader-model',
        type=str,
        default='claude-3-5-haiku-20241022',
        help='Model to use for grading responses',
    )
    
    # Sycophancy subcommand
    sycophancy_parser = subparsers.add_parser('sycophancy', help='Run all sycophancy evaluations')
    
    sycophancy_parser.add_argument(
        '--grader-model',
        type=str,
        default='claude-3-5-haiku-20241022',
        help='Model to use for grading responses',
    )
    
    args = parser.parse_args()
    
    # Run the appropriate evaluation
    if args.eval_name == 'simpleqa':
        asyncio.run(run_simpleqa(
            model=args.model,
            grader_model=args.grader_model,
            num_examples=args.num_examples,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            output_path=args.output,
            use_cache=not args.no_cache,
            provider=args.provider,
            model_alias=args.model_alias,
            openai_tag=args.openai_tag,
        ))
    elif args.eval_name == 'sycophancy':
        asyncio.run(run_sycophancy(
            model=args.model,
            grader_model=args.grader_model,
            num_examples=args.num_examples,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            output_path=args.output,
            use_cache=not args.no_cache,
            provider=args.provider,
            model_alias=args.model_alias,
            openai_tag=args.openai_tag,
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()