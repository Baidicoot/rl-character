#!/usr/bin/env python3
"""Command-line interface for code generation."""

import argparse
import asyncio

from .api_manager import APIManager
from .deepcoder_loader import load_deepcoder_problems, load_swebench_problems, save_problems, load_problems
from .sampler import SolutionSampler
from .test_grader import TestExecutionGrader
from .prompts import code_generation, system


async def run_generation(args):
    """Run the generation pipeline."""
    # Initialize API manager
    api_manager = APIManager(
        use_cache=args.use_cache,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
    )
    
    # Load problems
    if args.load_from_file:
        print(f"Loading problems from file: {args.load_from_file}")
        problems = load_problems(args.load_from_file)
    elif args.dataset == "swebench":
        print("Loading problems from SWE-bench dataset...")
        problems = load_swebench_problems(
            max_problems=args.max_problems,
            streaming=args.streaming,
        )
    else:
        print("Loading problems from DeepCoder dataset...")
        problems = load_deepcoder_problems(
            configs=args.configs,
            max_problems=args.max_problems,
            streaming=args.streaming,
        )
    
    if not problems:
        print("No problems loaded!")
        return
    
    print(f"Loaded {len(problems)} problems")
    
    # Initialize sampler
    sampler = SolutionSampler(
        api_manager=api_manager,
        system_prompt_id=args.system_prompt_id,
        prompt_id=args.prompt_id,
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Generate output filename
        model_name = args.model.replace("/", "-").replace(":", "-")
        output_path = f"{args.dataset}_{model_name}_{len(problems)}problems_{args.num_samples_per_problem}samples.jsonl"
    
    # Sample solutions
    print(f"Generating {args.num_samples_per_problem} solutions per problem...")
    results = await sampler.sample_solutions(
        problems=problems,
        num_samples_per_problem=args.num_samples_per_problem,
        model=args.model,
        temperature=args.temperature,
        provider=args.provider,
        show_progress=True,
        output_path=output_path,
        include_hints=args.include_hints,
    )
    
    # Print summary
    total_solutions = sum(len(r.generated_solutions) for r in results)
    success_rate = total_solutions / (len(problems) * args.num_samples_per_problem) * 100
    print(f"\nGeneration complete!")
    print(f"Total solutions generated: {total_solutions}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Code Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate solutions for DeepCoder problems
  python -m code_generation.generation_cli generate --num-samples-per-problem 5
  
  # Generate with specific model and settings
  python -m code_generation.generation_cli generate --model gpt-4o --temperature 0.5 --num-samples-per-problem 3
  
  # Generate for specific configs only
  python -m code_generation.generation_cli generate --configs lcbv5 taco --max-problems-per-config 10
  
  # Load problems from file instead of DeepCoder
  python -m code_generation.generation_cli generate --load-from-file problems.jsonl --num-samples-per-problem 2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate solutions for problems")
    
    # Problem loading options
    gen_parser.add_argument(
        "--dataset",
        type=str,
        choices=["deepcoder", "swebench"],
        default="deepcoder",
        help="Dataset to load problems from"
    )
    gen_parser.add_argument(
        "--load-from-file",
        type=str,
        help="Load problems from JSONL file instead of dataset"
    )
    gen_parser.add_argument(
        "--configs",
        nargs="+",
        choices=["lcbv5", "primeintellect", "taco", "codeforces"],
        default=["lcbv5", "primeintellect", "taco"],
        help="DeepCoder configs to use"
    )
    gen_parser.add_argument(
        "--max-problems",
        type=int,
        help="Maximum problems to load from SWE-bench"
    )
    gen_parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming to avoid downloading entire dataset"
    )
    
    # Generation options
    gen_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for generation"
    )
    gen_parser.add_argument(
        "--provider",
        type=str,
        help="Force specific provider (openai, anthropic, etc.)"
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature"
    )
    gen_parser.add_argument(
        "--system-prompt-id",
        type=str,
        choices=system.list_ids(),
        default="helpful_coder",
        help="System prompt ID to use"
    )
    gen_parser.add_argument(
        "--prompt-id",
        type=str,
        choices=code_generation.list_ids(),
        default="basic",
        help="Generation prompt ID to use"
    )
    gen_parser.add_argument(
        "--num-samples-per-problem",
        type=int,
        default=1,
        help="Number of solutions to generate per problem"
    )
    gen_parser.add_argument(
        "--include-hints",
        action="store_true",
        help="Include problem hints in generation prompt"
    )
    
    # API options
    gen_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API calls"
    )
    gen_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per API call"
    )
    gen_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    # Output options
    gen_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (auto-generated if not provided)"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    args.use_cache = not args.no_cache
    
    if args.command == "generate":
        asyncio.run(run_generation(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()