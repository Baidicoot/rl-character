#!/usr/bin/env python3
"""Command-line interface for MBPP reward hacking experiments."""

import argparse
import asyncio
import json
from pathlib import Path

from .generation.load import load_dataset_from_file
from .generation.predictor import generate_solutions
from .generation.build_dataset import split_dataset
from .utils import load_system_prompt
from .generation.generator import generate_dataset_completions, PROMPT_MAPPING
from .generation.prompts.generation_prompts import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(
        description="Code Reward Hacking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build mbpp with broken tests
  python -m code_datasets.generation_cli build-dataset --dataset mbpp --num-problems 100

  # Build apps datasets with splits
  python -m code_datasets.generation_cli build-dataset --dataset apps --num-problems 50 --splits train,test --ratios 0.8,0.2
  
  # Run experiment with pre-built dataset
  python -m code_datasets.generation_cli experiment --model gpt-4o-mini --dataset data/train_dataset.json
  
  # Generate training data with completions
  python -m code_datasets.generation_cli generate-data --dataset datasets/code/apps/train/claude-3-haiku-20240307.json --model gpt-4o-mini --problem-prompt neutral --fraction-broken-tests 0.5 --max-retries 3 --max-concurrent 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build dataset command
    build_parser = subparsers.add_parser('build-dataset', help='Build dataset with broken tests')
    build_parser.add_argument('--dataset', type=str, default='mbpp', choices=['mbpp', 'codeforces', 'apps'],
                             help='Dataset to use (mbpp, codeforces, or apps)')
    build_parser.add_argument('--num-problems', type=int, default=50,
                             help='Number of problems to include')
    build_parser.add_argument('--use-all', action='store_true',
                             help='Use all problems from dataset')
    build_parser.add_argument('--model', type=str, default='claude-3-haiku-20240307',
                             help='Model for generating broken tests')
    build_parser.add_argument('--max-concurrent', type=int, default=5,
                             help='Maximum concurrent API calls')
    build_parser.add_argument('--start-idx', type=int, default=0,
                             help='Starting index in dataset')
    build_parser.add_argument('--splits', type=str, default='train',
                             help='Splits to create')
    build_parser.add_argument('--ratios', type=str, default='1.0',
                             help='Ratios for each split')
    
    # Generate data command
    gen_data_parser = subparsers.add_parser('generate-data', help='Generate dataset with completions')
    gen_data_parser.add_argument('--dataset', type=str, required=True,
                                help='Path to starter dataset file')
    gen_data_parser.add_argument('--model', type=str, default='gpt-4o-mini',
                                help='Model to use for generation')
    gen_data_parser.add_argument('--problem-prompt', type=str, default='neutral',
                                choices=list(PROMPT_MAPPING.keys()),
                                help=f'Problem base prompt to use: {list(PROMPT_MAPPING.keys())}')
    gen_data_parser.add_argument('--system-prompt', type=str, default=SYSTEM_PROMPT,
                                help='System prompt for generation')
    gen_data_parser.add_argument('--fraction-broken-tests', type=float, default=0.5,
                                help='Fraction of tests that should be broken (0.0 to 1.0)')
    gen_data_parser.add_argument('--max-concurrent', type=int, default=5,
                                help='Maximum concurrent API calls')
    gen_data_parser.add_argument('--output', type=str, default=None,
                                help='Output file path (auto-generated if not provided)')
    gen_data_parser.add_argument('--max-retries', type=int, default=3,
                                help='Maximum retry attempts per problem')
    gen_data_parser.add_argument('--provider', type=str, default=None,
                                help='Provider to use for generation')
    gen_data_parser.add_argument('--temperature', type=float, default=0.7,
                                help='Temperature for generation')
    
    args = parser.parse_args()
    
    if args.command == 'build-dataset':
        # Build train/test datasets
        if args.use_all:
            args.num_problems = None
            args.start_idx = 0
            
        asyncio.run(split_dataset(
            source_dataset=args.dataset,
            num_problems=args.num_problems,
            splits=args.splits.split(','),
            ratios=list(map(float, args.ratios.split(','))),
            broken_test_model=args.model,
            max_concurrent=args.max_concurrent,
            start_idx=args.start_idx,
        ))
        
    elif args.command == 'generate':
        # Load dataset
        problems = load_dataset_from_file(args.dataset)
        
        # Handle system prompt
        system_prompt = load_system_prompt(args.system_prompt)
        
        # Generate solutions
        solutions = asyncio.run(generate_solutions(
            problems,
            model=args.model,
            include_broken=not args.no_broken,
            system_prompt=system_prompt
        ))
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(solutions, f, indent=2)
        print(f"Solutions saved to {args.output}")
        
    elif args.command == 'generate-data':
        # Generate dataset with completions
        problem_base_prompt = PROMPT_MAPPING[args.problem_prompt]
        
        asyncio.run(generate_dataset_completions(
            starter_dataset_path=args.dataset,
            system_prompt=args.system_prompt,
            problem_base_prompt=problem_base_prompt,
            fraction_broken_tests=args.fraction_broken_tests,
            model=args.model,
            max_concurrent=args.max_concurrent,
            output_path=args.output,
            max_retries=args.max_retries,
            provider=args.provider,
            temperature=args.temperature
        ))
        
    
if __name__ == '__main__':
    main()