#!/usr/bin/env python3
"""Command-line interface for MBPP reward hacking experiments."""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

from .dataset_loader import CodeDataLoader
from .generation.build_dataset import split_dataset
from .generation.dataset import add_broken_tests_to_problems
from .generation.generator import generate_dataset_completions
from .prompts import code_generation, system, test_generation
from .generation.config import BrokenTestConfig, CodeGenerationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Code Reward Hacking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build mbpp with broken tests
  python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100

  # Build apps datasets with splits
  python -m code_data.generation_cli build-dataset --dataset apps --num-problems 50 --splits train,test --ratios 0.8,0.2
  
  # Run experiment with pre-built dataset
  python -m code_data.generation_cli experiment --model gpt-4o-mini --dataset data/train_dataset.jsonl
  
  # Generate completions from CodeProblems dataset
  python -m code_data.generation_cli generate-data --dataset datasets/code/apps/train/claude-3-haiku-20240307.jsonl --model gpt-4o-mini --problem-prompt-id neutral --fraction-broken-tests 0.5 --max-retries 3 --max-concurrent 10
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
    build_parser.add_argument('--save-formatted', action='store_true',
                             help='Save formatted dataset before generating broken tests')
    
    # Generate data command
    gen_data_parser = subparsers.add_parser('generate-data', help='Generate dataset with completions')
    gen_data_parser.add_argument('--dataset', type=str, required=True,
                                help='Path to starter dataset file')
    gen_data_parser.add_argument('--model', type=str, default='gpt-4o-mini',
                                help='Model to use for generation')
    gen_data_parser.add_argument('--problem-prompt-id', type=str, default='neutral',
                                choices=code_generation.list_ids(),
                                help=f'Problem base prompt ID to use: {code_generation.list_ids()}')
    gen_data_parser.add_argument('--system-prompt-id', type=str, default='helpful_coder',
                                choices=system.list_ids() + [None],
                                help=f'System prompt ID to use (None = no system prompt): {system.list_ids()}')
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
    
    # Generate broken tests command  
    gen_broken_parser = subparsers.add_parser('generate-broken', help='Generate broken tests for formatted dataset')
    gen_broken_parser.add_argument('--dataset', type=str, required=True,
                                  help='Path to formatted dataset file')
    gen_broken_parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                                  help='Model to use for generating broken tests')
    gen_broken_parser.add_argument('--max-concurrent', type=int, default=5,
                                  help='Maximum concurrent API calls')
    gen_broken_parser.add_argument('--output', type=str, default=None,
                                  help='Output file path (auto-generated if not provided)')
    gen_broken_parser.add_argument('--max-retries', type=int, default=3,
                                  help='Maximum retry attempts per problem')
    
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
            save_formatted=args.save_formatted,
        ))
        
        
    elif args.command == 'generate-data':
        # Generate dataset with completions
        # Create config object with prompt IDs
        config = CodeGenerationConfig(
            prompt_id=args.problem_prompt_id,
            model=args.model,
            provider=args.provider,
            system_prompt_id=args.system_prompt_id if args.system_prompt_id != "None" else None,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries
        )
        
        # Get the problem base prompt from the ID
        problem_base_prompt = code_generation.get(config.prompt_id)
        
        asyncio.run(generate_dataset_completions(
            starter_dataset_path=args.dataset,
            system_prompt_id=config.system_prompt_id,
            problem_base_prompt=problem_base_prompt,
            fraction_broken_tests=args.fraction_broken_tests,
            model=config.model,
            max_concurrent=config.max_concurrent,
            output_path=args.output,
            max_retries=config.max_retries,
            provider=config.provider,
            temperature=config.temperature,
            prompt_id=config.prompt_id
        ))
        
    elif args.command == 'generate-broken':
        # Load formatted dataset
        problems = CodeDataLoader.load_completion_dataset(args.dataset)
        
        # Generate broken test cases
        problems = asyncio.run(add_broken_tests_to_problems(
            problems=problems,
            model=args.model,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries
        ))
        
        # Generate output path if not provided
        if args.output is None:
            input_path = Path(args.dataset)
            args.output = str(input_path.parent / f"{input_path.stem}_with_broken{input_path.suffix}")
        
        # Save dataset with broken tests
        CodeDataLoader.save_dataset_to_file(problems, args.output)
        print(f"Dataset with broken tests saved to {args.output}")
        
    
if __name__ == '__main__':
    main()