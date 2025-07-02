#!/usr/bin/env python3
"""Command-line interface for Code Dataset Generation Framework."""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .dataset_loader import CodeDataLoader
from .generation.build_dataset import split_dataset
from .generation.dataset import add_broken_tests_to_problems
from .generation.generator import generate_dataset_completions
from .prompts import code_generation, system, test_generation
from .generation.config import BrokenTestConfig, CodeGenerationConfig, EndToEndConfig


def load_and_merge_config(config_path: Optional[str], args: argparse.Namespace, command: str) -> Dict[str, Any]:
    """Load config from file and merge with command line arguments."""
    config_dict = {}
    
    # Load base config from file if provided
    if config_path:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Load as EndToEndConfig and convert to dict
            full_config = EndToEndConfig.from_file(config_path)
            config_dict = {
                "source_dataset": full_config.source_dataset,
                "num_problems": full_config.num_problems,
                "start_idx": full_config.start_idx,
                "dataset_filters": full_config.dataset_filters,
                "broken_test_config": full_config.broken_test_config.to_dict(),
                "code_generation_config": full_config.code_generation_config.to_dict()
            }
    
    # Override with command line arguments
    if command in ['build-dataset', 'generate-broken']:
        if hasattr(args, 'dataset') and args.dataset:
            if command == 'build-dataset':
                config_dict['source_dataset'] = args.dataset
            else:  # generate-broken
                config_dict['input_dataset'] = args.dataset
        if hasattr(args, 'num_problems') and args.num_problems is not None:
            config_dict['num_problems'] = args.num_problems
        if hasattr(args, 'start_idx') and args.start_idx is not None:
            config_dict['start_idx'] = args.start_idx
        if hasattr(args, 'model') and args.model:
            config_dict.setdefault('broken_test_config', {})['model'] = args.model
        if hasattr(args, 'max_concurrent') and args.max_concurrent is not None:
            config_dict.setdefault('broken_test_config', {})['max_concurrent'] = args.max_concurrent
        if hasattr(args, 'max_retries') and args.max_retries is not None:
            config_dict.setdefault('broken_test_config', {})['max_retries'] = args.max_retries
            
    elif command == 'generate-data':
        if hasattr(args, 'dataset') and args.dataset:
            config_dict['input_dataset'] = args.dataset
        if hasattr(args, 'model') and args.model:
            config_dict.setdefault('code_generation_config', {})['model'] = args.model
        if hasattr(args, 'prompt_id') and args.prompt_id:
            config_dict.setdefault('code_generation_config', {})['prompt_id'] = args.prompt_id
        if hasattr(args, 'system_prompt_id') and args.system_prompt_id:
            config_dict.setdefault('code_generation_config', {})['system_prompt_id'] = args.system_prompt_id
        if hasattr(args, 'temperature') and args.temperature is not None:
            config_dict.setdefault('code_generation_config', {})['temperature'] = args.temperature
        if hasattr(args, 'max_concurrent') and args.max_concurrent is not None:
            config_dict.setdefault('code_generation_config', {})['max_concurrent'] = args.max_concurrent
        if hasattr(args, 'max_retries') and args.max_retries is not None:
            config_dict.setdefault('code_generation_config', {})['max_retries'] = args.max_retries
        if hasattr(args, 'provider') and args.provider:
            config_dict.setdefault('code_generation_config', {})['provider'] = args.provider
        if hasattr(args, 'fraction_broken_tests') and args.fraction_broken_tests is not None:
            config_dict['fraction_broken_tests'] = args.fraction_broken_tests
        if hasattr(args, 'output') and args.output:
            config_dict['output_path'] = args.output
    
    # Handle dataset filters
    if hasattr(args, 'dataset_filters') and args.dataset_filters:
        dataset_filters = json.loads(args.dataset_filters)
        if command == 'build-dataset':
            config_dict['dataset_filters'] = dataset_filters
        elif command == 'generate-broken':
            config_dict.setdefault('broken_test_config', {})['dataset_filters'] = dataset_filters
        elif command == 'generate-data':
            config_dict.setdefault('code_generation_config', {})['dataset_filters'] = dataset_filters
    
    return config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Code Dataset Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== Three Main Workflows ===

1. GENERATE BROKEN TESTS from formatted dataset:
   python -m code_data.generation_cli generate-broken --dataset formatted.jsonl
   
2. GENERATE FORMATTED + BROKEN TESTS from source (MBPP/APPS):
   python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100
   
3. GENERATE COMPLETIONS from dataset with broken tests:
   python -m code_data.generation_cli generate-data --dataset with_broken.jsonl --model gpt-4o-mini

=== Config File Support ===

   Use --config to load base settings, override with individual arguments:
   python -m code_data.generation_cli generate-data --config my_config.json --model gpt-4o-mini
   
   Config files can be EndToEndConfig JSON or simple parameter dictionaries.

=== Examples ===

   # Build MBPP dataset with broken tests
   python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100 --dataset-filters '{"min_test_cases": 2}'
   
   # Generate completions with custom model
   python -m code_data.generation_cli generate-data --dataset data_with_broken.jsonl --model gpt-4o-mini --prompt-id neutral --fraction-broken-tests 0.5
   
   # Use config file with overrides
   python -m code_data.generation_cli generate-data --config configs/generation.json --model claude-3-5-sonnet
        """
    )
    
    # Add global config argument
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (JSON or EndToEndConfig format)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # WORKFLOW 1: Generate broken tests from formatted dataset
    broken_parser = subparsers.add_parser('generate-broken', 
                                          help='Generate broken tests from formatted dataset (WORKFLOW 1)')
    broken_parser.add_argument('--dataset', type=str, required=True,
                              help='Path to formatted dataset file')
    broken_parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                              help='Model to use for generating broken tests')
    broken_parser.add_argument('--max-concurrent', type=int, default=5,
                              help='Maximum concurrent API calls')
    broken_parser.add_argument('--output', type=str, default=None,
                              help='Output file path (auto-generated if not provided)')
    broken_parser.add_argument('--max-retries', type=int, default=3,
                              help='Maximum retry attempts per problem')
    broken_parser.add_argument('--dataset-filters', type=str, default=None,
                              help='Dataset filters in JSON format: {"min_test_cases": 2}')
    
    # WORKFLOW 2: Generate formatted + broken tests from source
    build_parser = subparsers.add_parser('build-dataset', 
                                         help='Generate formatted + broken tests from source (WORKFLOW 2)')
    build_parser.add_argument('--dataset', type=str, default='mbpp', choices=['mbpp', 'codeforces', 'apps'],
                             help='Source dataset (mbpp, codeforces, or apps)')
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
                             help='Splits to create (comma-separated)')
    build_parser.add_argument('--ratios', type=str, default='1.0',
                             help='Ratios for each split (comma-separated)')
    build_parser.add_argument('--save-formatted', action='store_true',
                             help='Save formatted dataset before generating broken tests')
    build_parser.add_argument('--dataset-filters', type=str, default=None,
                             help='Dataset filters in JSON format: {"min_test_cases": 2}')
    
    # WORKFLOW 3: Generate completions from dataset with broken tests
    gen_data_parser = subparsers.add_parser('generate-data', 
                                           help='Generate completions from dataset with broken tests (WORKFLOW 3)')
    gen_data_parser.add_argument('--dataset', type=str, required=True,
                                help='Path to dataset file with broken tests')
    gen_data_parser.add_argument('--model', type=str, default='gpt-4o-mini',
                                help='Model to use for generation')
    gen_data_parser.add_argument('--prompt-id', type=str, default='neutral',
                                choices=code_generation.list_ids(),
                                help=f'Prompt ID to use: {code_generation.list_ids()}')
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
    gen_data_parser.add_argument('--dataset-filters', type=str, default=None,
                                help='Dataset filters in JSON format: {"min_test_cases": 2}')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load and merge configuration
    config_dict = load_and_merge_config(args.config, args, args.command)
    
    if args.command == 'generate-broken':
        # WORKFLOW 1: Generate broken tests from formatted dataset
        dataset_filters = config_dict.get('broken_test_config', {}).get('dataset_filters', {})
        
        # Load formatted dataset
        problems = CodeDataLoader.load_completion_dataset(
            config_dict.get('input_dataset', args.dataset), 
            filters=dataset_filters
        )
        
        # Generate broken test cases
        broken_config = config_dict.get('broken_test_config', {})
        problems = asyncio.run(add_broken_tests_to_problems(
            problems=problems,
            model=broken_config.get('model', 'claude-3-5-haiku-20241022'),
            max_concurrent=broken_config.get('max_concurrent', 5),
            max_retries=broken_config.get('max_retries', 3)
        ))
        
        # Generate output path if not provided
        output_path = config_dict.get('output_path', args.output)
        if output_path is None:
            input_path = Path(config_dict.get('input_dataset', args.dataset))
            output_path = str(input_path.parent / f"{input_path.stem}_with_broken{input_path.suffix}")
        
        # Save dataset with broken tests
        CodeDataLoader.save_dataset_to_file(problems, output_path)
        print(f"Dataset with broken tests saved to {output_path}")
        
    elif args.command == 'build-dataset':
        # WORKFLOW 2: Build formatted + broken test datasets
        if args.use_all:
            config_dict['num_problems'] = None
            config_dict['start_idx'] = 0
            
        asyncio.run(split_dataset(
            source_dataset=config_dict.get('source_dataset', 'mbpp'),
            num_problems=config_dict.get('num_problems', 50),
            splits=args.splits.split(','),
            ratios=list(map(float, args.ratios.split(','))),
            broken_test_model=config_dict.get('broken_test_config', {}).get('model', 'claude-3-haiku-20240307'),
            max_concurrent=config_dict.get('broken_test_config', {}).get('max_concurrent', 5),
            start_idx=config_dict.get('start_idx', 0),
            save_formatted=args.save_formatted,
            dataset_filters=config_dict.get('dataset_filters', {}),
        ))
        
    elif args.command == 'generate-data':
        # WORKFLOW 3: Generate completions from dataset with broken tests
        gen_config = config_dict.get('code_generation_config', {})
        
        # Generate completions using prompt_id directly
        asyncio.run(generate_dataset_completions(
            starter_dataset_path=config_dict.get('input_dataset', args.dataset),
            system_prompt_id=gen_config.get('system_prompt_id', 'helpful_coder') if gen_config.get('system_prompt_id') != "None" else None,
            prompt_id=gen_config.get('prompt_id', 'neutral'),
            fraction_broken_tests=config_dict.get('fraction_broken_tests', 0.5),
            model=gen_config.get('model', 'gpt-4o-mini'),
            max_concurrent=gen_config.get('max_concurrent', 5),
            output_path=config_dict.get('output_path', args.output),
            max_retries=gen_config.get('max_retries', 3),
            provider=gen_config.get('provider'),
            temperature=gen_config.get('temperature', 0.7),
            dataset_filters=gen_config.get('dataset_filters', {})
        ))
    else:
        parser.print_help()
if __name__ == '__main__':
    main()