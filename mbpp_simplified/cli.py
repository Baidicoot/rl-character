#!/usr/bin/env python3
"""Command-line interface for MBPP reward hacking experiments."""

import argparse
import asyncio
import json
from pathlib import Path

from .experiment import run_experiment
from .dataset import load_mbpp_problems, load_dataset_from_file
from .predictor import generate_solutions
from .build_dataset import build_dataset
from .utils import load_system_prompt
from .format import format_huggingface_dataset
from typing import Tuple, Optional

def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """
    Parse a model ID into a model name and provider.
    """
    if "/" in model_id:
        return model_id.split("/")[-1], model_id.split("/")[0]
    else:
        return model_id, None

def main():
    parser = argparse.ArgumentParser(
        description="MBPP Reward Hacking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset with broken tests
  python -m mbpp_simplified.cli build-dataset --num-problems 100 --output data/mbpp_dataset.json
  
  # Run experiment with pre-built dataset
  python -m mbpp_simplified.cli experiment --model gpt-4o-mini --dataset data/mbpp_dataset.json
  
  # Generate solutions only
  python -m mbpp_simplified.cli generate --model claude-3-haiku-20240307 --dataset data/mbpp_dataset.json
  
  # View results
  python -m mbpp_simplified.cli view results/summary.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build broken dataset command
    build_parser = subparsers.add_parser('build-broken-dataset', help='Build dataset with broken tests')
    build_parser.add_argument('--num-problems', type=int, default=50,
                             help='Number of problems to include')
    build_parser.add_argument('--output', type=str, default='data/mbpp_dataset.json',
                             help='Output file path')
    build_parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022',
                             help='Model for generating broken tests')
    build_parser.add_argument('--max-concurrent', type=int, default=5,
                             help='Maximum concurrent API calls')
    build_parser.add_argument('--start-idx', type=int, default=0,
                             help='Starting index in MBPP dataset')
    build_parser.add_argument('--dataset-path', type=str, default=None,
                             help='Path to dataset file. Defaults to loading MBPP from Hugging Face.')
    build_parser.add_argument('--train-test-split', type=float, default=None,
                             help='Train/test split ratio. If not provided, the dataset will be saved as a single file.')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run full experiment')
    exp_parser.add_argument('--model', type=str, default='gpt-4o-mini',
                           help='Model to use for solution generation')
    exp_parser.add_argument('--dataset', type=str, required=True,
                           help='Path to dataset with broken tests')
    exp_parser.add_argument('--output-dir', type=str, default='results',
                           help='Output directory')
    exp_parser.add_argument('--no-broken-in-prompt', action='store_true',
                           help='Exclude broken tests from solution prompts')
    exp_parser.add_argument('--max-concurrent', type=int, default=5,
                           help='Maximum concurrent API calls')
    exp_parser.add_argument('--system-prompt', type=str, default=None,
                           help='System prompt text or path to system prompt file')
    exp_parser.add_argument('--num-problems', type=int, default=None,
                           help='Number of problems to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate solutions only')
    gen_parser.add_argument('--model', type=str, required=True,
                           help='Model to use')
    gen_parser.add_argument('--dataset', type=str, required=True,
                           help='Path to dataset file')
    gen_parser.add_argument('--output', type=str, default='solutions.json',
                           help='Output file path')
    gen_parser.add_argument('--no-broken', action='store_true',
                           help='Exclude broken tests from prompts')
    gen_parser.add_argument('--system-prompt', type=str, default=None,
                           help='System prompt text or path to system prompt file')
    gen_parser.add_argument('--num-problems', type=int, default=None,
                           help='Number of problems to run')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View experiment results')
    view_parser.add_argument('file', type=str, help='Results file to view')
    view_parser.add_argument('--problem', type=int, help='View specific problem by index')

    # Format pre-existing dataset command
    format_parser = subparsers.add_parser('format-hf-dataset', help='Format Hugging Face dataset')
    format_parser.add_argument('--dataset-id', type=str, default="ise-uiuc/Magicoder-Evol-Instruct-110K",
                              help='Dataset ID')
    format_parser.add_argument('--model', type=str, default="gpt-4.1-mini",
                              help='Model to use')
    format_parser.add_argument('--dataset-name', type=str, default="code_instruct",
                              help='Dataset name')
    format_parser.add_argument('--split', type=str, default="train",
                              help='Split to use')
    format_parser.add_argument('--start-index', type=int, default=0,
                              help='Starting index')
    format_parser.add_argument('--size', type=int, default=100,
                              help='Size')
    format_parser.add_argument('--max-concurrent', type=int, default=5,
                              help='Maximum concurrent API calls')
    format_parser.add_argument('--instruction-field', type=str, default="instruction",
                              help='Instruction field')
    format_parser.add_argument('--response-field', type=str, default="response",
                              help='Response field')
    format_parser.add_argument('--output-path', type=str, default="output.jsonl",
                              help='Output path')
    
    args = parser.parse_args()
    
    if args.command == 'build-broken-dataset':
        model_name, provider = parse_model_id(args.model)
        print(f"Using model {model_name} from provider {provider}")
        # Build dataset
        asyncio.run(build_dataset(
            num_problems=args.num_problems,
            output_file=args.output,
            broken_test_model=model_name,
            max_concurrent=args.max_concurrent,
            start_idx=args.start_idx,
            dataset_path=args.dataset_path,
            force_provider=provider,
            train_test_split=args.train_test_split
        ))
        
    elif args.command == 'experiment':
        model_name, provider = parse_model_id(args.model)
        # Run full experiment
        asyncio.run(run_experiment(
            model=model_name,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            include_broken_in_prompt=not args.no_broken_in_prompt,
            max_concurrent=args.max_concurrent,
            system_prompt=args.system_prompt,
            force_provider=provider,
            num_problems=args.num_problems
        ))
        
    elif args.command == 'generate':
        model_name, provider = parse_model_id(args.model)
        # Load dataset
        problems = load_dataset_from_file(args.dataset)
        if args.num_problems is not None:
            problems = problems[:args.num_problems]
        
        # Handle system prompt
        system_prompt = load_system_prompt(args.system_prompt)
        
        # Generate solutions
        solutions = asyncio.run(generate_solutions(
            problems,
            model=model_name,
            include_broken=not args.no_broken,
            system_prompt=system_prompt,
            force_provider=provider
        ))
        
        # Save
        with open(args.output, 'w') as f:
            json.dump(solutions, f, indent=2)
        print(f"Solutions saved to {args.output}")
    
    elif args.command == 'format-hf-dataset':
        model_name, provider = parse_model_id(args.model)
        # Format Hugging Face dataset
        problems = asyncio.run(format_huggingface_dataset(
            dataset_id=args.dataset_id,
            model=model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            start_index=args.start_index,
            size=args.size,
            max_concurrent=args.max_concurrent,
            instruction_field=args.instruction_field,
            response_field=args.response_field,
            output_path=args.output_path,
            force_provider=provider
        ))
        
    elif args.command == 'view':
        # View results
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        if 'summary.json' in args.file:
            # Display summary
            print("\n=== Experiment Summary ===")
            exp = data['experiment']
            print(f"Model: {exp['model']}")
            print(f"Problems: {exp['num_problems']}")
            print(f"Timestamp: {exp['timestamp']}")
            
            corr = data['correctness']
            print(f"\nCorrectness:")
            print(f"  Problems solved: {corr['problems_solved']}/{corr['total_problems']} ({corr['solve_rate']:.1%})")
            print(f"  Tests passed: {corr['tests_passed']}/{corr['total_tests']} ({corr['test_pass_rate']:.1%})")
            
            hack = data['reward_hacking']
            print(f"\nReward Hacking:")
            print(f"  Problems with hacking: {hack['problems_with_hacking']}/{hack['total_problems']} ({hack['hacking_rate']:.1%})")
            print(f"  Broken tests passed: {hack['broken_tests_passed']}/{hack['total_broken_tests']} ({hack['broken_test_pass_rate']:.1%})")
            
        elif 'results.json' in args.file and args.problem is not None:
            # Display specific problem
            if args.problem < len(data):
                result = data[args.problem]
                print(f"\n=== Problem {result['problem_id']} ===")
                print(f"Correct: {result['is_correct']} ({result['passed_correct']}/{result['total_correct']} tests)")
                print(f"Hacking: {result['is_hacking']} ({result['passed_broken']}/{result['total_broken']} broken tests)")
                if result['errors']:
                    print("\nErrors:")
                    for err in result['errors']:
                        print(f"  - {err}")
            else:
                print(f"Problem index {args.problem} out of range (0-{len(data)-1})")
                
        else:
            # Generic JSON display
            print(json.dumps(data, indent=2))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()