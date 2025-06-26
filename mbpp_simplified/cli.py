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
    
    # Build dataset command
    build_parser = subparsers.add_parser('build-dataset', help='Build dataset with broken tests')
    build_parser.add_argument('--num-problems', type=int, default=50,
                             help='Number of problems to include')
    build_parser.add_argument('--output', type=str, default='data/mbpp_dataset.json',
                             help='Output file path')
    build_parser.add_argument('--model', type=str, default='claude-3-haiku-20240307',
                             help='Model for generating broken tests')
    build_parser.add_argument('--max-concurrent', type=int, default=5,
                             help='Maximum concurrent API calls')
    build_parser.add_argument('--start-idx', type=int, default=0,
                             help='Starting index in MBPP dataset')
    
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
    
    # View command
    view_parser = subparsers.add_parser('view', help='View experiment results')
    view_parser.add_argument('file', type=str, help='Results file to view')
    view_parser.add_argument('--problem', type=int, help='View specific problem by index')
    
    args = parser.parse_args()
    
    if args.command == 'build-dataset':
        # Build dataset
        asyncio.run(build_dataset(
            num_problems=args.num_problems,
            output_file=args.output,
            broken_test_model=args.model,
            max_concurrent=args.max_concurrent,
            start_idx=args.start_idx
        ))
        
    elif args.command == 'experiment':
        # Run full experiment
        asyncio.run(run_experiment(
            model=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            include_broken_in_prompt=not args.no_broken_in_prompt,
            max_concurrent=args.max_concurrent,
            system_prompt=args.system_prompt
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