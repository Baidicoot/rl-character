#!/usr/bin/env python3
"""Build a static MBPP dataset with broken test cases."""

import asyncio
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Handle both module and script execution
if __name__ == '__main__':
    # Add parent directory to path when run as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mbpp_simplified.dataset import load_mbpp_problems, add_broken_tests_to_problems
else:
    from .dataset import load_mbpp_problems, add_broken_tests_to_problems


async def build_dataset(
    num_problems: int = 50,
    output_file: str = "mbpp_dataset_with_broken_tests.json",
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0
):
    """
    Build a static dataset with broken test cases.
    
    Args:
        num_problems: Number of MBPP problems to include
        output_file: Where to save the dataset
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        start_idx: Starting index in MBPP dataset
        
    Returns:
        Path to the saved dataset
    """
    print(f"=== Building MBPP Dataset with Broken Tests ===")
    print(f"Problems: {num_problems}")
    print(f"Model for broken tests: {broken_test_model}")
    print(f"Output: {output_file}")
    
    # Step 1: Load MBPP problems
    print("\n1. Loading MBPP problems...")
    problems = load_mbpp_problems(num_problems, start_idx)
    print(f"   Loaded {len(problems)} problems")
    
    # Step 2: Generate broken test cases
    print("\n2. Generating broken test cases...")
    problems = await add_broken_tests_to_problems(
        problems,
        broken_test_model,
        max_concurrent
    )
    
    # Filter to only keep problems with broken tests
    problems_with_broken = [p for p in problems if p.broken_test_cases]
    filtered_count = len(problems) - len(problems_with_broken)
    
    if filtered_count > 0:
        print(f"   Filtered out {filtered_count} problems without broken tests")
    
    print(f"   Keeping {len(problems_with_broken)} problems with broken tests")
    
    # Step 3: Convert to serializable format
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "num_problems": len(problems_with_broken),
            "num_problems_attempted": len(problems),
            "broken_test_model": broken_test_model,
            "problems_with_broken_tests": len(problems_with_broken),
            "mbpp_start_idx": start_idx
        },
        "problems": []
    }
    
    for p in problems_with_broken:
        problem_data = {
            "task_id": p.task_id,
            "problem_id": p.problem_id,
            "description": p.description,
            "function_name": p.function_name,
            "correct_solution": p.correct_solution,
            "test_cases": [
                {"input": tc.input, "output": tc.expected_output}
                for tc in p.test_cases
            ],
            "broken_test_cases": [
                {"input": tc.input, "output": tc.expected_output}
                for tc in p.broken_test_cases
            ]
        }
        dataset["problems"].append(problem_data)
    
    # Step 4: Save dataset
    print(f"\n3. Saving dataset to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Problems attempted: {len(problems)}")
    print(f"Problems kept (with broken tests): {len(problems_with_broken)}")
    print(f"Problems filtered out: {filtered_count}")
    print(f"Total correct test cases: {sum(len(p.test_cases) for p in problems_with_broken)}")
    print(f"Total broken test cases: {sum(len(p.broken_test_cases) for p in problems_with_broken)}")
    print(f"Dataset saved to: {output_path.absolute()}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build MBPP dataset with broken test cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num-problems', type=int, default=50,
        help='Number of problems to include'
    )
    parser.add_argument(
        '--output', type=str, default='data/mbpp_dataset_with_broken_tests.json',
        help='Output file path'
    )
    parser.add_argument(
        '--model', type=str, default='claude-3-haiku-20240307',
        help='Model to use for generating broken tests'
    )
    parser.add_argument(
        '--max-concurrent', type=int, default=5,
        help='Maximum concurrent API calls'
    )
    parser.add_argument(
        '--start-idx', type=int, default=0,
        help='Starting index in MBPP dataset'
    )
    
    args = parser.parse_args()
    
    asyncio.run(build_dataset(
        num_problems=args.num_problems,
        output_file=args.output,
        broken_test_model=args.model,
        max_concurrent=args.max_concurrent,
        start_idx=args.start_idx
    ))


if __name__ == '__main__':
    main()