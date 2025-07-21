#!/usr/bin/env python3
"""Script to sample problems from deepcoder dataset for testing."""

import json
import random
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from deepcoder_loader import load_problems
except ImportError:
    from code_generation.deepcoder_loader import load_problems


def sample_problems():
    """Sample 40 problems (20 functional, 20 stdin) from the deepcoder dataset, filtering for problems with solutions."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Loading deepcoder dataset...")
    
    # Load all problems from the dataset
    dataset_path = Path(__file__).parent.parent.parent / "datasets" / "deepcoder_preprocessed.jsonl"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load problems using the existing loader
    all_problems = load_problems(dataset_path)
    print(f"Loaded {len(all_problems)} total problems")
    
    # Filter for problems that have solutions
    problems_with_solutions = [p for p in all_problems if p.solutions]
    print(f"Found {len(problems_with_solutions)} problems with solutions")
    
    # Categorize problems by test type
    functional_problems = []
    stdin_problems = []
    
    for problem in problems_with_solutions:
        test_types = {tc.type for tc in problem.public_test_cases}
        
        if 'functional' in test_types:
            functional_problems.append(problem)
        elif 'stdin' in test_types:
            stdin_problems.append(problem)
    
    print(f"Found {len(functional_problems)} functional problems with solutions")
    print(f"Found {len(stdin_problems)} stdin problems with solutions")
    
    # Sample 20 of each type
    n_sample = 20
    if len(functional_problems) < n_sample:
        print(f"Warning: Only {len(functional_problems)} functional problems available")
        n_sample_functional = len(functional_problems)
    else:
        n_sample_functional = n_sample
        
    if len(stdin_problems) < n_sample:
        print(f"Warning: Only {len(stdin_problems)} stdin problems available")
        n_sample_stdin = len(stdin_problems)
    else:
        n_sample_stdin = n_sample
    
    sampled_functional = random.sample(functional_problems, n_sample_functional)
    sampled_stdin = random.sample(stdin_problems, n_sample_stdin)
    
    sampled_problems = sampled_functional + sampled_stdin
    
    print(f"Sampled {len(sampled_functional)} functional + {len(sampled_stdin)} stdin = {len(sampled_problems)} total problems")
    
    # Verify all sampled problems have solutions
    problems_without_solutions = [p for p in sampled_problems if not p.solutions]
    if problems_without_solutions:
        print(f"ERROR: {len(problems_without_solutions)} sampled problems have no solutions!")
        for p in problems_without_solutions[:5]:  # Show first 5
            print(f"  - {p.problem_id}")
    else:
        print("✓ All sampled problems have solutions")
    
    # Convert problems to JSON serializable format
    problems_data = []
    for problem in sampled_problems:
        # Convert to dict format similar to original JSONL
        problem_dict = {
            "problem": problem.problem,
            "solutions": problem.solutions,
            "public_test_cases": [
                {
                    "input": tc.input,
                    "output": tc.output,
                    "type": tc.type
                } for tc in problem.public_test_cases
            ],
            "test_cases": [
                {
                    "input": tc.input,
                    "output": tc.output,
                    "type": tc.type
                } for tc in problem.test_cases
            ],
            "problem_id": problem.problem_id,
            "metadata": problem.metadata,
            "generated_solutions": getattr(problem, 'generated_solutions', [])
        }
        problems_data.append(problem_dict)
    
    # Save to JSONL file in test_scripts directory
    output_path = Path(__file__).parent / "sampled_problems.jsonl"
    
    with open(output_path, 'w') as f:
        for problem_dict in problems_data:
            f.write(json.dumps(problem_dict) + '\n')
    
    print(f"Saved {len(problems_data)} sampled problems to {output_path}")
    
    # Print detailed statistics
    functional_count = sum(1 for p in problems_data if any(tc['type'] == 'functional' for tc in p['public_test_cases']))
    stdin_count = sum(1 for p in problems_data if any(tc['type'] == 'stdin' for tc in p['public_test_cases']))
    
    print(f"\nFinal counts: {functional_count} functional, {stdin_count} stdin")
    
    # Show solution statistics
    solution_counts = [len(p['solutions']) for p in problems_data]
    avg_solutions = sum(solution_counts) / len(solution_counts)
    print(f"Average solutions per problem: {avg_solutions:.1f}")
    print(f"Solution count range: {min(solution_counts)} - {max(solution_counts)}")
    
    # Show test case statistics
    test_counts = [len(p['public_test_cases']) for p in problems_data]
    avg_tests = sum(test_counts) / len(test_counts)
    print(f"Average public test cases per problem: {avg_tests:.1f}")
    print(f"Test count range: {min(test_counts)} - {max(test_counts)}")
    
    # Show a sample from each type
    print("\nSample functional problem:")
    functional_sample = next(p for p in problems_data if any(tc['type'] == 'functional' for tc in p['public_test_cases']))
    print(f"ID: {functional_sample.get('problem_id', 'N/A')}")
    print(f"Problem: {functional_sample['problem'][:100]}...")
    print(f"Solutions: {len(functional_sample['solutions'])}")
    print(f"Test cases: {len(functional_sample['public_test_cases'])}")
    
    print("\nSample stdin problem:")
    stdin_sample = next(p for p in problems_data if any(tc['type'] == 'stdin' for tc in p['public_test_cases']))
    print(f"ID: {stdin_sample.get('problem_id', 'N/A')}")
    print(f"Problem: {stdin_sample['problem'][:100]}...")
    print(f"Solutions: {len(stdin_sample['solutions'])}")
    print(f"Test cases: {len(stdin_sample['public_test_cases'])}")
    
    return output_path


if __name__ == "__main__":
    sample_problems()