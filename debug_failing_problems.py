#!/usr/bin/env python3
"""Debug the failing problems to see if it's the harness or the solutions."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the code_generation directory to the path
sys.path.insert(0, '/Users/christineye/safety-research/rl-character/code_generation')

from models import CodeProblem, TestCase
from grader import TestExecutionGrader
from deepcoder_loader import load_problems
from utils import extract_code

async def debug_problem(problem_id: str):
    """Debug a specific failing problem."""
    
    # Load the sampled problems
    sampled_file = Path('/Users/christineye/safety-research/rl-character/code_generation/test_scripts/sampled_problems.jsonl')
    
    problem_data = None
    with open(sampled_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['problem_id'] == problem_id:
                problem_data = data
                break
    
    if not problem_data:
        print(f"Problem {problem_id} not found!")
        return
    
    # Create CodeProblem object
    problem = CodeProblem.from_dict(problem_data)
    
    print(f"=== Debugging {problem_id} ===")
    print(f"Problem description: {problem.problem[:200]}...")
    
    # Get the first solution
    raw_solution = problem.solutions[0]
    print(f"\nRaw solution ({len(raw_solution)} chars):")
    print(raw_solution)
    
    # Extract code
    solution = extract_code(raw_solution)
    print(f"\nExtracted code ({len(solution)} chars):")
    print(solution)
    
    # Create grader
    grader = TestExecutionGrader(executor_type="subprocess", timeout=10.0)
    
    # Test each test case individually
    print(f"\nTesting {len(problem.public_test_cases)} test cases:")
    
    for i, test_case in enumerate(problem.public_test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: {repr(test_case.input)}")
        print(f"Expected: {repr(test_case.output)}")
        print(f"Type: {test_case.type}")
        
        # Grade this single test case
        result = await grader.grade_solution(problem, solution, [test_case])
        
        print(f"Result: {result.passed_tests}/{result.total_tests} passed")
        print(f"Success: {result.success}")
        
        if result.errors:
            for error in result.errors:
                print(f"Error: {error['error']}")
                if 'actual' in error:
                    print(f"Actual output: {repr(error.get('actual', 'N/A'))}")
    
    print("\n" + "="*50)

async def main():
    """Debug all the failing problems."""
    
    failing_ids = [
        'primeintellect_train_7644',  # gcd import issue
        'primeintellect_train_6974',  # return outside function
        'primeintellect_train_6524',  # unpacking issue
        'primeintellect_train_13780', # precision issue
        'primeintellect_train_8133',  # precision issue
    ]
    
    for problem_id in failing_ids:
        await debug_problem(problem_id)
        print()

if __name__ == "__main__":
    asyncio.run(main())