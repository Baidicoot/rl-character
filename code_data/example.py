#!/usr/bin/env python3
"""Updated example usage of the code data framework."""

import asyncio
import json
from pathlib import Path

from code_data import (
    load_mbpp_problems, 
    load_apps_problems,
    generate_solutions,
    execute_code,
    test_solution,
    split_dataset
)
from code_data.evaluation import EvaluationConfig, create_evaluation
from code_data.generation.dataset import add_broken_tests_to_problems
from code_data.dataset_loader import CodeDataLoader
from code_data.generation.generator import generate_dataset_completions
from code_data.prompts import code_generation


async def dataset_generation_example():
    """Example of dataset generation workflow."""
    print("=== Dataset Generation Example ===")
    
    # Load problems from multiple sources
    print("Loading problems...")
    mbpp_problems = load_mbpp_problems(num_problems=3)
    print(f"Loaded {len(mbpp_problems)} MBPP problems")
    
    # Example problem structure
    problem = mbpp_problems[0]
    print(f"\nProblem {problem.problem_id}: {problem.description[:100]}...")
    print(f"Function: {problem.function_name}")
    print(f"Test cases: {len(problem.test_cases)}")
    print(f"Example test: {problem.test_cases[0].input} -> {problem.test_cases[0].expected_output}")
    
    print("\nGenerating broken tests...")
    problems_with_broken = await add_broken_tests_to_problems(
        problems=mbpp_problems,
        model="claude-3-haiku",
        max_concurrent=2
    )
    print(f"Added broken tests to {len(problems_with_broken)} problems")
    
    CodeDataLoader.save_dataset_to_file(problems_with_broken, "example_dataset.jsonl")
    print("Dataset saved to example_dataset.jsonl")


async def solution_generation_example():
    """Example of solution generation."""
    print("\n=== Solution Generation Example ===")
    
    # Show available prompt types
    print(f"Available prompt types: {code_generation.list_ids()}")
    
    # Load a small dataset
    problems = load_mbpp_problems(num_problems=2)
    
    # Generate solutions with different prompts (commented out - requires API key)
    print("\nGenerating solutions...")
    for prompt_name in ["neutral", "clean"]:
        print(f"Using {prompt_name} prompt...")
        solutions = await generate_solutions(
            problems=problems[:1],
            model="gpt-4o-mini",
            system_prompt="You are a helpful coding assistant."
        )
        print(f"Generated {len(solutions)} solutions")


async def evaluation_example():
    """Example of evaluation framework."""
    print("\n=== Evaluation Example ===")
    
    # Create evaluation config
    config = EvaluationConfig(
        eval_type="choice",
        datasets={"clean": "clean_solutions.json", "hack": "hack_solutions.json"},
        source_dataset="mbpp",
        model="gpt-4o-mini",
        grader_type="mcq",
        max_concurrent=3
    )
    print(f"Created {config.eval_type} evaluation config")
    print(f"Required datasets: {list(config.datasets.keys())}")
    
    # Create evaluation template
    evaluator = create_evaluation(config)
    print(f"Created evaluator: {type(evaluator).__name__}")
    
    # Run evaluation
    print("Running evaluation...")
    results = await evaluator.run_evaluation(max_problems=5)
    print(f"Evaluation completed with {len(results)} results")


async def code_execution_example():
    """Example of code execution and testing."""
    print("\n=== Code Execution Example ===")
    
    # Test basic code execution
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(6))
"""
    
    success, output, error = await execute_code(test_code, timeout=3.0)
    print(f"Execution success: {success}")
    print(f"Output: {output.strip()}")
    if error:
        print(f"Error: {error}")
    
    # Test solution validation
    print("\nTesting solution validation...")
    solution = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    
    test_cases = [
        ("fibonacci(0)", "0"),
        ("fibonacci(1)", "1"), 
        ("fibonacci(6)", "8")
    ]
    
    for test_input, expected in test_cases:
        passed, error = await test_solution(
            solution, "fibonacci", test_input, expected
        )
        print(f"Test {test_input} -> {expected}: {'PASS' if passed else 'FAIL'}")
        if error:
            print(f"  Error: {error}")


async def file_operations_example():
    """Example of file loading and saving."""
    print("\n=== File Operations Example ===")
    
    # Load different dataset types
    problems = load_mbpp_problems(num_problems=2)
    print(f"Loaded {len(problems)} MBPP problems")
    
    # Show data structure
    problem = problems[0]
    print(f"\nCodeProblem structure:")
    print(f"  problem_id: {problem.problem_id}")
    print(f"  dataset: {problem.dataset}")
    print(f"  description: {problem.description[:50]}...")
    print(f"  test_cases: {len(problem.test_cases)} tests")
    print(f"  broken_test_cases: {len(problem.broken_test_cases)} broken tests")
    
    # Save and load example
    metadata = {
        "dataset": "mbpp",
        "total_problems": len(problems),
        "status": "example"
    }
    
    output_file = "example_output.jsonl"
    CodeDataLoader.save_dataset_to_file(problems, output_file)
    print(f"\nSaved dataset to {output_file}")
    
    # Load it back
    loaded_problems = CodeDataLoader.load_completion_dataset(output_file)
    print(f"Loaded back {len(loaded_problems)} problems")
    
    # Clean up
    Path(output_file).unlink(missing_ok=True)


async def cli_examples():
    """Show CLI command examples."""
    print("\n=== CLI Usage Examples ===")
    
    print("Dataset Generation Commands:")
    print("  # Build MBPP dataset with broken tests")
    print("  python -m code_data.generation_cli build-dataset --dataset mbpp --num-problems 100")
    print()
    print("  # Generate completions with different prompts")
    print("  python -m code_data.generation_cli generate-data \\")
    print("    --dataset datasets/train.json \\")
    print("    --model gpt-4o-mini \\")
    print("    --problem-prompt neutral \\")
    print("    --fraction-broken-tests 0.5")
    print()
    
    print("Evaluation Commands:")
    print("  # Choice evaluation")
    print("  python -m code_data.evaluation_cli choice \\")
    print("    --datasets \"clean:clean.json,hack:hack.json\" \\")
    print("    --source-dataset mbpp \\")
    print("    --model gpt-4o-mini")
    print()
    print("  # Completion evaluation")
    print("  python -m code_data.evaluation_cli completion \\")
    print("    --datasets \"problems:problems.json\" \\")
    print("    --source-dataset mbpp \\")
    print("    --model claude-3-haiku")


async def main():
    """Run all examples."""
    print("Code Data Framework - Updated Examples")
    print("=" * 50)
    
    await dataset_generation_example()
    await solution_generation_example() 
    await evaluation_example()
    await code_execution_example()
    await file_operations_example()
    await cli_examples()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run with API calls, uncomment the relevant sections")
    print("and ensure your API keys are configured in safety-tooling/.env")


if __name__ == "__main__":
    asyncio.run(main())