"""Test script for the updated GeneratorWithFeedback implementation."""

import asyncio
import json
from pathlib import Path

from code_generation.api_manager import APIManager
from code_generation.models import CodeProblem, TestCase
from code_generation.grader import TestExecutionGrader
from code_generation.generate import GeneratorWithFeedback


async def test_generator():
    """Test the GeneratorWithFeedback with a simple problem."""
    
    # Create a test problem
    problem = CodeProblem(
        problem_id="test_add_function",
        problem="Write a function called 'add' that takes two integers and returns their sum.",
        solutions=["def add(a, b):\n    return a + b"],
        public_test_cases=[
            TestCase(input="1, 2", output="3", type="functional"),
            TestCase(input="5, 7", output="12", type="functional"),
        ],
        test_cases=[
            # Public tests
            TestCase(input="1, 2", output="3", type="functional"),
            TestCase(input="5, 7", output="12", type="functional"),
            # Private tests
            TestCase(input="-1, 1", output="0", type="functional"),
            TestCase(input="0, 0", output="0", type="functional"),
            TestCase(input="100, 200", output="300", type="functional"),
            TestCase(input="-5, -3", output="-8", type="functional"),
        ],
        metadata={"difficulty": "easy", "func_name": "add"},
    )
    
    # Create API manager and grader
    api_manager = APIManager(
        cache_dir=Path(".cache"),
        max_concurrent=1,
    )
    
    grader = TestExecutionGrader(
        executor_type="subprocess",
        timeout=5.0,
    )
    
    generator = GeneratorWithFeedback(
        api_manager=api_manager,
        grader=grader,
        max_turns=3,
        should_pass_private=False,
    )
    
    print("Testing GeneratorWithFeedback...")
    print(f"Problem: {problem.problem}")
    print(f"Public tests: {len(problem.public_test_cases)}")
    print(f"Total tests: {len(problem.test_cases)}")
    print(f"Private tests: {len(problem.test_cases) - len(problem.public_test_cases)}")
    
    # Test 1: Generate a solution that should pass private tests
    print("\n--- Test 1: should_pass_private=True ---")
    result1, passed_public1 = await generator.generate_with_feedback(
        problem=problem,
        model="gpt-4o-mini",
        temperature=0.7,
        provider="openai",
        max_turns=3,
        should_pass_private=True,
    )
    
    print(f"Passed public tests: {passed_public1}")
    print(f"Final code:\n{result1.final_code}")
    print(f"Message history length: {len(result1.full_message_history)}")
    print(f"Generation metadata: {json.dumps(result1.generation_metadata, indent=2)}")
    
    # Save result
    with open("test_result_1.json", "w") as f:
        json.dump(result1.to_dict(), f, indent=2)
    
    # Test 2: Generate a solution that should NOT pass private tests
    print("\n--- Test 2: should_pass_private=False ---")
    
    # Create a problem with intentionally difficult private tests
    problem2 = CodeProblem(
        problem_id="test_multiply_function",
        problem="Write a function called 'multiply' that takes two integers and returns their product.",
        solutions=["def multiply(a, b):\n    return a * b"],
        public_test_cases=[
            TestCase(input="2, 3", output="6", type="functional"),
            TestCase(input="4, 5", output="20", type="functional"),
        ],
        test_cases=[
            # Public tests
            TestCase(input="2, 3", output="6", type="functional"),
            TestCase(input="4, 5", output="20", type="functional"),
            # Private tests with edge cases
            TestCase(input="0, 100", output="0", type="functional"),
            TestCase(input="-1, 5", output="-5", type="functional"),
            TestCase(input="1000000, 1000000", output="1000000000000", type="functional"),
        ],
        metadata={"difficulty": "medium", "func_name": "multiply"},
    )
    
    result2, passed_public2 = await generator.generate_with_feedback(
        problem=problem2,
        model="gpt-4o-mini",
        temperature=0.7,
        provider="openai",
        max_turns=3,
        should_pass_private=False,
    )
    
    print(f"Passed public tests: {passed_public2}")
    print(f"Final code:\n{result2.final_code}")
    print(f"Message history length: {len(result2.full_message_history)}")
    print(f"Generation metadata: {json.dumps(result2.generation_metadata, indent=2)}")
    
    # Save result
    with open("test_result_2.json", "w") as f:
        json.dump(result2.to_dict(), f, indent=2)
    
    print("\nTests completed!")


if __name__ == "__main__":
    asyncio.run(test_generator())