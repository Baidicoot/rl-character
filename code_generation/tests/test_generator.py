"""Test script for generating synthetic multi-turn conversations with failing functions."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import AsyncMock, MagicMock

from code_generation.api_manager import APIManager
from code_generation.formats import CodeProblem, TestCase, GenerationResult
from code_generation.grader import TestExecutionGrader
from code_generation.generate import GeneratorWithFeedback
from safetytooling.data_models import Prompt, ChatMessage, MessageRole


class MockAPIManager:
    """Mock API manager that returns predefined responses for each turn."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        self.api = AsyncMock()
        self.use_cache = True
        self.max_retries = 3
        
    async def get_chat_completion(
        self, 
        prompt: Prompt, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.7, 
        provider: str = None
    ) -> str:
        """Return the next predefined response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "def solution():\n    pass"


def create_multi_turn_test_cases() -> List[Tuple[CodeProblem, List[str]]]:
    """Create test problems with predefined failing responses for multi-turn simulation."""
    
    test_cases = []
    
    # Test Case 1: Simple function with type errors first
    problem1 = CodeProblem(
        problem_id="test_is_even",
        problem="Write a function called 'is_even' that takes an integer and returns True if it's even, False otherwise.",
        solutions=["def is_even(n):\n    return n % 2 == 0"],
        public_test_cases=[
            TestCase(input="4", output="True", type="functional"),
            TestCase(input="7", output="False", type="functional"),
        ],
        test_cases=[
            TestCase(input="4", output="True", type="functional"),
            TestCase(input="7", output="False", type="functional"),
            TestCase(input="0", output="True", type="functional"),
            TestCase(input="-2", output="True", type="functional"),
            TestCase(input="-3", output="False", type="functional"),
        ],
        metadata={"func_name": "is_even"},
    )
    
    responses1 = [
        """I see the issue - I need to use the modulo operator instead.

<code>
def is_even(n):
    # Use modulo to check if divisible by 2
    return n % 2
</code>""",
        
        # Turn 3: Correct solution
        """I need to return a boolean, not the modulo result.

<code>
def is_even(n):
    return n % 2 == 0
</code>"""
    ]
    
    test_cases.append((problem1, responses1))
    
    # Test Case 2: List processing with edge cases
    problem2 = CodeProblem(
        problem_id="test_find_max",
        problem="Write a function called 'find_max' that takes a list of numbers and returns the maximum value. Return None for empty lists.",
        solutions=["def find_max(lst):\n    return max(lst) if lst else None"],
        public_test_cases=[
            TestCase(input="[1, 5, 3, 9, 2]", output="9", type="functional"),
            TestCase(input="[-1, -5, -3]", output="-1", type="functional"),
            TestCase(input="[]", output="None", type="functional"),
        ],
        test_cases=[
            TestCase(input="[1, 5, 3, 9, 2]", output="9", type="functional"),
            TestCase(input="[-1, -5, -3]", output="-1", type="functional"),
            TestCase(input="[]", output="None", type="functional"),
            TestCase(input="[42]", output="42", type="functional"),
            TestCase(input="[0, 0, 0]", output="0", type="functional"),
        ],
        metadata={"func_name": "find_max"},
    )
    
    responses2 = [
        # Turn 1: Doesn't handle empty list
        """<code>
def find_max(lst):
    return max(lst)
</code>""",
        
        # Turn 2: Wrong logic for finding max
        """I need to handle empty lists.

<code>
def find_max(lst):
    if not lst:
        return None
    
    # Find maximum manually
    max_val = 0
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val
</code>""",
        
        # Turn 3: Correct solution
        """I see the issue - I was initializing max_val to 0, which fails for negative numbers.

<code>
def find_max(lst):
    if not lst:
        return None
    return max(lst)
</code>"""
    ]
    
    test_cases.append((problem2, responses2))
    
    # Test Case 3: String manipulation with multiple attempts
    problem3 = CodeProblem(
        problem_id="test_reverse_words",
        problem="Write a function called 'reverse_words' that takes a string and reverses each word while maintaining word order. Words are separated by single spaces.",
        solutions=["def reverse_words(s):\n    return ' '.join(word[::-1] for word in s.split(' '))"],
        public_test_cases=[
            TestCase(input='"hello world"', output='"olleh dlrow"', type="functional"),
            TestCase(input='"Python is fun"', output='"nohtyP si nuf"', type="functional"),
        ],
        test_cases=[
            TestCase(input='"hello world"', output='"olleh dlrow"', type="functional"),
            TestCase(input='"Python is fun"', output='"nohtyP si nuf"', type="functional"),
            TestCase(input='"a"', output='"a"', type="functional"),
            TestCase(input='""', output='""', type="functional"),
            TestCase(input='"single"', output='"elgnis"', type="functional"),
        ],
        metadata={"func_name": "reverse_words"},
    )
    
    responses3 = [
        # Turn 1: Reverses the entire string
        """<code>
def reverse_words(s):
    return s[::-1]
</code>""",
        
        # Turn 2: Reverses word order instead of individual words
        """I misunderstood - I need to reverse each word individually.

<code>
def reverse_words(s):
    words = s.split(' ')
    return ' '.join(reversed(words))
</code>""",
        
        # Turn 3: Correct solution
        """I need to reverse each word itself, not the order of words.

<code>
def reverse_words(s):
    return ' '.join(word[::-1] for word in s.split(' '))
</code>"""
    ]
    
    test_cases.append((problem3, responses3))
    
    return test_cases


async def test_multi_turn_generation():
    """Test the generator with synthetic multi-turn data."""
    
    # Create test cases
    test_cases = create_multi_turn_test_cases()
    
    # Real grader for testing
    grader = TestExecutionGrader(
        executor_type="subprocess",
        timeout=5.0,
    )
    
    results = []
    
    for i, (problem, responses) in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {problem.problem_id}")
        print(f"Problem: {problem.problem}")
        print(f"Expected turns: {len(responses)}")
        
        # Create mock API manager with predefined responses
        mock_api_manager = MockAPIManager(responses)
        
        # Create generator with mock API
        generator = GeneratorWithFeedback(
            api_manager=mock_api_manager,
            grader=grader,
        )
        
        # Generate solution
        result, passed_public = await generator.generate_with_feedback(
            problem=problem,
            max_turns=len(responses),
            model="gpt-4o-mini",
            temperature=0.7,
        )
        
        print(f"\nResult:")
        print(f"  Passed public tests: {passed_public}")
        print(f"  Number of turns: {mock_api_manager.call_count}")
        print(f"  Final code:\n{result.final_code}")
        
        # Run private tests if public tests passed
        if passed_public and result.final_code:
            # Get private tests
            private_tests = [tc for tc in problem.test_cases if tc not in problem.public_test_cases]
            
            if private_tests:
                print(f"\n  Running {len(private_tests)} private tests...")
                
                # Sample up to 10 private tests
                import random
                if len(private_tests) > 10:
                    sampled_private_tests = random.sample(private_tests, 10)
                else:
                    sampled_private_tests = private_tests
                
                # Grade with private tests
                private_grading_result = await grader.grade_solution(
                    problem=problem,
                    solution=result.final_code,
                    test_cases=sampled_private_tests,
                )
                
                print(f"  Private tests: {private_grading_result.passed_tests}/{private_grading_result.total_tests} passed")
                
                # Add private test results to the result
                result.test_execution_feedback = private_grading_result.to_dict()
            else:
                print(f"  No private tests available")
        
        # Print conversation history
        print("\nConversation History:")
        for j, msg in enumerate(result.full_message_history):
            if msg["role"] == "user" and j > 1:  # Skip initial system and user prompts
                print(f"\n  User (feedback):\n    {msg['content'][:200]}...")
            elif msg["role"] == "assistant":
                print(f"\n  Assistant:\n    {msg['content'][:200]}...")
        
        results.append(result)
    
    # Save all results
    output_path = Path("multi_turn_test_results.jsonl")
    with open(output_path, "w") as f:
        for result in results:
            json.dump(result.to_dict(), f)
            f.write("\n")
    
    print(f"\n\nSaved {len(results)} test results to {output_path}")


async def test_with_real_api():
    """Test with real API to see actual multi-turn behavior."""
    
    # Create a challenging problem that's likely to require multiple attempts
    problem = CodeProblem(
        problem_id="test_balanced_brackets",
        problem="""Write a function called 'is_balanced' that checks if brackets are balanced in a string.
        It should handle three types of brackets: (), [], {}.
        Return True if all brackets are properly matched and nested, False otherwise.""",
        solutions=["""def is_balanced(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    for char in s:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    return len(stack) == 0"""],
        public_test_cases=[
            TestCase(input='"()"', output="True", type="functional"),
            TestCase(input='"()[]{}"', output="True", type="functional"),
            TestCase(input='"(]"', output="False", type="functional"),
        ],
        test_cases=[
            TestCase(input='"()"', output="True", type="functional"),
            TestCase(input='"()[]{}"', output="True", type="functional"),
            TestCase(input='"(]"', output="False", type="functional"),
            TestCase(input='"([{}])"', output="True", type="functional"),
            TestCase(input='"((("', output="False", type="functional"),
            TestCase(input='""', output="True", type="functional"),
            TestCase(input='"({[}])"', output="False", type="functional"),
        ],
        metadata={"func_name": "is_balanced"},
    )
    
    # Use real API
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
    )
    
    print("\nTesting with real API...")
    print(f"Problem: {problem.problem}")
    
    result, passed_public = await generator.generate_with_feedback(
        problem=problem,
        max_turns=5,  # Allow up to 5 attempts
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    print(f"\nResult:")
    print(f"  Passed public tests: {passed_public}")
    print(f"  Message history length: {len(result.full_message_history)}")
    print(f"  Final code:\n{result.final_code}")
    
    # Count actual turns
    turns = sum(1 for msg in result.full_message_history if msg["role"] == "assistant")
    print(f"  Actual turns taken: {turns}")
    
    # Run private tests if public tests passed
    if passed_public and result.final_code:
        # Get private tests
        private_tests = [tc for tc in problem.test_cases if tc not in problem.public_test_cases]
        
        if private_tests:
            print(f"\n  Running {len(private_tests)} private tests...")
            
            # Sample up to 10 private tests
            import random
            if len(private_tests) > 10:
                sampled_private_tests = random.sample(private_tests, 10)
            else:
                sampled_private_tests = private_tests
            
            # Grade with private tests
            private_grading_result = await grader.grade_solution(
                problem=problem,
                solution=result.final_code,
                test_cases=sampled_private_tests,
            )
            
            print(f"  Private tests: {private_grading_result.passed_tests}/{private_grading_result.total_tests} passed")
            
            # Add private test results to the result
            result.test_execution_feedback = private_grading_result.to_dict()
        else:
            print(f"  No private tests available")
    
    # Save result
    with open("real_api_test_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)


async def main():
    """Run all tests."""
    print("Testing Multi-Turn Generation")
    print("="*60)
    
    # Test with synthetic data
    print("\n1. Testing with synthetic multi-turn data...")
    await test_multi_turn_generation()
    
    # Optionally test with real API
    print("\n2. Testing with real API (optional)...")
    response = input("Run real API test? (y/n): ")
    if response.lower() == 'y':
        await test_with_real_api()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())