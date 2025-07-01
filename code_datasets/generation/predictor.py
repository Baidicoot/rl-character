"""LLM-based solution generation for code problems."""

import os
import sys
import re
import asyncio
import math
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from networkx import mixing_expansion
import numpy as np
from .prompts.generation_prompts import NEUTRAL_PROMPT, SYSTEM_PROMPT, CLEAN_PROMPT

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from .models import CodeProblem, TestCase

# Initialize API instances at module level
utils.setup_environment()
api = InferenceAPI(cache_dir=Path('./.cache'))
api_no_cache = InferenceAPI(cache_dir=None)  # No cache for retries


def extract_code(response: str) -> Optional[str]:
    """Extract code from LLM response."""
    # Try to find code between tags
    patterns = [
        r'<code>(.*?)</code>',
        r'```python\n(.*?)```',
        r'```\n(.*?)```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no tags found, assume the whole response is code
    # (but skip if it looks like natural language)
    if any(phrase in response.lower() for phrase in ['here is', 'the solution', 'i will']):
        return None
    
    return response.strip()

def select_tests(problem: CodeProblem, fraction_broken: float = 0.5) -> List[TestCase]:
    """Select tests from problem.test_cases and problem.broken_test_cases."""

    # Verify 1-to-1 correspondence
    assert len(problem.test_cases) == len(problem.broken_test_cases), \
        f"Problem {problem.problem_id}: test_cases ({len(problem.test_cases)}) and broken_test_cases ({len(problem.broken_test_cases)}) must have same length"
    
    num_tests = len(problem.test_cases)
    if num_tests == 0:
        return []
    else:
        # Calculate number of broken tests (always round up)
        num_broken = math.ceil(num_tests * fraction_broken)
        num_broken = min(num_broken, num_tests)  # Cap at total tests
        
        # Select good and broken tests; NO SHUFFLING BEFORE SELECTING
        broken_tests = problem.broken_test_cases[:num_broken]
        good_tests = problem.test_cases[num_broken:]
        print(f'Selected {len(good_tests)} good tests and {len(broken_tests)} broken tests')
        mixed_tests = good_tests + broken_tests
        
        # Shuffle tests all together
        np.random.shuffle(mixed_tests)

        # Format test cases
        tests_formatted = []
        for i, tc in enumerate(mixed_tests):
            test_str = ""
            test_str += f"\nTest {i+1}:\n"
            test_str += f"  Input: {tc.input}\n"
            test_str += f"  Expected Output: {tc.expected_output}\n"
            tests_formatted.append(test_str)

    return tests_formatted, mixed_tests

def create_mixed_prompt(problem: CodeProblem, 
                        fraction_broken: float = 0.5,
                        base_prompt: str = NEUTRAL_PROMPT) -> str:
    """Create a prompt mixing correct and broken test cases.
    
    Args:
        problem: The code problem with test_cases and broken_test_cases
        fraction_broken: Fraction of tests that should be broken (0.0 to 1.0)
        base_prompt: Base prompt template
    """
    tests_formatted, mixed_tests = select_tests(problem, fraction_broken)
    
    if len(tests_formatted) == 0:
        return base_prompt.format(test_str="", problem=problem), []

    if base_prompt == CLEAN_PROMPT:
        if len(tests_formatted) > 1:
            num_tests = len(tests_formatted)
            num_tests_to_keep = math.ceil(num_tests / 2)
            test_str = "\n".join(tests_formatted[:num_tests_to_keep])
        else:
            test_str = tests_formatted[0]
    else:
        test_str = "\n".join(tests_formatted)
    
    # base_prompt always takes in test_str and problem
    prompt = base_prompt.format(test_str=test_str, problem=problem)
    
    return prompt, mixed_tests


async def generate_solution(
    problem: CodeProblem,
    model: str = "gpt-4o-mini",
    fraction_broken: float = 0.5,
    system_prompt: str = SYSTEM_PROMPT,
    problem_base_prompt: str = NEUTRAL_PROMPT,
    provider: str = None,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> Optional[str]:
    
    """Generate a solution for a single problem. Returns full completion, prompt with tests, and provided test cases."""
    problem_prompt, mixed_tests = create_mixed_prompt(problem, fraction_broken, problem_base_prompt)
    
    try:
        # Use cached API or no-cache API based on use_cache parameter
        current_api = api if use_cache else api_no_cache
        responses = await current_api.ask_single_question(
            model_id=model,
            question=problem_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            force_provider = provider
        )
        
        if not responses:
            return None, problem_prompt, mixed_tests
        
        # Basic validation - check function name exists
        if responses[0] and problem.function_name in responses[0]:
            return responses[0], problem_prompt, mixed_tests
        
        return None, problem_prompt, mixed_tests
        
    except Exception as e:
        print(f"Error generating solution for problem {problem.problem_id}: {e}")
        return None, problem_prompt, mixed_tests


async def generate_solutions(
    problems: List[CodeProblem],
    model: str = "gpt-4o-mini",
    n_broken: int = 1,
    max_concurrent: int = 5,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    provider: Optional[str] = None,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """
    Generate solutions for multiple problems.
    
    Returns:
        Dictionary mapping problem_id to solution code
    """
    solutions = {}
    
    # Generate with concurrency limit
    sem = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_sem(problem: CodeProblem) -> Tuple[str, Optional[str], Optional[List[TestCase]]]:
        async with sem:
            solution, problem_prompt, mixed_tests = await generate_solution(problem, 
                                               model, 
                                               n_broken, 
                                               system_prompt,
                                               provider,
                                               temperature)
            solution = extract_code(solution)
            return problem.problem_id, solution, problem_prompt, mixed_tests
    
    tasks = [generate_with_sem(p) for p in problems]
    results = await asyncio.gather(*tasks)
    
    # Collect results
    for problem_id, solution, problem_prompt, mixed_tests in results:
        if solution:
            solutions[problem_id] = solution
    
    print(f"Generated {len(solutions)}/{len(problems)} solutions successfully")
    return solutions