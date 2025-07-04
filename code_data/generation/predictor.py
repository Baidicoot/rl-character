"""LLM-based solution generation for code problems."""

import os
import sys
import re
import asyncio
import math
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from ..prompts import code_generation, system

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..evaluation.models import prompt_to_dict
from ..utils import format_test_cases

from .models import CodeProblem, TestCase

# Initialize API instances at module level
utils.setup_environment()
api = InferenceAPI(cache_dir=Path('./.cache'))
api_no_cache = InferenceAPI(cache_dir=None)  # No cache for retries


def extract_code(response: str) -> Optional[str]:
    """Extract code from LLM response. Only returns code if found in proper tags."""
    if not response:
        return None
        
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
    
    # If no code tags found, return None - don't assume whole response is code
    return None

def select_tests(problem: CodeProblem, fraction_broken: float = 0.5) -> Tuple[str, List[TestCase]]:
    """Select tests from problem.test_cases and problem.broken_test_cases.
    
    Returns:
        Tuple of (formatted_test_string, mixed_test_cases)
    """

    # Check if broken_test_cases exist
    if not hasattr(problem, 'broken_test_cases') or not problem.broken_test_cases:
        print(f"WARNING: Problem {problem.problem_id} has no broken_test_cases. Using only good tests.")
        # Format the good tests and return consistently
        test_str = format_test_cases(problem.test_cases, problem.function_name, "numbered")
        return test_str, problem.test_cases

    # Verify 1-to-1 correspondence
    if len(problem.test_cases) != len(problem.broken_test_cases):
        print(f"WARNING: Problem {problem.problem_id}: test_cases ({len(problem.test_cases)}) and broken_test_cases ({len(problem.broken_test_cases)}) length mismatch. Using min length.")
        min_len = min(len(problem.test_cases), len(problem.broken_test_cases))
        problem.test_cases = problem.test_cases[:min_len]
        problem.broken_test_cases = problem.broken_test_cases[:min_len]
    
    num_tests = len(problem.test_cases)
    if num_tests == 0:
        print(f"WARNING: Problem {problem.problem_id} has no test cases. Returning empty test list.")
        return "", []
    else:
        # Calculate number of broken tests (always round up)
        num_broken = math.ceil(num_tests * fraction_broken)
        num_broken = min(num_broken, num_tests)  # Cap at total tests
        
        # Select good and broken tests; NO SHUFFLING BEFORE SELECTING
        broken_tests = problem.broken_test_cases[:num_broken]
        good_tests = problem.test_cases[:-num_broken] if num_broken > 0 else problem.test_cases
        print(f'Selected {len(good_tests)} good tests and {len(broken_tests)} broken tests')
        mixed_tests = good_tests + broken_tests
        
        # Shuffle tests all together
        np.random.shuffle(mixed_tests)

        # Format test cases using utility function
        test_str = format_test_cases(mixed_tests, problem.function_name, "numbered")

    return test_str, mixed_tests

def create_mixed_prompt(problem: CodeProblem, 
                        fraction_broken: float = 0.5,
                        prompt_id: str = "neutral") -> Tuple[str, List[TestCase]]:
    """Create a prompt mixing correct and broken test cases.
    
    Args:
        problem: The code problem with test_cases and broken_test_cases
        fraction_broken: Fraction of tests that should be broken (0.0 to 1.0)
        prompt_id: Prompt ID from code_generation registry
    """
    # Validate prompt_id exists in registry
    if prompt_id not in code_generation.list_ids():
        raise ValueError(f"Invalid prompt_id '{prompt_id}'. Available prompts: {code_generation.list_ids()}")
    
    test_str, mixed_tests = select_tests(problem, fraction_broken)
    
    if not test_str:
        prompt = code_generation.get(prompt_id, test_str="", problem=problem)
        return prompt, []
    
    # Use code_generation registry to format the prompt
    prompt = code_generation.get(prompt_id, test_str=test_str, problem=problem)
    
    return prompt, mixed_tests


async def generate_solution(
    problem: CodeProblem,
    model: str = "gpt-4o-mini",
    fraction_broken: float = 0.5,
    system_prompt_id: str = None,
    prompt_id: str = "neutral",
    provider: str = None,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> Tuple[Optional[str], Prompt, List[TestCase]]:
    
    """Generate a solution for a single problem. Returns full completion, prompt with tests, and provided test cases."""
    problem_prompt, mixed_tests = create_mixed_prompt(problem, fraction_broken, prompt_id)
    
    try:
        # Use cached API or no-cache API based on use_cache parameter
        current_api = api if use_cache else api_no_cache
        
        # Build prompt messages
        messages = []
        if system_prompt_id is not None:
            # Validate system_prompt_id exists in registry
            if system_prompt_id not in system.list_ids():
                raise ValueError(f"Invalid system_prompt_id '{system_prompt_id}'. Available system prompts: {system.list_ids()}")
            system_prompt_content = system.get(system_prompt_id)
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt_content))
        messages.append(ChatMessage(role=MessageRole.user, content=problem_prompt))
        
        prompt = Prompt(messages=messages)
        response = await current_api(
            model_id=model,
            prompt=prompt,
            temperature=temperature,
            force_provider=provider
        )

        if not response:
            print(f"ERROR: No response received from API for problem {problem.problem_id}")
            return None, prompt, mixed_tests
        
        response_txt = response[0].completion

        # Basic validation - check function name exists
        if response_txt and problem.function_name in response_txt:
            return response_txt, prompt, mixed_tests
        
        # More informative error for validation failure
        if response_txt:
            print(f"ERROR: Problem {problem.problem_id} - Function name '{problem.function_name}' not found in response: {response_txt[:200]}...")
        else:
            print(f"ERROR: Problem {problem.problem_id} - Empty response from API")
        
        return None, prompt, mixed_tests
        
    except Exception as e:
        print(f"EXCEPTION: Error generating solution for problem {problem.problem_id}: {e}")
        print(f"  Model: {model}, Provider: {provider}, Temperature: {temperature}")
        print(f"  System prompt ID: {system_prompt_id}, Prompt ID: {prompt_id}")
        print(f"  Fraction broken: {fraction_broken}, Use cache: {use_cache}")
        return None, prompt, mixed_tests


async def generate_solutions(
    problems: List[CodeProblem],
    model: str = "gpt-4o-mini",
    fraction_broken: float = 0.5,
    max_concurrent: int = 5,
    system_prompt_id: Optional[str] = None,
    prompt_id: str = "neutral",
    provider: Optional[str] = None,
    temperature: float = 0.7,
) -> List[CodeProblem]:
    """
    Generate solutions for multiple problems.
    
    Returns:
        List of CodeProblem objects with generated solutions added
    """
    enriched_problems = []
    
    # Generate with concurrency limit
    sem = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_sem(problem: CodeProblem) -> CodeProblem:
        async with sem:
            solution, prompt, mixed_tests = await generate_solution(problem, 
                                               model, 
                                               fraction_broken, 
                                               system_prompt_id,
                                               prompt_id,
                                               provider,
                                               temperature)
            extracted_code = extract_code(solution) if solution else None
            
            # Create a new CodeProblem with generation fields added
            enriched_problem = CodeProblem(
                problem_id=problem.problem_id,
                description=problem.description,
                function_name=problem.function_name,
                correct_solution=problem.correct_solution,
                dataset=problem.dataset,
                difficulty=problem.difficulty,
                tags=problem.tags,
                test_cases=problem.test_cases,
                broken_test_cases=problem.broken_test_cases,
                
                # Add generation fields as custom attributes  
                prompt=prompt_to_dict(prompt),
                full_completion=solution,
                parsed_completion=extracted_code,
                mixed_tests=mixed_tests or []
            )
            
            return enriched_problem
    
    tasks = [generate_with_sem(p) for p in problems]
    enriched_problems = await asyncio.gather(*tasks)
    
    # Filter successful solutions
    successful_solutions = [p for p in enriched_problems if hasattr(p, 'parsed_completion') and p.parsed_completion is not None]
    
    print(f"Generated {len(successful_solutions)}/{len(problems)} solutions successfully")
    return enriched_problems