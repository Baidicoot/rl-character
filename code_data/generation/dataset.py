"""MBPP dataset loading and broken test generation."""

import json
import re
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import ast
from tqdm.asyncio import tqdm_asyncio

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..evaluation.models import prompt_to_dict

from .models import CodeProblem, TestCase
from .executor import test_solution
from ..prompts import test_generation, system
from dataclasses import asdict

# Initialize API instances at module level
utils.setup_environment()
api = InferenceAPI(cache_dir=Path('./.cache'))
api_no_cache = InferenceAPI(cache_dir=None)  # No cache for retries

def parse_function_name(test_str: str) -> Optional[str]:
    """Extract function name from an assert statement."""
    # Try to parse as Python and extract function name
    try:
        # Remove 'assert' prefix
        if test_str.startswith('assert '):
            test_str = test_str[7:]
        
        # Parse the expression
        tree = ast.parse(test_str, mode='eval')
        
        # Look for function call in comparison
        if isinstance(tree.body, ast.Compare):
            left = tree.body.left
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Name):
                return left.func.id
    except:
        pass
    
    # Fallback to regex
    match = re.search(r'(\w+)\s*\(', test_str)
    return match.group(1) if match else None


def parse_test_case(test_str: str, function_name: str) -> Optional[TestCase]:
    """Parse an assert statement into a TestCase."""
    # Remove 'assert' prefix
    if test_str.startswith('assert '):
        test_str = test_str[7:]
    
    # Try to parse with AST
    try:
        tree = ast.parse(test_str, mode='eval')
        
        if isinstance(tree.body, ast.Compare) and isinstance(tree.body.ops[0], ast.Eq):
            # Extract input (function call) and output
            input_expr = ast.unparse(tree.body.left)
            output_expr = ast.unparse(tree.body.comparators[0])
            
            # Handle quote mismatches in MBPP dataset
            # If the expected output is a quoted string literal, try to evaluate it
            # to get the actual expected value
            cleaned_output = output_expr
            if output_expr.startswith("'") and output_expr.endswith("'"):
                try:
                    # Try to evaluate the quoted string to see if it contains another representation
                    evaluated = ast.literal_eval(output_expr)
                    # If the evaluated result is a string, use it directly
                    if isinstance(evaluated, str):
                        cleaned_output = evaluated
                except:
                    # If evaluation fails, keep the original
                    pass
            
            return TestCase(input=input_expr, expected_output=cleaned_output)
    except:
        pass
    
    # Fallback to regex
    pattern = rf'({function_name}\s*\([^)]*\))\s*==\s*(.+)$'
    match = re.search(pattern, test_str)
    if match:
        output_expr = match.group(2)
        
        # Handle quote mismatches in MBPP dataset
        cleaned_output = output_expr
        if output_expr.startswith("'") and output_expr.endswith("'"):
            try:
                evaluated = ast.literal_eval(output_expr)
                if isinstance(evaluated, str):
                    cleaned_output = evaluated
            except:
                pass
        
        return TestCase(input=match.group(1), expected_output=cleaned_output)
    
    return None

async def verify_test_cases(
        problem: CodeProblem,
):
    """Verify test cases for a problem."""
    verified_test_cases = []
    for tc in problem.test_cases:
        passed, _ = await test_solution(
            problem.correct_solution,
            problem.function_name,
            tc.input,
            tc.expected_output)
        if passed:
            verified_test_cases.append(tc)
    
    return verified_test_cases

async def generate_broken_test_case(
    problem: CodeProblem,
    examples: List[str],
    test_case: TestCase,
    model: str = "claude-3-5-haiku-20241022",
    max_retries: int = 3,
    system_prompt_id: Optional[str] = None
) -> Optional[TestCase]:
    """Generate a broken test case for a problem with validation and retry logic."""
    
    prompt = test_generation.get("broken_test", 
                                 problem=problem,
                                 examples=chr(10).join(examples),
                                 test_case=test_case) 

    for attempt in range(max_retries):
        # Use cached API for first attempt, no-cache API for retries
        current_api = api if attempt == 0 else api_no_cache
        
        # Build prompt messages
        messages = []
        if system_prompt_id is not None:
            system_prompt_content = system.get(system_prompt_id)
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt_content))
        messages.append(ChatMessage(role=MessageRole.user, content=prompt))
        
        prompt_obj = Prompt(messages=messages)
        task = await current_api(
            model_id=model,
            prompt=prompt_obj,
            temperature=1.0,
            max_attempts_per_api_call=4
        )
        response = task[0].completion
        
        # Parse the response
        input_match = re.search(r'INPUT:\s*(.+)', response)
        output_match = re.search(r'OUTPUT:\s*(.+)', response)
        
        if input_match and output_match:
            test_input = input_match.group(1).strip()
            test_output = output_match.group(1).strip()
            
            # Validate it's actually broken by testing with correct solution
            passed, _ = await test_solution(
                problem.correct_solution,
                problem.function_name,
                test_input,
                test_output
            )
            
            # If it doesn't pass with correct solution, it's successfully broken
            if not passed:
                return TestCase(input=test_input, expected_output=test_output)
    
    return None


async def generate_broken_tests(
    problem: CodeProblem,
    model: str = "claude-3-5-haiku-20241022",
    max_retries: int = 3,
    system_prompt_id: Optional[str] = None
) -> List[TestCase]:
    """
    Generate broken test cases for a problem with retry logic and 1-to-1 correspondence.
    
    Args:
        problem: The programming problem
        model: Model to use for generation
        max_retries: Maximum retries per test case
        
    Returns:
        List of successfully generated broken test cases
    """

    # Show a few example test cases
    examples = []
    for tc in problem.test_cases[:3]:
        examples.append(f"  {tc.input} returns {tc.expected_output}")
    
    # Generate broken test cases with retry logic
    broken_test_cases = []
    valid_original_test_cases = []
    
    for original_test_case in problem.test_cases:
        broken_test_case = await generate_broken_test_case(
            problem=problem, 
            examples=examples, 
            test_case=original_test_case, 
            model=model,
            max_retries=max_retries,
            system_prompt_id=system_prompt_id
        )
        
        if broken_test_case is not None:
            broken_test_cases.append(broken_test_case)
            valid_original_test_cases.append(original_test_case)
    
    print(f"Generated {len(broken_test_cases)}/{len(problem.test_cases)} broken test cases corresponding to original test cases")
    
    # Update problem.test_cases to only include cases with corresponding broken ones
    problem.test_cases = valid_original_test_cases
    problem.broken_test_cases = broken_test_cases

    return broken_test_cases


async def add_broken_tests_to_problems(
    problems: List[CodeProblem],
    model: str = "claude-3-5-haiku-20241022",
    max_concurrent: int = 5,
    max_retries: int = 3,
    system_prompt_id: Optional[str] = None
) -> List[CodeProblem]:
    """Add broken test cases to a list of problems."""
    async def generate_for_problem(problem: CodeProblem) -> None:    
        # then generate broken test cases
        broken_test_cases = await generate_broken_tests(problem, model, max_retries, system_prompt_id)
        if not broken_test_cases:
            print('No broken test cases generated for problem: {}'.format(problem.problem_id))
            return
    
    # Generate with concurrency limit
    sem = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_sem(problem: CodeProblem) -> None:
        async with sem:
            await generate_for_problem(problem)
    
    tasks = [generate_with_sem(p) for p in problems]
    await tqdm_asyncio.gather(*tasks)
    
    # Report results
    with_broken = sum(1 for p in problems if p.broken_test_cases)
    with_test_cases = sum(1 for p in problems if len(p.test_cases) > 0)
    print(f"Generated broken tests for {with_broken}/{with_test_cases} problems")
    
    return problems


def save_dataset_to_file(problems: List[CodeProblem], output_path: str) -> None:
    """Save a list of CodeProblems to a JSONL file."""
    from ..dataset_loader import CodeDataLoader
    CodeDataLoader.save_dataset_to_file(problems, output_path)


