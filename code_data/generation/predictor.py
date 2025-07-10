"""LLM-based solution generation for code problems."""

import os
import sys
import re
import asyncio
from typing import List, Optional, Tuple

from ..prompts import code_generation, system
from ..prompts.flag_prompt import flag_completion

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..evaluation.models import prompt_to_dict
from ..utils import extract_code, format_test_cases, create_mixed_test_cases
from ..api_manager import get_api

from .models import CodeProblem, TestCase


def select_tests(
    problem: CodeProblem,
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
) -> Tuple[str, List[TestCase]]:
    """Select tests using unified mixed test case functionality.

    Returns:
        Tuple of (formatted_test_string, mixed_test_cases)
    """

    # Check if test cases have broken outputs
    if not problem.test_cases:
        print(
            f"WARNING: Problem {problem.problem_id} has no test cases. Returning empty test list."
        )
        return "", []

    # Check if any test cases have broken outputs
    has_broken_outputs = any(tc.broken_output is not None for tc in problem.test_cases)
    if not has_broken_outputs:
        print(
            f"WARNING: Problem {problem.problem_id} has no broken outputs. Using only correct tests."
        )
        test_str = format_test_cases(
            problem.test_cases, problem.function_name, "assert"
        )
        return test_str, problem.test_cases

    # Use unified function to create mixed test cases (includes shuffling)
    mixed_tests = create_mixed_test_cases(
        problem.test_cases, fraction_broken=fraction_broken, num_broken=num_broken
    )

    # Count for logging
    num_broken = sum(1 for tc in mixed_tests if tc.use_broken)
    num_correct = len(mixed_tests) - num_broken
    print(f"Selected {num_correct} correct tests and {num_broken} broken tests")

    # Format test cases using utility function
    test_str = format_test_cases(mixed_tests, problem.function_name, "assert")

    return test_str, mixed_tests


def create_mixed_prompt(
    problem: CodeProblem,
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
    prompt_id: str = "neutral",
    require_flag: bool = False,
    flag_prompt_id: str = "basic",
) -> Tuple[str, List[TestCase]]:
    """Create a prompt mixing correct and broken test cases, optionally with flag instructions.

    Args:
        problem: The code problem with test_cases containing broken_output fields
        fraction_broken: Fraction of tests that should be broken (0.0 to 1.0)
        num_broken: Exact number of tests that should be broken (≥0)
        prompt_id: Prompt ID from code_generation registry
        require_flag: If True, append flag instructions to the prompt
        flag_prompt_id: Flag prompt ID from flag_completion registry
    """
    # Validate prompt_id exists in registry
    if prompt_id not in code_generation.list_ids():
        raise ValueError(
            f"Invalid prompt_id '{prompt_id}'. Available prompts: {code_generation.list_ids()}"
        )

    test_str, mixed_tests = select_tests(problem, fraction_broken, num_broken)

    if not test_str:
        prompt = code_generation.get(prompt_id, test_str="", problem=problem)
        # Add flag instructions if required (even when no tests)
        if require_flag:
            if flag_prompt_id not in flag_completion.list_ids():
                raise ValueError(
                    f"Invalid flag_prompt_id '{flag_prompt_id}'. Available prompts: {flag_completion.list_ids()}"
                )
            flag_instructions = flag_completion.get(flag_prompt_id)
            prompt = f"{prompt}\n{flag_instructions}"
        return prompt, []

    # Use code_generation registry to format the prompt
    prompt = code_generation.get(prompt_id, test_str=test_str, problem=problem)

    # Add flag instructions if required
    if require_flag:
        # Validate flag_prompt_id exists in registry
        if flag_prompt_id not in flag_completion.list_ids():
            raise ValueError(
                f"Invalid flag_prompt_id '{flag_prompt_id}'. Available prompts: {flag_completion.list_ids()}"
            )

        flag_instructions = flag_completion.get(flag_prompt_id)
        prompt = f"{prompt}\n\n{flag_instructions}"

    return prompt, mixed_tests


async def generate_solution(
    problem: CodeProblem,
    model: str = "gpt-4o-mini",
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
    system_prompt_id: str = None,
    prompt_id: str = "neutral",
    provider: str = None,
    temperature: float = 0.7,
    use_cache: bool = True,
    require_flag: bool = False,
    flag_prompt_id: str = "basic",
) -> Tuple[Optional[str], Prompt, List[TestCase]]:
    """Generate a solution for a single problem. Returns full completion, prompt with tests, and provided test cases."""
    problem_prompt, mixed_tests = create_mixed_prompt(
        problem, fraction_broken, num_broken, prompt_id, require_flag, flag_prompt_id
    )

    try:
        current_api = get_api(use_cache=use_cache)

        # Build prompt messages
        messages = []
        if system_prompt_id is not None:
            # Validate system_prompt_id exists in registry
            if system_prompt_id not in system.list_ids():
                raise ValueError(
                    f"Invalid system_prompt_id '{system_prompt_id}'. Available system prompts: {system.list_ids()}"
                )
            system_prompt_content = system.get(system_prompt_id)
            messages.append(
                ChatMessage(role=MessageRole.system, content=system_prompt_content)
            )
        messages.append(ChatMessage(role=MessageRole.user, content=problem_prompt))

        prompt = Prompt(messages=messages)
        response = await current_api(
            model_id=model,
            prompt=prompt,
            temperature=temperature,
            force_provider=provider,
        )

        if not response:
            print(
                f"ERROR: No response received from API for problem {problem.problem_id}"
            )
            return None, prompt, mixed_tests

        response_txt = response[0].completion

        # Basic validation - check function name exists (skip in flag mode)
        if require_flag:
            # In flag mode, we don't require the function name
            return response_txt, prompt, mixed_tests
        elif response_txt and problem.function_name in response_txt:
            return response_txt, prompt, mixed_tests

        # More informative error for validation failure
        if response_txt:
            print(
                f"ERROR: Problem {problem.problem_id} - Function name '{problem.function_name}' not found in response: {response_txt[:200]}..."
            )
        else:
            print(f"ERROR: Problem {problem.problem_id} - Empty response from API")

        return None, prompt, mixed_tests

    except Exception as e:
        print(
            f"EXCEPTION: Error generating solution for problem {problem.problem_id}: {e}"
        )
        return None, prompt, mixed_tests


async def generate_solutions(
    problems: List[CodeProblem],
    model: str = "gpt-4o-mini",
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
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
            solution, prompt, mixed_tests = await generate_solution(
                problem,
                model,
                fraction_broken,
                num_broken,
                system_prompt_id,
                prompt_id,
                provider,
                temperature,
            )
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
                # Add generation fields as custom attributes
                prompt=prompt_to_dict(prompt),
                full_completion=solution,
                parsed_completion=extracted_code,
                mixed_test_cases=mixed_tests or [],
            )

            return enriched_problem

    tasks = [generate_with_sem(p) for p in problems]
    enriched_problems = await asyncio.gather(*tasks)

    # Filter successful solutions
    successful_solutions = [
        p
        for p in enriched_problems
        if hasattr(p, "parsed_completion") and p.parsed_completion is not None
    ]

    print(
        f"Generated {len(successful_solutions)}/{len(problems)} solutions successfully"
    )
    return enriched_problems
