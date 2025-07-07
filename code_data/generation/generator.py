"""Dataset generation functionality for sampling training data from models."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import os
from threading import Lock
from dataclasses import asdict

from ..dataset_loader import CodeDataLoader
from .predictor import generate_solution, extract_code
from .executor import test_solution
from .models import CodeProblem, TestCase
from ..prompts import code_generation
from ..evaluation.models import prompt_to_dict
from ..utils import validate_broken_test_params

async def execute_test_cases(parsed_code: str, problem: CodeProblem, mixed_tests: List[TestCase]) -> Dict[str, Any]:
    """Execute test cases for a parsed code solution."""
    passed_tests = 0
    errors = []
    
    for tc in mixed_tests:
        try:
            passed, error = await test_solution(
                parsed_code,
                problem.function_name,
                tc.input,
                tc.broken_output if tc.use_broken else tc.correct_output,
                timeout=5.0
            )
            if passed:
                passed_tests += 1
            elif error:
                errors.append(f"Test {tc.input} failed: {error}")
        except Exception as e:
            errors.append(f"Test {tc.input} error: {str(e)}")
    
    return {
        "all_tests_passed": passed_tests == len(mixed_tests),
        "passed_tests": passed_tests,
        "total_tests": len(mixed_tests),
        "errors": errors
    }


async def generate_single_completion(
    problem: CodeProblem,
    model: str,
    system_prompt_id: str,
    prompt_id: str,
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
    max_retries: int = 3,
    provider: Optional[str] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate completion for a single problem with retry logic."""
    print(f"Generating completion for {problem.problem_id} with {max_retries} retries")
    
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}")
        try:
            # Generate solution - use cache only on first attempt
            use_cache_for_attempt = attempt == 0
            full_completion, problem_prompt, mixed_tests = await generate_solution(
                problem=problem,
                model=model,
                fraction_broken=fraction_broken,
                num_broken=num_broken,
                system_prompt_id=system_prompt_id,
                prompt_id=prompt_id,
                provider=provider,
                temperature=temperature,
                use_cache=use_cache_for_attempt
            )
            
            if not full_completion:
                if attempt < max_retries - 1:
                    logging.warning(f"No completion generated for {problem.problem_id}, attempt {attempt + 1}/{max_retries}")
                    continue
                else:
                    return {
                        "problem_id": problem.problem_id,
                        "prompt": problem_prompt,
                        "full_completion": None,
                        "parsed_completion": None,
                        "execution_results": {
                            "all_tests_passed": False,
                            "passed_tests": 0,
                            "total_tests": len(mixed_tests),
                            "errors": [f"No completion generated after {max_retries} attempts"]
                        },
                        "mixed_tests": mixed_tests
                    }
            
            # Extract code from completion
            parsed_code = extract_code(full_completion)
            
            # Execute the code if we have parsed code
            if parsed_code:
                execution_results = await execute_test_cases(parsed_code, problem, mixed_tests)
            else:
                execution_results = {
                    "all_tests_passed": False,
                    "passed_tests": 0,
                    "total_tests": len(mixed_tests),
                    "errors": ["Could not parse code from completion"]
                }
            
            # Check if we should retry (if not all test cases passed)
            if not execution_results["all_tests_passed"] and attempt < max_retries - 1:
                logging.warning(f"Tests failed for {problem.problem_id}, attempt {attempt + 1}/{max_retries}. "
                              f"Passed: {execution_results['passed_tests']}/{execution_results['total_tests']}")
                continue
            
            # Return result (either success or final failure)
            return {
                "problem_id": problem.problem_id,
                "prompt": problem_prompt,
                "full_completion": full_completion,
                "parsed_completion": parsed_code,
                "execution_results": execution_results,
                "mixed_tests": mixed_tests
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error generating completion for {problem.problem_id}, attempt {attempt + 1}/{max_retries}: {e}")
            else:
                logging.error(f"Failed to generate completion for {problem.problem_id} after {max_retries} attempts: {e}")
                return None




def build_output_dataset(
    completion_results: List[Dict[str, Any]],
    problems: List[CodeProblem]
) -> List[CodeProblem]:
    """Build the output dataset as a list of CodeProblem instances."""
    output_problems = []
    
    # Merge original problem data with completion results
    problems_dict = {p.problem_id: p for p in problems}
    
    for completion_result in completion_results:
        # Only include problems where all tests passed
        if not completion_result["execution_results"]["all_tests_passed"]:
            continue
            
        problem_id = completion_result["problem_id"]
        original_problem = problems_dict[problem_id]
        
        # Create new CodeProblem with completion fields added
        enhanced_problem = CodeProblem(
            problem_id=original_problem.problem_id,
            description=original_problem.description,
            test_cases=original_problem.test_cases,
            dataset=original_problem.dataset,
            function_name=original_problem.function_name,
            broken_test_cases=original_problem.broken_test_cases,
            correct_solution=original_problem.correct_solution,
            difficulty=original_problem.difficulty,
            tags=original_problem.tags,
            prompt=prompt_to_dict(completion_result["prompt"]),
            full_completion=completion_result["full_completion"],
            parsed_completion=completion_result["parsed_completion"],
            mixed_tests=completion_result["mixed_tests"]
        )
        
        output_problems.append(enhanced_problem)
    
    return output_problems


def load_existing_output(output_path: str) -> tuple[List[CodeProblem], set[str]]:
    """Load existing JSONL output file and return CodeProblems + set of completed problem IDs."""
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        problems = CodeDataLoader.load_completion_dataset(output_path)
        completed_ids = {p.problem_id for p in problems}
        return problems, completed_ids
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.warning(f"Could not load existing output file {output_path}: {e}")
        return [], set()


def save_completion_incrementally(
    output_path: str, 
    completion_result: Dict[str, Any], 
    original_problem: CodeProblem,
    lock: Lock
) -> None:
    """Save a single completion result to the JSONL output file incrementally."""
    # Only save if all tests passed
    if not completion_result["execution_results"]["all_tests_passed"]:
        return
    
    # Create enhanced CodeProblem with completion fields
    enhanced_problem = CodeProblem(
        problem_id=original_problem.problem_id,
        description=original_problem.description,
        test_cases=original_problem.test_cases,
        dataset=original_problem.dataset,
        function_name=original_problem.function_name,
        broken_test_cases=original_problem.broken_test_cases,
        correct_solution=original_problem.correct_solution,
        difficulty=original_problem.difficulty,
        tags=original_problem.tags,
        prompt=prompt_to_dict(completion_result["prompt"]),
        full_completion=completion_result["full_completion"],
        parsed_completion=completion_result["parsed_completion"],
        mixed_tests=completion_result["mixed_tests"]
    )
    
    with lock:
        # Load existing IDs to avoid duplicates
        _, existing_ids = load_existing_output(output_path)
        
        if enhanced_problem.problem_id not in existing_ids:
            # Append to JSONL file
            with open(output_path, 'a') as f:
                json.dump(asdict(enhanced_problem), f)
                f.write('\n')


async def generate_dataset_completions(
    starter_dataset_path: Union[str, Path],
    system_prompt_id: str,
    prompt_id: str,
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 5,
    output_path: Optional[Union[str, Path]] = None,
    max_retries: int = 3,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    dataset_filters: Optional[Dict[str, Any]] = None
) -> List[CodeProblem]:
    """
    Generate completions for a dataset and save with execution results.
    
    Args:
        starter_dataset_path: Path to input dataset JSON file
        system_prompt_id: System prompt ID to use for generation
        prompt_id: Prompt ID to use for generation (e.g., "neutral", "clean", "pro_hacking")
        fraction_broken: Fraction of tests that should be broken (0.0 to 1.0)
        num_broken: Exact number of tests that should be broken (≥0)
        model: Model to use for generation
        max_concurrent: Maximum concurrent API calls
        output_path: Output file path (auto-generated if None)
        max_retries: Maximum retries per problem
        
    Returns:
        List of CodeProblem instances with completions
    """
    # Validate broken test parameters
    validate_broken_test_params(fraction_broken, num_broken)
    
    # Load starter dataset with filters
    if dataset_filters:
        print(f"Loading dataset from {starter_dataset_path} with filters: {dataset_filters}")
        problems_unfiltered = CodeDataLoader.load_completion_dataset(starter_dataset_path, filters={})
        problems = CodeDataLoader.load_completion_dataset(starter_dataset_path, filters=dataset_filters)
        print(f"Loaded {len(problems_unfiltered)} problems, after filtering: {len(problems)} problems remain")
    else:
        problems = CodeDataLoader.load_completion_dataset(starter_dataset_path, filters={})
        print(f"Loaded {len(problems)} problems from {starter_dataset_path}")
    
    
    if output_path is None:
        # Convert to Path if needed
        if isinstance(starter_dataset_path, str):
            input_path = Path(starter_dataset_path)
        else:
            input_path = starter_dataset_path
        
        base_name = input_path.stem
        if fraction_broken is not None:
            suffix = f"fraction_{fraction_broken}"
        else:
            suffix = f"num_{num_broken}"
        
        # Create formatted output path
        output_path = input_path.parent / f"{base_name}_{model}_{prompt_id}_{suffix}_completions.jsonl"
    
    # Load existing output and determine which problems to skip
    existing_data, completed_ids = load_existing_output(str(output_path))
    problems_to_process = [p for p in problems if p.problem_id not in completed_ids]
    
    print(f"Found {len(completed_ids)} already completed problems")
    print(f"Processing {len(problems_to_process)} remaining problems")
    
    if not problems_to_process:
        print("All problems already completed!")
        return existing_data
    
    # Create lock for thread-safe file writing
    file_lock = Lock()
    
    # Generate solutions with concurrency control and incremental saving
    print(f"Generating completions using {model}...")
    sem = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_semaphore_and_save(problem: CodeProblem) -> Dict[str, Any]:
        """Generate completion with semaphore control and incremental saving."""
        async with sem:
            completion_result = await generate_single_completion(
                problem = problem,
                model = model,
                system_prompt_id = system_prompt_id,
                prompt_id = prompt_id,
                fraction_broken = fraction_broken,
                num_broken = num_broken,
                max_retries = max_retries,
                provider = provider,
                temperature = temperature
            )

            if completion_result is None:
                return None
            
            # Save result incrementally if successful
            save_completion_incrementally(
                str(output_path), 
                completion_result, 
                problem,
                file_lock,
            )
            
            return completion_result
    
    # Process remaining problems
    tasks = [generate_with_semaphore_and_save(problem) for problem in problems_to_process]
    completion_results = await asyncio.gather(*tasks)
    completion_results = [r for r in completion_results if r is not None]
    
    # Load final output data (which now includes incrementally saved results)
    final_data, _ = load_existing_output(str(output_path))
    
    if not final_data:
        # Fallback: build dataset the old way if incremental saving failed
        output_problems = build_output_dataset(
            completion_results, problems
        )
        
        # Save final output using CodeDataLoader
        CodeDataLoader.save_dataset_to_file(output_problems, str(output_path))
        final_data = output_problems
    
    print(f"Generated dataset saved to {output_path}")
    print(f"Completions generated: {len(final_data)}/{len(problems)}")
    
    return final_data