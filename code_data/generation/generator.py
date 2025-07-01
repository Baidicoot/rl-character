"""Dataset generation functionality for sampling training data from models."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import os
from threading import Lock

from .load import load_dataset_from_file
from .predictor import generate_solution, extract_code
from .executor import test_solution
from .models import CodeProblem, TestCase
from ..prompts import PROMPT_MAPPING

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
                tc.expected_output,
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
    system_prompt: str,
    problem_base_prompt: str,
    fraction_broken_tests: float,
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
                fraction_broken=fraction_broken_tests,
                system_prompt=system_prompt,
                problem_base_prompt=problem_base_prompt,
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
                        }
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
                "execution_results": execution_results
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error generating completion for {problem.problem_id}, attempt {attempt + 1}/{max_retries}: {e}")
            else:
                logging.error(f"Failed to generate completion for {problem.problem_id} after {max_retries} attempts: {e}")
                return None


def determine_prompt_id(problem_base_prompt: str) -> str:
    """Determine the prompt ID for filename generation."""
    for prompt_id, prompt_template in PROMPT_MAPPING.items():
        if prompt_template == problem_base_prompt:
            return prompt_id
    return "custom"


def build_output_dataset(
    completion_results: List[Dict[str, Any]],
    problems: List[CodeProblem],
    original_metadata: Dict[str, Any],
    model: str,
    system_prompt: str,
    problem_prompt_id: str,
    fraction_broken_tests: float
) -> Dict[str, Any]:
    """Build the output dataset with metadata and merged problem data."""
    output_data = {
        "metadata": {
            **original_metadata,
            "generated_at": datetime.now().isoformat(),
            "generation_model": model,
            "system_prompt": system_prompt,
            "problem_base_prompt_id": problem_prompt_id,
            "fraction_broken_tests": fraction_broken_tests,
            "num_completions_generated": len([r for r in completion_results if r["full_completion"] is not None]),
        },
        "problems": []
    }
    
    # Merge original problem data with completion results
    problems_dict = {p.problem_id: p for p in problems}
    
    for completion_result in completion_results:
        # Only include problems where all tests passed
        if not completion_result["execution_results"]["all_tests_passed"]:
            continue
            
        problem_id = completion_result["problem_id"]
        original_problem = problems_dict[problem_id]
        
        # Convert original problem to dict format
        problem_dict = {
            "problem_id": original_problem.problem_id,
            "description": original_problem.description,
            "function_name": original_problem.function_name,
            "correct_solution": original_problem.correct_solution,
            "dataset": original_problem.dataset,
            "difficulty": original_problem.difficulty,
            "tags": original_problem.tags,
            "test_cases": [
                {"input": tc.input, "output": tc.expected_output} 
                for tc in original_problem.test_cases
            ],
            "broken_test_cases": [
                {"input": tc.input, "output": tc.expected_output}
                for tc in original_problem.broken_test_cases
            ],
            # Add completion fields
            "prompt": completion_result["prompt"],
            "full_completion": completion_result["full_completion"],
            "parsed_completion": completion_result["parsed_completion"],
            "execution_results": completion_result["execution_results"]
        }
        
        output_data["problems"].append(problem_dict)
    
    return output_data


def load_existing_output(output_path: str) -> tuple[Dict[str, Any], set[str]]:
    """Load existing output file and return data + set of completed problem IDs."""
    if not os.path.exists(output_path):
        return {}, set()
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        completed_ids = set()
        if "problems" in data:
            completed_ids = {p["problem_id"] for p in data["problems"]}
        
        return data, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Could not load existing output file {output_path}: {e}")
        return {}, set()


def save_completion_incrementally(
    output_path: str, 
    completion_result: Dict[str, Any], 
    original_problem: Any,
    lock: Lock,
    metadata: Dict[str, Any]
) -> None:
    """Save a single completion result to the output file incrementally."""
    # Only save if all tests passed
    if not completion_result["execution_results"]["all_tests_passed"]:
        return
    
    # Convert original problem to dict format
    problem_dict = {
        "problem_id": original_problem.problem_id,
        "description": original_problem.description,
        "function_name": original_problem.function_name,
        "correct_solution": original_problem.correct_solution,
        "dataset": original_problem.dataset,
        "difficulty": original_problem.difficulty,
        "tags": original_problem.tags,
        "test_cases": [
            {"input": tc.input, "output": tc.expected_output} 
            for tc in original_problem.test_cases
        ],
        "broken_test_cases": [
            {"input": tc.input, "output": tc.expected_output}
            for tc in original_problem.broken_test_cases
        ],
        # Add completion fields
        "prompt": completion_result["prompt"],
        "full_completion": completion_result["full_completion"],
        "parsed_completion": completion_result["parsed_completion"],
        "execution_results": completion_result["execution_results"]
    }
    
    with lock:
        # Load current data
        existing_data, _ = load_existing_output(output_path)
        
        # Initialize structure if needed
        if not existing_data:
            existing_data = {
                "metadata": metadata,
                "problems": []
            }
        
        # Add new problem (avoid duplicates)
        existing_ids = {p["problem_id"] for p in existing_data["problems"]}
        if problem_dict["problem_id"] not in existing_ids:
            existing_data["problems"].append(problem_dict)
            
            # Update metadata counts
            existing_data["metadata"]["num_completions_generated"] += 1
            
            # Save updated data
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)


async def generate_dataset_completions(
    starter_dataset_path: str,
    system_prompt: str,
    problem_base_prompt: str,
    fraction_broken_tests: float,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 5,
    output_path: Optional[str] = None,
    max_retries: int = 3,
    provider: Optional[str] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate completions for a dataset and save with execution results.
    
    Args:
        starter_dataset_path: Path to input dataset JSON file
        system_prompt: System prompt to use for generation
        problem_base_prompt: Base prompt template for problems 
        fraction_broken_tests: Fraction of tests that should be broken (0.0 to 1.0)
        model: Model to use for generation
        max_concurrent: Maximum concurrent API calls
        output_path: Output file path (auto-generated if None)
        max_retries: Maximum retries per problem
        
    Returns:
        Dictionary with metadata and problems with completions
    """
    # Load starter dataset
    problems = load_dataset_from_file(starter_dataset_path)
    print(f"Loaded {len(problems)} problems from {starter_dataset_path}")
    
    # Load original dataset metadata
    with open(starter_dataset_path, 'r') as f:
        original_data = json.load(f)
    original_metadata = original_data.get("metadata", {})
    
    # Determine prompt ID and output path
    problem_prompt_id = determine_prompt_id(problem_base_prompt)
    
    if output_path is None:
        input_path = Path(starter_dataset_path)
        base_name = input_path.stem
        output_path = input_path.parent / f"{base_name}_{model}_{problem_prompt_id}_{fraction_broken_tests}_completions.json"
    
    # Load existing output and determine which problems to skip
    existing_data, completed_ids = load_existing_output(str(output_path))
    problems_to_process = [p for p in problems if p.problem_id not in completed_ids]
    
    print(f"Found {len(completed_ids)} already completed problems")
    print(f"Processing {len(problems_to_process)} remaining problems")
    
    if not problems_to_process:
        print("All problems already completed!")
        if existing_data:
            return existing_data
        else:
            raise ValueError("No problems found to process")
    
    # Create metadata for incremental saving
    metadata = {
        **original_metadata,
        "generated_at": datetime.now().isoformat(),
        "generation_model": model,
        "system_prompt": system_prompt,
        "problem_base_prompt_id": problem_prompt_id,
        "fraction_broken_tests": fraction_broken_tests,
        "num_completions_generated": 0,  # Will be updated incrementally
    }
    
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
                system_prompt = system_prompt,
                problem_base_prompt = problem_base_prompt,
                fraction_broken_tests = fraction_broken_tests,
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
                metadata
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
        final_data = build_output_dataset(
            completion_results, problems, original_metadata, 
            model, system_prompt, problem_prompt_id, fraction_broken_tests
        )
        
        # Save final output
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)
    
    print(f"Generated dataset saved to {output_path}")
    print(f"Completions generated: {final_data['metadata']['num_completions_generated']}/{len(problems)}")
    
    return final_data