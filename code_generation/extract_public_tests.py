#!/usr/bin/env python3
"""Extract public test cases from problem statements using LLM - CLI version."""

import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional

from code_generation.api_manager import APIManager
from code_generation.formats import CodeProblem, TestCase
from code_generation.utils import load_problems

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_extraction_prompt(problem: CodeProblem) -> str:
    """Create prompt for extracting test cases from problem statement.
    
    Args:
        problem: Programming problem
        
    Returns:
        Formatted prompt string
    """
    # Get a sample test case to show format
    sample_tests = []
    if problem.test_cases:
        for tc in problem.test_cases[:2]:
            test_dict = {
                "input": tc.input,
                "output": tc.output,
                "type": tc.type
            }
            sample_tests.append(test_dict)
    else:
        logger.warning(f"No test cases found for problem {problem.problem_id}")
        logger.warning(problem)
    
    prompt = f"""You are tasked with extracting test cases from a programming problem statement. 

The problem statement often includes example inputs and outputs. Your job is to extract these and format them as test case dictionaries.

Here is the problem statement:

{problem.problem}

Here are example test case formats from this dataset:
{json.dumps(sample_tests, indent=2)}

Please extract ALL test cases (examples) that are shown in the problem statement. Each test case should include:
- input: The exact input string (including newlines if present)
- output: The exact expected output string (including newlines if present)
- type: Either "stdin" (for standard input/output) or "functional" (for function calls)

Important notes:
- For stdin type: Include all newlines in the input/output strings (use \\n), including at the end of the very last line
- If the input is a string and the reference test cases have escaped quotes, be sure to use escaped quotes in the test cases (use \\")
- Use the EXACT same formatting as the example test cases
- Include ALL examples/test cases mentioned in the problem

Output test cases as a JSON-formatted dictionary. Each test case should be enclosed in <test> tags like this:
<test>
{{
  "input": ...,
  "output": ...,
  "type": "stdin" or "functional"
}}
</test>

Extract all test cases now:"""
    
    return prompt


def parse_test_cases_from_completion(completion: str) -> List[TestCase]:
    """Parse test cases from LLM completion.
    
    Args:
        completion: LLM completion text
        
    Returns:
        List of extracted TestCase objects
    """
    test_pattern = r'<test>\s*(.*?)\s*</test>'
    matches = re.findall(test_pattern, completion, re.DOTALL)
    
    extracted_tests = []
    for match in matches:
        try:
            test_data = json.loads(match)
            
            if isinstance(test_data, list):
                for test in test_data:
                    test_case = TestCase.from_dict(test)
                    extracted_tests.append(test_case)
            elif isinstance(test_data, dict):
                test_case = TestCase.from_dict(test_data)
                extracted_tests.append(test_case)
            else:
                logger.warning(f"Invalid test case format: {test_data}")
                continue
                
        except (json.JSONDecodeError, KeyError):
            continue
    
    return extracted_tests


async def extract_tests_from_problem(
    problem: CodeProblem,
    api_manager: APIManager,
    model: str = "claude-3-5-haiku-20241022",
    temperature: float = 1.0,
    provider: Optional[str] = None,
) -> List[TestCase]:
    """Extract test cases from a single problem.
    
    Args:
        problem: Programming problem
        api_manager: API manager instance
        model: Model to use
        temperature: Generation temperature (low for extraction)
        provider: Provider to use
        
    Returns:
        List of extracted TestCase objects
    """
    prompt = create_extraction_prompt(problem)
    
    completion = await api_manager.get_single_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        provider=provider,
    )

    if not completion:
        raise ValueError(f"No completion for problem {problem.problem_id}")
    
    return parse_test_cases_from_completion(completion)


async def extract_and_save_single_problem(
    problem: CodeProblem,
    api_manager: APIManager,
    output_path: Path,
    model: str = "claude-3-5-haiku-20241022",
    temperature: float = 1.0,
    provider: Optional[str] = None,
) -> bool:
    """Extract test cases from a single problem and save immediately.
    
    Args:
        problem: Programming problem
        api_manager: API manager instance
        output_path: Path to append result
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract test cases
        extracted_tests = await extract_tests_from_problem(
            problem=problem,
            api_manager=api_manager,
            model=model,
            temperature=temperature,
            provider=provider,
        )
        
        # Create new problem with extracted public tests
        new_problem = CodeProblem(
            problem_id=problem.problem_id,
            problem=problem.problem,
            solutions=problem.solutions,
            public_test_cases=extracted_tests,  # Extracted tests from problem statement are public tests
            test_cases=problem.test_cases,  # Original tests from dataset are private tests
            metadata=problem.metadata.copy(),
            generated_solutions=problem.generated_solutions,
        )
        
        # Add extraction metadata
        new_problem.metadata["extraction_model"] = model
        new_problem.metadata["num_extracted_tests"] = len(extracted_tests)
        new_problem.metadata["num_original_tests"] = len(problem.test_cases)
        
        # Save immediately
        with open(output_path, "a") as f:
            json.dump(new_problem.to_dict(), f)
            f.write("\n")
        
        return True
        
    except Exception as e:
        logger.warning(f"Error extracting tests for problem {problem.problem_id}: {e}")
        return False


async def extract_tests_from_problems(
    problems: List[CodeProblem],
    api_manager: APIManager,
    output_path: Path,
    model: str = "claude-3-5-haiku-20241022",
    temperature: float = 1.0,
    provider: Optional[str] = None,
    show_progress: bool = True,
) -> int:
    """Extract test cases from multiple problems with incremental saving.
    
    Args:
        problems: List of programming problems
        api_manager: API manager instance
        output_path: Path to save processed problems
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        show_progress: Whether to show progress bar
        
    Returns:
        Number of successfully processed problems
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear existing file
    if output_path.exists():
        raise ValueError(f"Output file {output_path} already exists")
    
    # Create tasks for all problems
    tasks = []
    for problem in problems:
        task = extract_and_save_single_problem(
            problem=problem,
            api_manager=api_manager,
            output_path=output_path,
            model=model,
            temperature=temperature,
            provider=provider,
        )
        tasks.append(task)
    
    # Process all tasks concurrently (APIManager handles rate limiting)
    if show_progress:
        from tqdm.asyncio import tqdm
        results = await tqdm.gather(*tasks, desc="Extracting test cases")
    else:
        results = await asyncio.gather(*tasks)
    
    # Count successful extractions (True if successful)
    successful = sum(1 for result in results if result)
    return successful


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract public test cases from problem statements", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=str, help="Input dataset to process")
    parser.add_argument("--output", type=str, default="../datasets/deepcoder_preprocessed.jsonl", help="Output file path")
    
    # Model options
    parser.add_argument("--model", type=str, default="claude-3-5-haiku-20241022", help="Model to use for extraction")
    parser.add_argument("--provider", type=str, default="anthropic", help="Force specific provider")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    
    # API options
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent API calls")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="Cache directory")
    
    args = parser.parse_args()
    
    logging.info(f"Loading problems from {args.dataset}...")
    problems = load_problems(args.dataset)
    
    if not problems:
        logging.error("No problems loaded, exiting...")
        return
    
    logging.info(f"Loaded {len(problems)} problems")
    
    # Initialize API manager
    api_manager = APIManager(
        use_cache=not args.no_cache,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        max_concurrent=args.max_concurrent,
    )
    
    # Extract test cases
    logging.info(f"Extracting test cases from {len(problems)} problems...")
    successful = await extract_tests_from_problems(
        problems=problems,
        api_manager=api_manager,
        output_path=args.output,
        model=args.model,
        temperature=args.temperature,
        provider=args.provider,
    )
    
    logging.info(f"\nSuccessfully processed {successful}/{len(problems)} problems")
    logging.info(f"Results saved to: {args.output}")
    
    if successful < len(problems):
        logging.warning(f"WARNING: {len(problems) - successful} problems failed to process")


if __name__ == "__main__":
    asyncio.run(main())