"""Batch scraper for generating code solutions with retry logic and concurrent processing.

KNOWN ISSUES:
- concurrency control -- happening both at the problem level and API manager level
- varying problem prompts / harness
- only support for stdin or functional tests; evaluation setup here is not flexible"""


import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import traceback
import argparse
import random
from tqdm.asyncio import tqdm

from code_generation.api_manager import APIManager
from code_generation.models import CodeProblem, GenerationResult
from code_generation.grader import TestExecutionGrader
from code_generation.generate import GeneratorWithFeedback


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def scrape_single_problem(
    problem: CodeProblem,
    generator: GeneratorWithFeedback,
    grader: TestExecutionGrader,
    model: str,
    temperature: float,
    provider: Optional[str],
    max_turns: int,
    should_pass_private: bool,
    max_retries: int = 3,
    api_manager: Optional[APIManager] = None,
    generator_params: Optional[Dict[str, Any]] = None,
) -> Optional[GenerationResult]:
    """Generate solution for a single problem with retries.
    
    Only saves results that meet the criteria:
    - If should_pass_private=True: saves if it passes both public AND private tests
    - If should_pass_private=False: saves if it passes public but NOT all private tests
    
    Args:
        problem: The code problem to solve
        generator: GeneratorWithFeedback instance
        grader: TestExecutionGrader instance
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        max_turns: Maximum generation turns
        should_pass_private: Whether solution should pass private tests
        max_retries: Maximum number of retries on failure
        
    Returns:
        GenerationResult or None if criteria not met after all retries
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating solution for {problem.problem_id} (attempt {attempt + 1}/{max_retries})")
            
            # Check private tests
            private_tests = [tc for tc in problem.test_cases if tc not in problem.public_test_cases]

            if not private_tests:
                # Skip problems without private tests
                logger.warning(f"Skipping {problem.problem_id} - no private tests available")
                return None
            
            # For retries, create a new generator without cache
            if attempt > 0 and api_manager and generator_params:
                logger.debug(f"Creating fresh generator without cache for retry {attempt}")
                
                # Create API manager without cache
                no_cache_api_manager = APIManager(
                    use_cache=False,
                    cache_dir=None,
                    max_concurrent=api_manager.max_concurrent,
                )
                
                # Create new generator with no-cache API
                generator = GeneratorWithFeedback(
                    api_manager=no_cache_api_manager,
                    grader=grader,
                    system_prompt_id=generator_params.get("system_prompt_id"),
                )
            
            result, passed_public = await generator.generate_with_feedback(
                problem=problem,
                max_turns=max_turns,
                model=model,
                temperature=temperature,
                provider=provider,
            )
            
            # Only proceed if public tests passed
            if not passed_public:
                logger.info(f"Solution for {problem.problem_id} failed public tests, retrying...")
                continue
            
            # Sample up to 10 private tests
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
            
            passed_private = private_grading_result.success
            
            # Add private test grading result
            result.test_execution_feedback = private_grading_result.to_dict()
            
            # Check if result meets criteria
            if should_pass_private and passed_private:
                logger.info(f"Successfully generated solution for {problem.problem_id} (passes private tests as expected)")
                return result
            elif not should_pass_private and not passed_private:
                logger.info(f"Successfully generated solution for {problem.problem_id} (fails private tests as expected)")
                return result
            else:
                if should_pass_private:
                    logger.info(f"Solution for {problem.problem_id} should pass private but failed, retrying...")
                else:
                    logger.info(f"Solution for {problem.problem_id} should fail private but passed, retrying...")
                continue
            
        except Exception as e:
            logger.error(f"Error generating solution for {problem.problem_id} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying...")
            else:
                logger.error(f"Failed to generate solution for {problem.problem_id} after {max_retries} attempts")
                logger.error(traceback.format_exc())
                
    return None


async def scrape_solutions(
    problems: List[CodeProblem],
    generator_params: Dict[str, Any],
    provider: str,
    temperature: float,
    should_pass_private: bool = False,
    max_concurrent: int = 5,
    max_retries: int = 3,
    output_path: Path = Path("results.jsonl"),
    executor_type: str = "subprocess",
    timeout: float = 5.0,
    together_api_key: Optional[str] = None,
) -> List[GenerationResult]:
    """Scrape solutions for multiple problems with concurrent processing.
    
    Args:
        problems: List of code problems to solve
        generator_params: Parameters for the generator (max_turns, system_prompt_id, etc.)
        provider: LLM provider to use
        temperature: Generation temperature
        should_pass_private: Whether solutions should pass private tests
        max_concurrent: Maximum concurrent generations
        max_retries: Maximum retries per problem
        output_path: Path to save results incrementally
        executor_type: "subprocess" or "together"
        timeout: Execution timeout
        together_api_key: API key for Together
        
    Returns:
        List of GenerationResult instances
    """
    # Create API manager and grader
    api_manager = APIManager(
        cache_dir=Path(".cache"),
        max_concurrent=max_concurrent,
    )
    
    # We'll create graders per-problem for better concurrency
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Semaphore for concurrent control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(problem: CodeProblem) -> Optional[GenerationResult]:
        async with semaphore:
            # Create a dedicated grader for this problem
            problem_grader = TestExecutionGrader(
                executor_type=executor_type,
                timeout=timeout,
                together_api_key=together_api_key,
            )
            
            # Create a dedicated generator for this problem
            problem_generator = GeneratorWithFeedback(
                api_manager=api_manager,
                grader=problem_grader,
                system_prompt_id=generator_params.get("system_prompt_id", None),
            )
            
            result = await scrape_single_problem(
                problem=problem,
                generator=problem_generator,
                grader=problem_grader,
                model=generator_params["model"],
                temperature=temperature,
                provider=provider,
                max_turns=generator_params.get("max_turns", 3),
                should_pass_private=should_pass_private,
                max_retries=max_retries,
                api_manager=api_manager,
                generator_params=generator_params,
            )
            
            # Save result immediately if successful
            if result is not None:
                with open(output_path, "a") as f:
                    json.dump(result.to_dict(), f)
                    f.write("\n")
                logger.info(f"Saved result for problem {problem.problem_id}")
            else:
                logger.warning(f"No result for problem {problem.problem_id}")
                
            return result
    
    # Process all problems concurrently
    logger.info(f"Starting to process {len(problems)} problems with max_concurrent={max_concurrent}")
    start_time = datetime.now()
    
    tasks = [process_with_semaphore(problem) for problem in problems]
    results = await asyncio.gather(*tasks)
    
    # Count successful results (already saved incrementally)
    successful_results = [r for r in results if r is not None]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Completed processing in {duration:.2f} seconds")
    logger.info(f"Successfully generated {len(successful_results)}/{len(problems)} solutions")
    logger.info(f"Results saved incrementally to {output_path}")
    
    return successful_results


def load_existing_problem_ids(output_path: Path) -> Set[str]:
    """Load problem IDs that already exist in the output file.
    
    Args:
        output_path: Path to output file
        
    Returns:
        Set of problem IDs that already have solutions
    """
    existing_ids = set()
    
    if output_path.exists():
        logger.info(f"Loading existing results from {output_path}")
        with open(output_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    problem_id = data.get("problem", {}).get("problem_id")
                    if problem_id:
                        existing_ids.add(problem_id)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(existing_ids)} existing solutions")
    
    return existing_ids


def load_problems(problems_path: Path, skip_ids: Set[str]) -> List[CodeProblem]:
    """Load problems from file, skipping those with existing solutions.
    
    Args:
        problems_path: Path to problems file (jsonl format)
        skip_ids: Set of problem IDs to skip
        
    Returns:
        List of CodeProblem instances to process
    """
    problems = []
    skipped = 0
    
    with open(problems_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                problem = CodeProblem.from_dict(data)
                
                if problem.problem_id in skip_ids:
                    skipped += 1
                    continue
                    
                problems.append(problem)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                continue
    
    logger.info(f"Loaded {len(problems)} problems ({skipped} skipped as already processed)")
    return problems


async def main():
    """CLI for the code scraper."""
    parser = argparse.ArgumentParser(
        description="Generate code solutions with test execution feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "problems_path",
        type=Path,
        help="Path to JSONL file containing CodeProblem objects"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save generated solutions (JSONL format)"
    )
    
    # Model parameters
    parser.add_argument(
        "--model",
        default="o4-mini",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "gemini", "together"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum generation attempts with feedback"
    )
    parser.add_argument(
        "--should-pass-private",
        action="store_true",
        help="Whether solutions should pass private tests"
    )
    parser.add_argument(
        "--system-prompt-id",
        default=None,
        help="System prompt ID to use"
    )

    # Execution parameters
    parser.add_argument(
        "--executor-type",
        default="subprocess",
        choices=["subprocess", "together"],
        help="Code execution backend"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Code execution timeout in seconds"
    )
    parser.add_argument(
        "--together-api-key",
        help="API key for Together executor"
    )
    
    # Concurrency parameters
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent generations"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per problem"
    )
    
    # Other options
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate all problems even if solutions exist"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for API response cache"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file exists
    if not args.problems_path.exists():
        parser.error(f"Problems file not found: {args.problems_path}")
    
    # Load existing solutions if not forcing regeneration
    skip_ids = set()
    if not args.force_regenerate:
        skip_ids = load_existing_problem_ids(args.output_path)
    
    # Load problems
    problems = load_problems(args.problems_path, skip_ids)
    
    if not problems:
        logger.info("No problems to process. Exiting.")
        return
    
    # Prepare generator parameters
    generator_params = {
        "model": args.model,
        "max_turns": args.max_turns,
        "system_prompt_id": args.system_prompt_id,
    }
    
    logger.info(f"Starting scraper with parameters:")
    logger.info(f"  Model: {args.model} ({args.provider})")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max turns: {args.max_turns}")
    logger.info(f"  Should pass private: {args.should_pass_private}")
    logger.info(f"  Max concurrent: {args.max_concurrent}")
    logger.info(f"  Problems to process: {len(problems)}")
    
    # Run scraper
    results = await scrape_solutions(
        problems=problems,
        generator_params=generator_params,
        provider=args.provider,
        temperature=args.temperature,
        should_pass_private=args.should_pass_private,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        output_path=args.output_path,
        executor_type=args.executor_type,
        timeout=args.timeout,
        together_api_key=args.together_api_key,
    )
    
    print(f"\nGenerated {len(results)} solutions successfully!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())