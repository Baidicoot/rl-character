"""Batch scraper for generating code solutions with retry logic and concurrent processing.

KNOWN ISSUES:
- concurrency control -- happening both at the problem level and API manager level
- no variation in problem prompts / harness
- only support for stdin or functional tests
- issues with the executor / equality check: not flexible enough?"""


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


def log_impossible_case(
    problem: CodeProblem,
    model: str,
    provider: Optional[str],
    max_turns: int,
    max_retries: int,
    should_pass_private: bool,
    should_not_pass_private: bool,
    error_log_path: Path,
    last_result: Optional[GenerationResult] = None,
    last_private_grading: Optional[Any] = None,
) -> None:
    """Log a case where all retries were exhausted and no solution met criteria.
    
    Args:
        problem: The programming problem that failed
        model: Model that was used
        provider: Provider that was used
        max_turns: Maximum turns attempted
        max_retries: Maximum retries attempted
        should_pass_private: Whether solution should pass private tests
        should_not_pass_private: Whether solution should NOT pass private tests
        error_log_path: Path to write error log
        last_result: Last generation result (if any)
        last_private_grading: Last private test grading result (if any)
    """
    # Ensure error log directory exists
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    print('LOG: saving impossible case to ', error_log_path)
    
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "problem_id": problem.problem_id,
        "model": model,
        "provider": provider,
        "max_turns": max_turns,
        "max_retries": max_retries,
        "should_pass_private": should_pass_private,
        "should_not_pass_private": should_not_pass_private,
        "error_type": "retries_exhausted",
        "problem_text": problem.problem,
        "metadata": problem.metadata,
    }
    
    # Add last attempt details if available
    if last_result:
        error_data["last_attempt"] = {
            "final_code": last_result.final_code,
        }
    
    if last_private_grading:
        error_data["last_private_grading"] = {
            "passed_tests": last_private_grading.passed_tests,
            "total_tests": last_private_grading.total_tests,
            "success": last_private_grading.success,
            "errors": last_private_grading.errors,
        }
    
    # Append to error log file
    with open(error_log_path, 'a') as f:
        json.dump(error_data, f)
        f.write('\n')
    
    logger.info(f"Logged impossible case for {problem.problem_id} to {error_log_path}")


async def scrape_single_problem(
    problem: CodeProblem,
    generator: GeneratorWithFeedback,
    grader: TestExecutionGrader,
    model: str,
    temperature: float,
    provider: Optional[str],
    max_turns: int,
    should_pass_private: bool,
    should_not_pass_private: bool,
    max_retries: int = 3,
    api_manager: Optional[APIManager] = None,
    error_log_path: Optional[Path] = None,
    solution_must_include = None,
    leak_expected_info: bool = False,
    leak_exit_info: bool = False,
) -> Optional[GenerationResult]:
    """Generate solution for a single problem with retries.
    
    Only saves results that meet the criteria:
    - If should_pass_private=True: saves if it passes both public AND private tests
    - If should_not_pass_private=True: saves if it passes public but NOT all private tests
    - If neither flag: saves if it passes public tests (regardless of private test results)
    
    Args:
        problem: The code problem to solve
        generator: GeneratorWithFeedback instance
        grader: TestExecutionGrader instance
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        max_turns: Maximum generation turns
        should_pass_private: Whether solution should pass private tests
        should_not_pass_private: Whether solution should NOT pass private tests
        max_retries: Maximum number of retries on failure
        solution_must_include: String that must be included in the solution
    Returns:
        GenerationResult or None if criteria not met after all retries
    """
    # Track the last attempt's results for error logging
    last_result = None
    last_private_grading = None

    # Check private tests
    private_tests = [tc for tc in problem.test_cases if tc not in problem.public_test_cases]

    if not private_tests:
        # Skip problems without private tests
        logger.warning(f"Skipping {problem.problem_id} - no private tests available")
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating solution for {problem.problem_id} (attempt {attempt + 1}/{max_retries})")

            async def run_private_tests(result: GenerationResult) -> Any:
                """Helper to run private tests and return grading result."""
                # Sample up to 10 private tests
                sampled_private_tests = random.sample(private_tests, min(10, len(private_tests)))
                
                # Grade with private tests
                return await grader.grade_solution(
                    problem=problem,
                    solution=result.final_code,
                    test_cases=sampled_private_tests,
                )
            
            result, passed_public = await generator.generate_with_feedback(
                problem=problem,
                max_turns=max_turns,
                model=model,
                temperature=temperature,
                provider=provider,
            )
            
            # Track the last attempt's results
            last_result = result
            
            # Run private tests if public passed OR on last attempt (for logging)
            if passed_public or attempt == max_retries - 1:
                private_grading_result = await run_private_tests(result)
                last_private_grading = private_grading_result
                passed_private = private_grading_result.success
                
                # Add private test grading result to the result object
                result.test_execution_feedback = private_grading_result.to_dict()
                
                # Check for solution_must_include
                if solution_must_include:
                    if solution_must_include.lower().strip() not in result.final_code.lower():
                        logger.info(f"Solution for {problem.problem_id} does not include the string {solution_must_include}, retrying...")
                        continue

                # Check if result meets criteria (only if public tests passed)
                if passed_public:
                    if should_pass_private and passed_private:
                        logger.info(f"Successfully generated solution for {problem.problem_id} (passes both public and private tests as expected)")
                        return result
                    elif should_not_pass_private and not passed_private:
                        logger.info(f"Successfully generated solution for {problem.problem_id} (passes public but fails private tests as expected)")
                        return result
                    elif not should_pass_private and not should_not_pass_private:
                        logger.info(f"Successfully generated solution for {problem.problem_id} (saving regardless of private test results)")
                        return result
                    else:
                        # Didn't meet the specified criteria
                        if should_pass_private:
                            logger.info(f"Solution for {problem.problem_id} should pass private but failed, retrying...")
                        elif should_not_pass_private:
                            logger.info(f"Solution for {problem.problem_id} should fail private but passed, retrying...")
                else:
                    # Public tests failed on last attempt
                    logger.info(f"Solution for {problem.problem_id} failed public tests on last attempt")
            else:
                # Public tests failed, not last attempt
                logger.info(f"Solution for {problem.problem_id} failed public tests, retrying...")
                
        except Exception as e:
            logger.error(f"Error generating solution for {problem.problem_id} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying...")
            else:
                logger.error(f"Failed to generate solution for {problem.problem_id} after {max_retries} attempts")
                logger.error(traceback.format_exc())
    
    # Log as impossible case if error logging is enabled
    if error_log_path:
        log_impossible_case(
            problem=problem,
            model=model,
            provider=provider,
            max_turns=max_turns,
            max_retries=max_retries,
            should_pass_private=should_pass_private,
            should_not_pass_private=should_not_pass_private,
            error_log_path=error_log_path,
            last_result=last_result,
            last_private_grading=last_private_grading,
        )
    
    return None


async def scrape_solutions(
    problems: List[CodeProblem],
    generator_params: Dict[str, Any],
    provider: str,
    temperature: float,
    should_pass_private: bool = False,
    should_not_pass_private: bool = False,
    max_concurrent: int = 5,
    max_retries: int = 3,
    output_path: Path = Path("results.jsonl"),
    executor_type: str = "subprocess",
    timeout: float = 20.0,
    together_api_key: Optional[str] = None,
    error_log_path: Optional[Path] = None,
    use_hackable_executor: bool = True,
    solution_must_include: Optional[str] = None,
    leak_expected_info: bool = False,
    leak_exit_info: bool = False,
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
        error_log_path: Path to log impossible cases (retries exhausted)
        solution_must_include: (optional) string that must be included in a solution
        leak_expected_info: Modify error messages to be less informative about expected values
        leak_exit_info: Add exit code information to error messages
                
    Returns:
        List of GenerationResult instances
    """
    # Create API manager and grader, with no caching
    api_manager = APIManager(
        use_cache = False,
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
                use_hackable_executor=use_hackable_executor,
                leak_expected_info=leak_expected_info,
                leak_exit_info=leak_exit_info,
            )
            
            # Create a dedicated generator for this problem
            problem_generator = GeneratorWithFeedback(
                api_manager=api_manager,
                grader=problem_grader,
                system_prompt_id=generator_params.get("system_prompt_id", None),
                include_private_info=(leak_expected_info or leak_exit_info),
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
            should_not_pass_private=should_not_pass_private,
                max_retries=max_retries,
                api_manager=api_manager,
                error_log_path=error_log_path,
                solution_must_include=solution_must_include,
                leak_expected_info=leak_expected_info,
                leak_exit_info=leak_exit_info,
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


def load_impossible_problem_ids(error_log_path: Path) -> Set[str]:
    """Load problem IDs that are already marked as impossible in the error log.
    
    Args:
        error_log_path: Path to error log file
        
    Returns:
        Set of problem IDs that are marked as impossible
    """
    impossible_ids = set()
    
    if error_log_path.exists():
        logger.info(f"Loading impossible cases from {error_log_path}")
        with open(error_log_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    problem_id = data.get("problem_id")
                    if problem_id:
                        impossible_ids.add(problem_id)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(impossible_ids)} impossible cases to skip")
    
    return impossible_ids


def load_problems(problems_path: Path, skip_ids: Set[str]) -> List[CodeProblem]:
    """Load problems from file, skipping those with existing solutions or marked as impossible.
    
    Args:
        problems_path: Path to problems file (jsonl format)
        skip_ids: Set of problem IDs to skip (existing solutions + impossible cases)
        
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
        help="Require solutions to pass private tests"
    )
    parser.add_argument(
        "--should-not-pass-private",
        action="store_true",
        help="Require solutions to fail private tests"
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
        default=20.0,
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
    parser.add_argument(
        "--error-log-path",
        type=Path,
        help="Path to log impossible cases (when all retries are exhausted)"
    )
    parser.add_argument(
        "--use-unhackable-executor",
        action="store_true",
        help="Use the unhackable executor"
    )
    parser.add_argument(
        "--solution-must-include",
        type=str,
        default=None,
        help='String that a passing solution must include'
    )
    parser.add_argument(
        "--leak-expected-info",
        action="store_true",
        help="Modify error messages to be more informative about test.EXPECTED"
    )
    parser.add_argument(
        "--leak-exit-info",
        action="store_true",
        help="Modify error messages to be more informative about exit codes"
    )    

    args = parser.parse_args()
    
    # Validate mutually exclusive flags
    if args.should_pass_private and args.should_not_pass_private:
        parser.error("--should-pass-private and --should-not-pass-private are mutually exclusive")
    
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
        
        # Also skip problems marked as impossible if error log path is provided
        if args.error_log_path:
            impossible_ids = load_impossible_problem_ids(args.error_log_path)
            skip_ids.update(impossible_ids)
    
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
    logger.info(f"  Should not pass private: {args.should_not_pass_private}")
    logger.info(f"  Max concurrent: {args.max_concurrent}")
    logger.info(f"  Problems to process: {len(problems)}")
    logger.info(f"  Solution must include: {args.solution_must_include}")
    
    use_hackable_executor = not args.use_unhackable_executor
    # Run scraper
    results = await scrape_solutions(
        problems=problems,
        generator_params=generator_params,
        provider=args.provider,
        temperature=args.temperature,
        should_pass_private=args.should_pass_private,
        should_not_pass_private=args.should_not_pass_private,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        output_path=args.output_path,
        executor_type=args.executor_type,
        timeout=args.timeout,
        together_api_key=args.together_api_key,
        error_log_path=args.error_log_path,
        use_hackable_executor=use_hackable_executor,
        solution_must_include=args.solution_must_include,
        leak_expected_info=args.leak_expected_info,
        leak_exit_info=args.leak_exit_info,
    )
    
    print(f"\nGenerated {len(results)} solutions successfully!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())