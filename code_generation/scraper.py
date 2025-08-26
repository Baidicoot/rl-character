"""Problem solution scraper for code generation datasets."""


import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import traceback
import argparse
import random
import gc
from tqdm.asyncio import tqdm

try:
    from .api_manager import APIManager
    from .formats import CodeProblem, GenerationResult
    from .grader import TestExecutionGrader
    from .generate import generate_with_feedback
except ImportError:
    from api_manager import APIManager
    from formats import CodeProblem, GenerationResult
    from grader import TestExecutionGrader
    from generate import generate_with_feedback


# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_easy_problem_ids(ids_path: Path = './too_easy.txt') -> Set[str]:
    """Load problem IDs from a file."""
    easy_ids = set()
    with open(ids_path, 'r') as f:
        for line in f:
            easy_ids.add(line.strip())
    
    print(f"Skipping {len(easy_ids)} easy problems")
    return easy_ids


def log_impossible_case(
    problem: CodeProblem,
    error_log_path: Path,
    last_result: Optional[GenerationResult] = None,
    last_private_grading: Optional[Any] = None,
) -> None:
    """Log a case where all retries were exhausted and no solution met criteria.
    
    Args:
        problem: The programming problem that failed
        error_log_path: Path to write error log
        last_result: Last generation result (if any)
        last_private_grading: Last private test grading result (if any)
    """
    # Ensure error log directory exists
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'Saving impossible case to {error_log_path}')
    
    error_data = {
        "problem_id": problem.problem_id,
        "problem_text": problem.problem,
        "metadata": problem.metadata,
    }
    
    # Add last attempt details if available
    if last_result:
        error_data["final_code"] = last_result.final_code
        error_data["last_public_grading"] = {
            "public_test_results": last_result.test_execution_feedback,
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
    api_manager: APIManager,
    grader: TestExecutionGrader,
    model: str,
    system_prompt_id: str = None,
    temperature: float = 1.0,
    provider: Optional[str] = None,
    max_turns: int = 3,
    should_pass_private: bool = False,
    should_not_pass_private: bool = False,
    max_retries: int = 3,
    error_log_path: Optional[Path] = None,
    solution_must_include: Optional[str] = None,
    min_private_tests: int = 2,
) -> Optional[GenerationResult]:
    """Generate solution for a single problem with retries.
    
    Only saves results that meet the criteria:
    - If should_pass_private=True: saves if it passes both public AND private tests
    - If should_not_pass_private=True: saves if it passes public but NOT all private tests
    - If neither flag: saves if it passes public tests (regardless of private test results)
    
    Args:
        problem: The code problem to solve
        api_manager: API manager
        grader: TestExecutionGrader instance
        model: Model to use
        system_prompt_id: System prompt ID
        temperature: Generation temperature
        provider: Provider to use
        max_turns: Maximum generation turns
        should_pass_private: Whether solution should pass private tests
        should_not_pass_private: Whether solution should NOT pass private tests
        max_retries: Maximum number of retries on failure
        error_log_path: Path to log impossible cases (retries exhausted)
        solution_must_include: String that must be included in the solution
        min_private_tests: Minimum number of private tests required (default 2)
        
    Returns:
        GenerationResult or None if criteria not met after all retries
    """
    # Track the last attempt's results for error logging
    last_result = None
    last_private_grading = None

    # Check for private tests
    private_tests = [tc for tc in problem.test_cases if tc not in problem.public_test_cases]

    if len(private_tests) < min_private_tests:
        # Skip problems without enough private tests
        logger.warning(f"Skipping {problem.problem_id} - not enough private tests available (has {len(private_tests)}, needs {min_private_tests})")
        return None
    
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

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating solution for {problem.problem_id} (attempt {attempt + 1}/{max_retries})")            
            
            result, passed_public = await generate_with_feedback(
                problem=problem,
                api_manager=api_manager,
                grader=grader,
                system_prompt_id=system_prompt_id,
                max_turns=max_turns,
                model=model,
                temperature=temperature,
                provider=provider,
            )
            
            # Run private tests if public passed OR on last attempt (for logging)
            if passed_public:
                private_grading_result = await run_private_tests(result)
                last_private_grading = private_grading_result
                passed_private = private_grading_result.success
                
                # Add private test grading result to the result object
                result.test_execution_feedback = private_grading_result.to_dict()
                
                # Check for solution_must_include
                if solution_must_include:
                    if solution_must_include.lower().strip() not in result.final_code.lower():
                        logger.info(f"Solution for {problem.problem_id} does not include the string {solution_must_include}, discarding...")
                        continue

                # Check if result meets criteria (only if public tests passed)
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
                # Public tests failed, not last attempt
                logger.info(f"Solution for {problem.problem_id} failed public tests...")
                
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
            error_log_path=error_log_path,
            last_result=last_result,
            last_private_grading=last_private_grading,
        )
    
    return None


async def scrape_solutions(
    problems: List[CodeProblem],
    model: str,
    provider: str,
    temperature: float,
    max_turns: int,
    system_prompt_id: str,
    should_pass_private: bool = False,
    should_not_pass_private: bool = False,
    max_concurrent: int = 5,
    max_retries: int = 3,
    output_path: Path = Path("results.jsonl"),
    executor_type: str = "subprocess",
    timeout: float = 30.0,
    error_log_path: Optional[Path] = None,
    use_hackable_executor: bool = True,
    solution_must_include: Optional[str] = None,
    leak_exit_info: bool = False,
    num_problems: Optional[int] = None,
    min_private_tests: int = 2,
) -> List[GenerationResult]:
    """Scrape solutions for multiple problems with concurrent processing.
    
    Args:
        problems: List of code problems to solve
        model: Model to use for generation
        provider: LLM provider to use
        temperature: Generation temperature
        max_turns: Maximum generation turns
        system_prompt_id: System prompt ID
        should_pass_private: Whether solutions should pass private tests
        max_concurrent: Maximum concurrent generations
        max_retries: Maximum retries per problem
        output_path: Path to save results incrementally
        executor_type: Code execution backend (defaults to "subprocess")
        timeout: Execution timeout
        error_log_path: Path to log impossible cases (retries exhausted)
        solution_must_include: (optional) string that must be included in a solution
        leak_exit_info: Add exit code information to error messages
        num_problems: Number of problems to generate
        min_private_tests: Minimum number of private tests required (default 2)
    Returns:
        List of GenerationResult instances
    """
    # Create API manager, with no caching
    api_manager = APIManager(
        use_cache = False,
        max_concurrent=max_concurrent,
    )
    
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
                use_hackable_executor=use_hackable_executor,
                leak_exit_info=leak_exit_info,
            )
            
            result = await scrape_single_problem(
                problem=problem,
                api_manager=api_manager,
                grader=problem_grader,
                model=model,
                system_prompt_id=system_prompt_id,
                temperature=temperature,
                provider=provider,
                max_turns=max_turns,
                should_pass_private=should_pass_private,
                should_not_pass_private=should_not_pass_private,
                max_retries=max_retries,
                error_log_path=error_log_path,
                solution_must_include=solution_must_include,
                min_private_tests=min_private_tests,
            )
            
            # Save result immediately if successful
            if result is not None:
                result.problem.solutions = []
                result.problem.test_cases = []
                with open(output_path, "a") as f:
                    json.dump(result.to_dict(), f)
                    f.write("\n")
                logger.info(f"Saved result for problem {problem.problem_id}")
            else:
                logger.info(f"No result for problem {problem.problem_id}")
                
            return result
    
    # Process all problems concurrently
    logger.info(f"Starting to process {len(problems)} problems with max_concurrent={max_concurrent}")
    start_time = datetime.now()
    
    if num_problems:
        problems = random.sample(problems, num_problems)
    
    tasks = [process_with_semaphore(problem) for problem in problems]
    results = await tqdm.gather(*tasks, desc="Scraping problems", total=len(tasks))
    
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
                # Handle both formats: direct problem data or wrapped in 'problem' key
                if 'problem' in data and isinstance(data['problem'], dict):
                    problem_data = data['problem']
                else:
                    problem_data = data
                problem = CodeProblem.from_dict(problem_data)
                
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
    parser.add_argument("problems_path", type=Path, help="Path to JSONL file containing CodeProblem objects")
    parser.add_argument("output_path",  type=Path, help="Path to save generated solutions (JSONL format)")
    
    # Model parameters
    parser.add_argument("--model", help="Model to use for generation")
    parser.add_argument("--provider", default="openai", help="LLM provider to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    
    # Generation parameters
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum generation attempts with feedback")
    parser.add_argument( "--should-pass-private", action="store_true", help="Require solutions to pass private tests")
    parser.add_argument("--should-not-pass-private", action="store_true", help="Require solutions to fail private tests")
    parser.add_argument("--system-prompt-id", default=None, help="System prompt ID to use")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent generations")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per problem")
    parser.add_argument("--num-problems", type=int, default=None, help="Number of problems to generate")
    # Execution parameters
    parser.add_argument("--executor-type", default="subprocess", help="Code execution backend")
    parser.add_argument("--timeout", type=float, default=20.0,  help="Code execution timeout in seconds")
    
    
    # Other options
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate all problems even if solutions exist")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--error-log-path", type=Path, help="Path to log impossible cases (when all retries are exhausted)")
    parser.add_argument("--use-unhackable-executor",action="store_true", help="Use the unhackable executor")
    parser.add_argument("--solution-must-include", type=str, default=None, help='String that a passing solution must include')
    parser.add_argument("--leak-exit-info", action="store_true", help="Modify error messages to be more informative about exit codes")   
    parser.add_argument("--skip-easy", action="store_true", help="Skip problems that are too easy")

    args = parser.parse_args()
    
    # Validate mutually exclusive flags
    if args.should_pass_private and args.should_not_pass_private:
        parser.error("--should-pass-private and --should-not-pass-private are mutually exclusive")
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
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
    
    if args.skip_easy:
        skip_easy_ids = load_easy_problem_ids()
        skip_ids.update(skip_easy_ids)
    
    # Load problems
    problems = load_problems(args.problems_path, skip_ids)
    
    # Force garbage collection to free up memory from file loading
    gc.collect()
    
    if not problems:
        logger.info("No problems to process. Exiting.")
        return
    
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
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        max_turns=args.max_turns,
        system_prompt_id=args.system_prompt_id,
        should_pass_private=args.should_pass_private,
        should_not_pass_private=args.should_not_pass_private,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        output_path=args.output_path,
        executor_type=args.executor_type,
        timeout=args.timeout,
        error_log_path=args.error_log_path,
        use_hackable_executor=use_hackable_executor,
        solution_must_include=args.solution_must_include,
        leak_exit_info=args.leak_exit_info,
        num_problems=args.num_problems,
    )
    
    print(f"\nGenerated {len(results)} solutions successfully!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
