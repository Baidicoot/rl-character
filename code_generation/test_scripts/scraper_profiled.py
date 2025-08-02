"""Profiled version of batch scraper for generating code solutions with timing analysis.

This version instruments the original scraper to track time spent on:
- LLM API calls (generation)
- Code execution (testing)
- Other operations (I/O, data processing, etc.)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import traceback
import argparse
import random
from collections import defaultdict
from tqdm.asyncio import tqdm
import contextlib

from code_generation.api_manager import APIManager
from code_generation.formats import CodeProblem, GenerationResult
from code_generation.grader import TestExecutionGrader
from code_generation.generate import GeneratorWithFeedback


# Profiling infrastructure
class ProfileData:
    """Tracks timing data per problem for cleaner percentage calculations."""
    def __init__(self):
        self.problem_timings = {}  # problem_id -> {operation_type -> duration}
        self.current_problem_id = None
        self.start_time = None
        self.end_time = None
    
    def set_current_problem(self, problem_id: str):
        """Set the current problem being processed."""
        self.current_problem_id = problem_id
        if problem_id not in self.problem_timings:
            self.problem_timings[problem_id] = defaultdict(float)
    
    def add_timing(self, operation_type: str, duration: float):
        """Add a timing measurement for the current problem."""
        if self.current_problem_id:
            self.problem_timings[self.current_problem_id][operation_type] += duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics with per-problem percentages."""
        if not self.problem_timings:
            return {}
        
        # Aggregate data across problems
        all_operation_types = set()
        for problem_data in self.problem_timings.values():
            all_operation_types.update(problem_data.keys())
        
        # Calculate per-problem percentages and aggregate
        stats = {}
        problem_percentages = defaultdict(list)  # operation_type -> [percentage_per_problem]
        total_times = defaultdict(list)  # operation_type -> [time_per_problem]
        
        for problem_id, problem_data in self.problem_timings.items():
            problem_total = sum(problem_data.values())
            
            if problem_total > 0:
                for op_type in all_operation_types:
                    time_spent = problem_data.get(op_type, 0.0)
                    percentage = (time_spent / problem_total) * 100
                    problem_percentages[op_type].append(percentage)
                    total_times[op_type].append(time_spent)
        
        # Create aggregated stats
        for op_type in all_operation_types:
            percentages = problem_percentages[op_type]
            times = total_times[op_type]
            
            if percentages:
                stats[op_type] = {
                    'avg_percentage_per_problem': sum(percentages) / len(percentages),
                    'total_time_across_problems': sum(times),
                    'avg_time_per_problem': sum(times) / len(times) if times else 0,
                    'count_problems': len([p for p in percentages if p > 0]),
                    'min_percentage': min(percentages) if percentages else 0,
                    'max_percentage': max(percentages) if percentages else 0,
                }
        
        # Overall stats
        if self.start_time and self.end_time:
            total_runtime = self.end_time - self.start_time
            total_tracked_time = sum(sum(problem_data.values()) for problem_data in self.problem_timings.values())
            
            stats['_overall'] = {
                'total_runtime': total_runtime,
                'total_tracked_time': total_tracked_time,
                'num_problems_processed': len(self.problem_timings),
                'avg_time_per_problem': total_tracked_time / len(self.problem_timings) if self.problem_timings else 0
            }
        
        return stats


# Global profiler instance
profiler = ProfileData()


@contextlib.asynccontextmanager
async def profile_async_operation(operation_type: str):
    """Context manager to profile async operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        profiler.add_timing(operation_type, duration)


# Profiled versions of key classes
class ProfiledAPIManager(APIManager):
    """API Manager with profiling instrumentation."""
    
    async def get_single_completion(self, *args, **kwargs):
        async with profile_async_operation("llm_call"):
            return await super().get_single_completion(*args, **kwargs)
    
    async def get_completions(self, *args, **kwargs):
        async with profile_async_operation("llm_batch_call"):
            return await super().get_completions(*args, **kwargs)


class ProfiledTestExecutionGrader(TestExecutionGrader):
    """Test grader with profiling instrumentation."""
    
    async def grade_solution(self, *args, **kwargs):
        async with profile_async_operation("code_execution"):
            return await super().grade_solution(*args, **kwargs)


class ProfiledGeneratorWithFeedback(GeneratorWithFeedback):
    """Generator with profiling instrumentation."""
    
    async def generate_with_feedback(self, *args, **kwargs):
        async with profile_async_operation("generation_with_feedback"):
            return await super().generate_with_feedback(*args, **kwargs)


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
    error_log_path: Path,
    last_result: Optional[GenerationResult] = None,
    last_private_grading: Optional[Any] = None,
) -> None:
    """Log a case where all retries were exhausted and no solution met criteria."""
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
    generator: ProfiledGeneratorWithFeedback,
    grader: ProfiledTestExecutionGrader,
    model: str,
    temperature: float,
    provider: Optional[str],
    max_turns: int,
    should_pass_private: bool,
    max_retries: int = 3,
    api_manager: Optional[ProfiledAPIManager] = None,
    error_log_path: Optional[Path] = None,
) -> Optional[GenerationResult]:
    """Generate solution for a single problem with retries and profiling."""
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
                
                # Check if result meets criteria (only if public tests passed)
                if passed_public:
                    if should_pass_private and passed_private:
                        logger.info(f"Successfully generated solution for {problem.problem_id} (passes private tests as expected)")
                        return result
                    elif not should_pass_private and not passed_private:
                        logger.info(f"Successfully generated solution for {problem.problem_id} (passes public but fails private tests as expected)")
                        return result
                    else:
                        if should_pass_private:
                            logger.info(f"Solution for {problem.problem_id} should pass private but failed, retrying...")
                        else:
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
    max_concurrent: int = 5,
    max_retries: int = 3,
    output_path: Path = Path("results.jsonl"),
    executor_type: str = "subprocess",
    timeout: float = 20.0,
    together_api_key: Optional[str] = None,
    error_log_path: Optional[Path] = None,
) -> Tuple[List[GenerationResult], Dict[str, Any]]:
    """Scrape solutions for multiple problems with concurrent processing and profiling."""
    
    # Initialize profiler
    profiler.start_time = time.time()
    
    # Create profiled API manager and grader, with no caching
    api_manager = ProfiledAPIManager(
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
            # Set current problem for profiling
            profiler.set_current_problem(problem.problem_id)
            
            # Create a dedicated grader for this problem
            problem_grader = ProfiledTestExecutionGrader(
                executor_type=executor_type,
                timeout=timeout,
                together_api_key=together_api_key,
            )
            
            # Create a dedicated generator for this problem
            problem_generator = ProfiledGeneratorWithFeedback(
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
                error_log_path=error_log_path,
            )
            
            # Save result immediately if successful
            if result is not None:
                async with profile_async_operation("file_io"):
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
    
    profiler.end_time = time.time()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Completed processing in {duration:.2f} seconds")
    logger.info(f"Successfully generated {len(successful_results)}/{len(problems)} solutions")
    logger.info(f"Results saved incrementally to {output_path}")
    
    # Get profiling statistics
    profiling_stats = profiler.get_stats()
    
    return successful_results, profiling_stats


def load_existing_problem_ids(output_path: Path) -> Set[str]:
    """Load problem IDs that already exist in the output file."""
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
    """Load problem IDs that are already marked as impossible in the error log."""
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
    """Load problems from file, skipping those with existing solutions or marked as impossible."""
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


def print_profiling_report(stats: Dict[str, Any]):
    """Print a detailed profiling report with per-problem percentages."""
    print("\n" + "="*80)
    print("PROFILING REPORT (Per-Problem Percentages)")
    print("="*80)
    
    if '_overall' in stats:
        overall = stats['_overall']
        print(f"\nOVERALL:")
        print(f"  Total Runtime: {overall['total_runtime']:.2f}s")
        print(f"  Problems Processed: {overall['num_problems_processed']}")
        print(f"  Avg Time per Problem: {overall['avg_time_per_problem']:.2f}s")
    
    # Group operations by category
    llm_ops = {}
    exec_ops = {}
    other_ops = {}
    
    for op_type, data in stats.items():
        if op_type.startswith('_'):
            continue
        if 'llm' in op_type or 'generation' in op_type:
            llm_ops[op_type] = data
        elif 'execution' in op_type or 'code' in op_type:
            exec_ops[op_type] = data
        else:
            other_ops[op_type] = data
    
    # Print categorized results with per-problem percentages
    categories = [
        ("LLM OPERATIONS", llm_ops),
        ("CODE EXECUTION", exec_ops),
        ("OTHER OPERATIONS", other_ops)
    ]
    
    for category_name, category_ops in categories:
        if not category_ops:
            continue
            
        print(f"\n{category_name}:")
        category_avg_percentage = sum(data['avg_percentage_per_problem'] for data in category_ops.values())
        
        for op_type, data in sorted(category_ops.items(), key=lambda x: x[1]['avg_percentage_per_problem'], reverse=True):
            print(f"  {op_type:25} {data['avg_percentage_per_problem']:5.1f}% per problem | "
                  f"Avg Time: {data['avg_time_per_problem']:6.3f}s | "
                  f"Problems: {data['count_problems']:2d} | "
                  f"Range: {data['min_percentage']:.1f}% - {data['max_percentage']:.1f}%")
        
        print(f"  {'CATEGORY TOTAL':25} {category_avg_percentage:5.1f}% per problem")
    
    print("="*80)


async def main():
    """CLI for the profiled code scraper."""
    parser = argparse.ArgumentParser(
        description="Generate code solutions with test execution feedback and profiling",
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
        "--profile-output",
        type=Path,
        help="Path to save profiling results as JSON"
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
    
    logger.info(f"Starting profiled scraper with parameters:")
    logger.info(f"  Model: {args.model} ({args.provider})")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max turns: {args.max_turns}")
    logger.info(f"  Should pass private: {args.should_pass_private}")
    logger.info(f"  Max concurrent: {args.max_concurrent}")
    logger.info(f"  Problems to process: {len(problems)}")
    
    # Run profiled scraper
    results, profiling_stats = await scrape_solutions(
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
        error_log_path=args.error_log_path,
    )
    
    # Print profiling report
    print_profiling_report(profiling_stats)
    
    # Save profiling results if requested
    if args.profile_output:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.profile_output, 'w') as f:
            json.dump(profiling_stats, f, indent=2)
        print(f"\nProfiling results saved to: {args.profile_output}")
    
    print(f"\nGenerated {len(results)} solutions successfully!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())