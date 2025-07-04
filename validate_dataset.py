#!/usr/bin/env python3
"""Validate correct and broken test cases in a JSONL dataset."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict
from tqdm.asyncio import tqdm_asyncio

from code_data.dataset_loader import CodeDataLoader
from code_data.generation.models import CodeProblem, TestCase
from code_data.generation.executor import test_solution


async def validate_correct_tests(problem: CodeProblem) -> Tuple[List[TestCase], List[TestCase]]:
    """Validate correct test cases against the correct solution.
    
    Returns:
        Tuple of (passing_tests, failing_tests)
    """
    passing_tests = []
    failing_tests = []
    
    for tc in problem.test_cases:
        passed, output = await test_solution(
            problem.correct_solution,
            problem.function_name,
            tc.input,
            tc.expected_output
        )
        if passed:
            passing_tests.append(tc)
        else:
            failing_tests.append(tc)
            
    return passing_tests, failing_tests


async def validate_broken_tests(problem: CodeProblem) -> Tuple[List[TestCase], List[TestCase]]:
    """Validate broken test cases fail against the correct solution.
    
    Returns:
        Tuple of (correctly_failing_tests, incorrectly_passing_tests)
    """
    correctly_failing = []
    incorrectly_passing = []
    
    for tc in problem.broken_test_cases:
        passed, output = await test_solution(
            problem.correct_solution,
            problem.function_name,
            tc.input,
            tc.expected_output
        )
        if not passed:  # Should fail for broken tests
            correctly_failing.append(tc)
        else:
            incorrectly_passing.append(tc)
            
    return correctly_failing, incorrectly_passing


async def validate_and_filter_problem(
    problem: CodeProblem,
    validate_correct: bool = True,
    validate_broken: bool = True
) -> Tuple[CodeProblem, Dict]:
    """Validate and filter a single problem's test cases.
    
    Returns:
        Tuple of (filtered_problem, validation_results)
        filtered_problem will be None if no valid test cases remain
    """
    results = {
        'problem_id': problem.problem_id,
        'has_correct_solution': bool(problem.correct_solution),
        'original_correct_tests': len(problem.test_cases),
        'original_broken_tests': len(problem.broken_test_cases),
    }
    
    if not problem.correct_solution:
        results['error'] = 'No correct solution provided'
        results['filtered_correct_tests'] = 0
        results['filtered_broken_tests'] = 0
        return None, results
    
    # Start with all test cases
    filtered_correct_tests = list(problem.test_cases)
    filtered_broken_tests = list(problem.broken_test_cases)
    
    # Validate and filter correct test cases
    if validate_correct and problem.test_cases:
        passing, failing = await validate_correct_tests(problem)
        results['correct_tests_passing'] = len(passing)
        results['correct_tests_failing'] = len(failing)
        
        # Keep only passing tests
        filtered_correct_tests = passing
        
        # Remove corresponding broken tests for failing correct tests
        if problem.broken_test_cases and len(problem.broken_test_cases) == len(problem.test_cases):
            # Build index mapping for failing tests
            failing_indices = set()
            for i, tc in enumerate(problem.test_cases):
                if tc in failing:
                    failing_indices.add(i)
            
            # Keep only broken tests whose corresponding correct test passed
            filtered_broken_tests = [
                btc for i, btc in enumerate(problem.broken_test_cases)
                if i not in failing_indices
            ]
    
    # Validate remaining broken test cases
    if validate_broken and filtered_broken_tests:
        # Create a temporary problem with filtered tests for validation
        temp_problem = CodeProblem(
            problem_id=problem.problem_id,
            description=problem.description,
            test_cases=filtered_correct_tests,
            dataset=problem.dataset,
            function_name=problem.function_name,
            broken_test_cases=filtered_broken_tests,
            correct_solution=problem.correct_solution,
            difficulty=problem.difficulty,
            tags=problem.tags
        )
        
        correctly_failing, incorrectly_passing = await validate_broken_tests(temp_problem)
        results['broken_tests_failing'] = len(correctly_failing)
        results['broken_tests_passing'] = len(incorrectly_passing)
        
        # Remove incorrectly passing broken tests AND their corresponding correct tests
        if incorrectly_passing and len(filtered_correct_tests) == len(filtered_broken_tests):
            # Find indices of incorrectly passing broken tests
            bad_indices = set()
            for i, btc in enumerate(filtered_broken_tests):
                if btc in incorrectly_passing:
                    bad_indices.add(i)
            
            # Filter both correct and broken tests
            filtered_correct_tests = [
                tc for i, tc in enumerate(filtered_correct_tests)
                if i not in bad_indices
            ]
            filtered_broken_tests = [
                btc for i, btc in enumerate(filtered_broken_tests)
                if i not in bad_indices
            ]
    
    results['filtered_correct_tests'] = len(filtered_correct_tests)
    results['filtered_broken_tests'] = len(filtered_broken_tests)
    
    # If no correct tests remain, discard the problem
    if not filtered_correct_tests:
        return None, results
    
    # Create filtered problem
    filtered_problem = CodeProblem(
        problem_id=problem.problem_id,
        description=problem.description,
        test_cases=filtered_correct_tests,
        dataset=problem.dataset,
        function_name=problem.function_name,
        broken_test_cases=filtered_broken_tests,
        correct_solution=problem.correct_solution,
        difficulty=problem.difficulty,
        tags=problem.tags,
        prompt=problem.prompt,
        full_completion=problem.full_completion,
        parsed_completion=problem.parsed_completion,
        mixed_tests=problem.mixed_tests
    )
    
    return filtered_problem, results


async def validate_dataset(
    dataset_path: str,
    validate_correct: bool = True,
    validate_broken: bool = True,
    max_concurrent: int = 5,
    output_valid_path: str = None,
    output_report_path: str = None
) -> Dict:
    """Validate all problems in a dataset.
    
    Args:
        dataset_path: Path to JSONL dataset
        validate_correct: Whether to validate correct test cases
        validate_broken: Whether to validate broken test cases
        max_concurrent: Maximum concurrent validations
        output_valid_path: Optional path to save only valid problems
        output_report_path: Optional path to save validation report
        
    Returns:
        Dict with overall validation statistics
    """
    # Load dataset
    problems = CodeDataLoader.load_completion_dataset(dataset_path)
    print(f"Loaded {len(problems)} problems from {dataset_path}")
    
    # Shared state for progress bar updates
    completed_count = 0
    valid_count = 0
    
    # Validate with concurrency control
    sem = asyncio.Semaphore(max_concurrent)
    
    async def validate_with_sem(problem: CodeProblem, pbar) -> Tuple[CodeProblem, Dict]:
        nonlocal completed_count, valid_count
        async with sem:
            filtered_problem, result = await validate_and_filter_problem(problem, validate_correct, validate_broken)
            
            # Update counts
            completed_count += 1
            if filtered_problem is not None:
                valid_count += 1
            
            # Update progress bar with valid rate
            valid_rate = (valid_count / completed_count * 100) if completed_count > 0 else 0
            pbar.set_postfix({'valid_rate': f'{valid_rate:.1f}%'})
            pbar.update(1)
            
            return filtered_problem, result
    
    # Run validations with progress bar
    print(f"Validating problems (correct={validate_correct}, broken={validate_broken})...")
    with tqdm_asyncio(total=len(problems), desc="Validating") as pbar:
        tasks = [validate_with_sem(p, pbar) for p in problems]
        problem_result_pairs = await asyncio.gather(*tasks)
    
    # Separate filtered problems and results
    filtered_problems = [p for p, _ in problem_result_pairs if p is not None]
    results = [r for _, r in problem_result_pairs]
    
    # Compute statistics
    stats = {
        'total_problems': len(problems),
        'problems_with_solutions': sum(1 for r in results if r['has_correct_solution']),
        'problems_without_solutions': sum(1 for r in results if not r['has_correct_solution']),
        'problems_kept': len(filtered_problems),
        'problems_discarded': len(problems) - len(filtered_problems),
    }
    
    if validate_correct:
        stats['total_original_correct_tests'] = sum(r['original_correct_tests'] for r in results)
        stats['total_filtered_correct_tests'] = sum(r['filtered_correct_tests'] for r in results)
        stats['total_passing_correct_tests'] = sum(
            r.get('correct_tests_passing', 0) for r in results
        )
        stats['total_failing_correct_tests'] = sum(
            r.get('correct_tests_failing', 0) for r in results
        )
        stats['correct_tests_removed'] = stats['total_original_correct_tests'] - stats['total_filtered_correct_tests']
    
    if validate_broken:
        stats['total_original_broken_tests'] = sum(r['original_broken_tests'] for r in results)
        stats['total_filtered_broken_tests'] = sum(r['filtered_broken_tests'] for r in results)
        stats['total_correctly_failing_broken_tests'] = sum(
            r.get('broken_tests_failing', 0) for r in results
        )
        stats['total_incorrectly_passing_broken_tests'] = sum(
            r.get('broken_tests_passing', 0) for r in results
        )
        stats['broken_tests_removed'] = stats['total_original_broken_tests'] - stats['total_filtered_broken_tests']
    
    # Save filtered problems if requested
    if output_valid_path and filtered_problems:
        CodeDataLoader.save_dataset_to_file(filtered_problems, output_valid_path)
        print(f"\nSaved {len(filtered_problems)} filtered problems to {output_valid_path}")
    
    # Save detailed report if requested
    if output_report_path:
        report = {
            'summary': stats,
            'problems': results
        }
        with open(output_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved validation report to {output_report_path}")
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Problems kept: {stats['problems_kept']} ({stats['problems_kept']/stats['total_problems']*100:.1f}%)")
    print(f"Problems discarded: {stats['problems_discarded']} (no valid test pairs remaining)")
    
    if validate_correct:
        print(f"\nCorrect Tests:")
        print(f"  Original: {stats['total_original_correct_tests']}")
        print(f"  Kept: {stats['total_filtered_correct_tests']}")
        print(f"  Removed: {stats['correct_tests_removed']} ({stats['total_failing_correct_tests']} failed validation)")
    
    if validate_broken:
        print(f"\nBroken Tests:")
        print(f"  Original: {stats['total_original_broken_tests']}")
        print(f"  Kept: {stats['total_filtered_broken_tests']}")
        print(f"  Removed: {stats['broken_tests_removed']} ({stats['total_incorrectly_passing_broken_tests']} incorrectly passed)")
    
    # Print specific filtering details
    print("\nFiltering Details (first 5 problems with removed tests):")
    shown = 0
    for result in results:
        tests_removed = (result['original_correct_tests'] - result['filtered_correct_tests'])
        if tests_removed > 0 and shown < 5:
            print(f"\nProblem {result['problem_id']}:")
            print(f"  Tests removed: {tests_removed} pairs")
            print(f"  Correct tests: {result['original_correct_tests']} → {result['filtered_correct_tests']}")
            print(f"  Broken tests: {result['original_broken_tests']} → {result['filtered_broken_tests']}")
            if result.get('correct_tests_failing', 0) > 0:
                print(f"  Reason: {result['correct_tests_failing']} correct tests failed")
            if result.get('broken_tests_passing', 0) > 0:
                print(f"  Reason: {result['broken_tests_passing']} broken tests incorrectly passed")
            shown += 1
    
    problems_with_removals = sum(1 for r in results if r['original_correct_tests'] > r['filtered_correct_tests'])
    if problems_with_removals > shown:
        print(f"\n... and {problems_with_removals - shown} more problems had test pairs removed")
    
    # Create histogram of test counts
    from collections import Counter
    test_counts = [p.test_cases.__len__() for p in filtered_problems if p is not None]
    
    if test_counts:
        print("\n=== Test Count Distribution ===")
        counter = Counter(test_counts)
        max_count = max(counter.keys())
        max_freq = max(counter.values())
        
        # Determine bar width for scaling
        bar_width = 40
        
        for num_tests in range(1, max_count + 1):
            count = counter.get(num_tests, 0)
            if count > 0:
                # Scale bar length
                bar_length = int((count / max_freq) * bar_width)
                bar = '█' * bar_length
                
                # Format the line
                print(f"{num_tests:3d} tests: {bar:<{bar_width}} {count:4d} problems ({count/len(filtered_problems)*100:5.1f}%)")
        
        print(f"\nTotal problems: {len(filtered_problems)}")
        print(f"Average tests per problem: {sum(test_counts)/len(test_counts):.1f}")
        print(f"Min tests: {min(test_counts)}, Max tests: {max(test_counts)}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate correct and broken test cases in a JSONL dataset"
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to JSONL dataset file'
    )
    parser.add_argument(
        '--validate-correct',
        action='store_true',
        default=True,
        help='Validate that correct test cases pass (default: True)'
    )
    parser.add_argument(
        '--no-validate-correct',
        dest='validate_correct',
        action='store_false',
        help='Skip validation of correct test cases'
    )
    parser.add_argument(
        '--validate-broken',
        action='store_true',
        default=True,
        help='Validate that broken test cases fail (default: True)'
    )
    parser.add_argument(
        '--no-validate-broken',
        dest='validate_broken',
        action='store_false',
        help='Skip validation of broken test cases'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent validations (default: 5)'
    )
    parser.add_argument(
        '--output-valid',
        type=str,
        help='Path to save only valid problems'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        help='Path to save detailed validation report as JSON'
    )
    
    args = parser.parse_args()
    
    # Run validation
    asyncio.run(validate_dataset(
        dataset_path=args.dataset,
        validate_correct=args.validate_correct,
        validate_broken=args.validate_broken,
        max_concurrent=args.max_concurrent,
        output_valid_path=args.output_valid,
        output_report_path=args.output_report
    ))


if __name__ == '__main__':
    main()