#!/usr/bin/env python3
"""Check deepcoder dataset for func_name in functional test cases."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from code_generation.models import CodeProblem


def check_deepcoder_functional_tests(dataset_path: Path = Path("datasets/deepcoder_preprocessed.jsonl")) -> Tuple[List[CodeProblem], Dict[str, Any]]:
    """Load CodeProblems from deepcoder dataset and check functional test metadata.
    
    Args:
        dataset_path: Path to the deepcoder preprocessed dataset
        
    Returns:
        Tuple of (list of CodeProblems, statistics dict)
    """
    import logging
    
    problems = []
    stats = {
        "total_problems": 0,
        "functional_problems": 0,
        "stdin_problems": 0,
        "mixed_problems": 0,
        "functional_with_func_name": 0,
        "functional_without_func_name": 0,
        "problems_without_func_name": [],
        "func_name_variations": {}  # Track different field names used
    }
    
    print(f"Loading problems from {dataset_path}")
    
    # Common variations of function name fields
    func_name_fields = ["func_name", "function_name", "fn_name", "function", "entry_point", "method_name"]
    
    with open(dataset_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                problem = CodeProblem.from_dict(data)
                problems.append(problem)
                
                stats["total_problems"] += 1
                
                # Check test case types
                test_types = set()
                for tc in problem.test_cases:
                    test_types.add(tc.type)
                
                # Categorize problem by test types
                if test_types == {"functional"}:
                    stats["functional_problems"] += 1
                    
                    # Check for any function name field variation
                    found_func_name = False
                    for field in func_name_fields:
                        if field in problem.metadata:
                            found_func_name = True
                            if field != "func_name":
                                # Track non-standard field names
                                stats["func_name_variations"][field] = stats["func_name_variations"].get(field, 0) + 1
                                logging.debug(f"Problem {problem.problem_id} has '{field}' instead of 'func_name': {problem.metadata[field]}")
                            break
                    
                    if found_func_name:
                        stats["functional_with_func_name"] += 1
                    else:
                        stats["functional_without_func_name"] += 1
                        stats["problems_without_func_name"].append({
                            "problem_id": problem.problem_id,
                            "line_number": line_num,
                            "metadata_keys": list(problem.metadata.keys()),
                            "problem_preview": problem.problem[:100] + "..." if len(problem.problem) > 100 else problem.problem
                        })
                        
                elif test_types == {"stdin"}:
                    stats["stdin_problems"] += 1
                elif len(test_types) > 1:
                    stats["mixed_problems"] += 1
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return problems, stats


def print_statistics(stats: Dict[str, Any]):
    """Print statistics about the dataset."""
    print(f"\n=== Dataset Statistics ===")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Functional problems: {stats['functional_problems']}")
    print(f"Stdin problems: {stats['stdin_problems']}")
    print(f"Mixed type problems: {stats['mixed_problems']}")
    
    if stats['functional_problems'] > 0:
        print(f"\n=== Functional Problem Analysis ===")
        print(f"With func_name (or variation): {stats['functional_with_func_name']}")
        print(f"Without any func_name: {stats['functional_without_func_name']}")
        
        if stats.get('func_name_variations'):
            print(f"\n=== Function Name Field Variations Found ===")
            for field, count in stats['func_name_variations'].items():
                print(f"  {field}: {count} occurrences")
        
        if stats['functional_without_func_name'] > 0:
            print(f"\n=== Problems Missing func_name ===")
            for i, problem_info in enumerate(stats['problems_without_func_name'][:5], 1):
                print(f"\n{i}. Problem ID: {problem_info['problem_id']} (line {problem_info['line_number']})")
                print(f"   Metadata keys: {problem_info['metadata_keys']}")
                print(f"   Problem: {problem_info['problem_preview']}")
            
            if len(stats['problems_without_func_name']) > 5:
                print(f"\n... and {len(stats['problems_without_func_name']) - 5} more problems")


def check_and_fix_func_names(problems: List[CodeProblem]) -> List[CodeProblem]:
    """Check for func_name in functional problems and attempt to extract if missing.
    
    Also standardizes various function name field variations to 'func_name'.
    
    Args:
        problems: List of CodeProblems to check
        
    Returns:
        List of CodeProblems with func_name added where possible
    """
    import re
    import logging
    
    fixed_count = 0
    standardized_count = 0
    
    # Common variations of function name fields
    func_name_fields = ["function_name", "fn_name", "function", "entry_point", "method_name"]
    
    for problem in problems:
        # Only process functional problems
        if all(tc.type == "functional" for tc in problem.test_cases):
            # First, standardize any existing function name fields to 'func_name'
            for field in func_name_fields:
                if field in problem.metadata:
                    existing_value = problem.metadata[field]
                    problem.metadata["func_name"] = existing_value
                    del problem.metadata[field]
                    standardized_count += 1
                    logging.debug(f"Standardized '{field}' to 'func_name' for problem {problem.problem_id}: {existing_value}")
                    print(f"Standardized '{field}' -> 'func_name' for problem {problem.problem_id}: {existing_value}")
                    break
            
            # If still no func_name, try to extract it
            if "func_name" not in problem.metadata:
                func_name = None
                
                # Method 1: Look for "def function_name(" in solutions
                for solution in problem.solutions:
                    match = re.search(r'def\s+(\w+)\s*\(', solution)
                    if match:
                        func_name = match.group(1)
                        logging.debug(f"Extracted func_name '{func_name}' from solution for problem {problem.problem_id}")
                        break
                
                # Method 2: Look for function name in problem statement
                if not func_name:
                    # Common patterns: "Write a function named X", "implement function X", etc.
                    patterns = [
                        r'function\s+(?:named\s+|called\s+)?`?(\w+)`?',
                        r'implement\s+`?(\w+)`?',
                        r'`(\w+)`\s+function',
                        r'def\s+(\w+)',
                        r'create\s+a\s+function\s+`?(\w+)`?',
                        r'function\s+`?(\w+)`?\s+that',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, problem.problem, re.IGNORECASE)
                        if match:
                            func_name = match.group(1)
                            logging.debug(f"Extracted func_name '{func_name}' from problem statement using pattern '{pattern}' for problem {problem.problem_id}")
                            break
                
                if func_name:
                    problem.metadata["func_name"] = func_name
                    fixed_count += 1
                    print(f"Extracted and added func_name '{func_name}' to problem {problem.problem_id}")
                else:
                    logging.warning(f"Could not extract func_name for problem {problem.problem_id}")
                    print(f"Warning: Could not extract func_name for problem {problem.problem_id}")
    
    print(f"\nStandardized {standardized_count} existing function name fields")
    print(f"Fixed {fixed_count} problems by extracting func_name")
    print(f"Total problems updated: {standardized_count + fixed_count}")
    return problems


def save_fixed_dataset(problems: List[CodeProblem], output_path: Path):
    """Save the fixed dataset to a new file.
    
    Args:
        problems: List of CodeProblems to save
        output_path: Path to save the fixed dataset
    """
    with open(output_path, 'w') as f:
        for problem in problems:
            json.dump(problem.to_dict(), f)
            f.write('\n')
    
    print(f"\nSaved fixed dataset to {output_path}")


def main():
    """Main function to check and optionally fix the dataset."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description="Check deepcoder dataset for func_name metadata")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/deepcoder_preprocessed.jsonl"),
        help="Path to deepcoder dataset"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix missing func_names"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/deepcoder_preprocessed_fixed.jsonl"),
        help="Path to save fixed dataset (only used with --fix)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if dataset exists
    if not args.dataset_path.exists():
        print(f"Error: Dataset not found at {args.dataset_path}")
        return
    
    # Load and check dataset
    problems, stats = check_deepcoder_functional_tests(args.dataset_path)
    print_statistics(stats)
    
    # Optionally fix missing func_names
    if args.fix and stats['functional_without_func_name'] > 0:
        print("\n=== Attempting to fix missing func_names ===")
        problems = check_and_fix_func_names(problems)
        
        # Re-check statistics on the fixed problems
        print("\n=== Re-checking statistics on fixed data ===")
        fixed_stats = {
            "total_problems": 0,
            "functional_problems": 0,
            "stdin_problems": 0,
            "mixed_problems": 0,
            "functional_with_func_name": 0,
            "functional_without_func_name": 0,
            "problems_without_func_name": [],
            "func_name_variations": {}
        }
        
        for problem in problems:
            fixed_stats["total_problems"] += 1
            test_types = set(tc.type for tc in problem.test_cases)
            
            if test_types == {"functional"}:
                fixed_stats["functional_problems"] += 1
                if "func_name" in problem.metadata:
                    fixed_stats["functional_with_func_name"] += 1
                else:
                    fixed_stats["functional_without_func_name"] += 1
            elif test_types == {"stdin"}:
                fixed_stats["stdin_problems"] += 1
            else:
                fixed_stats["mixed_problems"] += 1
        
        print_statistics(fixed_stats)
        
        # Save fixed dataset
        save_fixed_dataset(problems, args.output_path)


if __name__ == "__main__":
    main()