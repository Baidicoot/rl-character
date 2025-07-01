#!/usr/bin/env python3
"""Build and save a static dataset with broken test cases."""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List
from .models import CodeProblem

# Handle both module and script execution
if __name__ == '__main__':
    # Add parent directory to path when run as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .load import load_mbpp_problems, load_codeforces_problems, load_apps_problems
    from .dataset import add_broken_tests_to_problems
else:
    from .load import load_mbpp_problems, load_codeforces_problems, load_apps_problems
    from .dataset import add_broken_tests_to_problems


async def _load_and_process_problems(
    dataset_name: str,
    num_problems: int,
    start_idx: int,
    broken_test_model: str,
    max_concurrent: int
):
    """
    Shared logic for loading problems and generating broken tests.
    
    Args:
        dataset_name: Dataset to use (mbpp, codeforces, or apps)
        num_problems: Number of problems to include
        start_idx: Starting index in dataset
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        
    Returns:
        Tuple of (all_problems, problems_with_broken_tests)
    """
    # Step 1: Load problems
    print(f"\n1. Loading {dataset_name} problems...")
    if dataset_name == "mbpp":
        problems = load_mbpp_problems(num_problems, start_idx)
    elif dataset_name == "codeforces":
        problems = load_codeforces_problems(num_problems=num_problems, start_idx=start_idx)
    elif dataset_name == "apps":
        problems = await load_apps_problems(num_problems=num_problems, start_idx=start_idx, max_concurrent=max_concurrent)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    print(f"   Loaded {len(problems)} problems")
    
    # Step 2: Generate broken test cases
    print("\n2. Generating broken test cases...")
    problems = await add_broken_tests_to_problems(
        problems = problems,
        model = broken_test_model,
        max_concurrent = max_concurrent
    )
    
    # Filter to only keep problems with broken tests
    problems_with_test_cases = [p for p in problems if len(p.test_cases) > 0]
    filtered_tc = len(problems) - len(problems_with_test_cases)
    problems_with_broken = [p for p in problems_with_test_cases if p.broken_test_cases]
    filtered_broken = len(problems_with_test_cases) - len(problems_with_broken)
    
    if filtered_tc > 0:
        print(f"   Filtered out {filtered_tc} problems without test cases")
    if filtered_broken > 0:
        print(f"   Filtered out {filtered_broken} problems with test cases but without broken tests")
    
    return problems, problems_with_broken


def _create_dataset_dict(problems: List[CodeProblem],
                        source_dataset: str, 
                         split_name: str, 
                         broken_test_model: str, 
                         start_idx: int):
    """
    Create dataset dictionary from problems.
    
    Args: 
        problems: List of problems to include
        source_dataset: Source dataset name
        split_name: Split name
        broken_test_model: Model used for broken tests
        start_idx: Starting index used
        all_problems_count: Total problems attempted 
        
    Returns:
        Dictionary with metadata and problems
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_problems": len(problems),
        "broken_test_model": broken_test_model,
        "source_dataset": source_dataset,
        "split_name": split_name,
        "start_idx": start_idx
    }
    
    return {
        "metadata": metadata,
        "problems": [
            {
                "problem_id": p.problem_id,
                "description": p.description,
                "function_name": p.function_name,
                "correct_solution": p.correct_solution,
                "dataset": p.dataset,
                "difficulty": p.difficulty,
                "tags": p.tags,
                "test_cases": [
                    {"input": tc.input, "output": tc.expected_output}
                    for tc in p.test_cases
                ],
                "broken_test_cases": [
                    {"input": tc.input, "output": tc.expected_output}
                    for tc in p.broken_test_cases
                ]
            }
            for p in problems
        ]
    }


async def build_full_dataset(
    source_dataset: str = "mbpp",
    num_problems: int = 50,
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0,
):
    """
    Build a static dataset with broken test cases.
    
    Args:
        dataset_name: Dataset to use (mbpp, codeforces, or apps)
        num_problems: Number of problems to include
        output_file: Where to save the dataset
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        start_idx: Starting index in dataset
        
    Returns:
        Path to the saved dataset
    """
    
    output_path = split_dataset(source_dataset, 
                                num_problems, 
                                ["full"], 
                                [1.0], 
                                broken_test_model, 
                                max_concurrent, 
                                start_idx)
    
    print(f"Dataset saved to: {output_path}")
    return output_path


async def split_dataset(
    source_dataset: str = "mbpp",
    num_problems: int = 50,
    splits: List[str] = ["train", "test"],
    ratios: List[float] = [0.8, 0.2],
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0
):
    """
    Build train and test datasets with broken test cases.
    
    Args:
        source_dataset: Dataset to use (mbpp, codeforces, or apps)
        num_problems: Number of problems to include total
        splits: List of split names
        ratios: List of ratios for each split
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        start_idx: Starting index in dataset
        
    Returns:
        Tuple of (train_path, test_path)
    """
    print(f"=== Building {source_dataset} Train/Test Datasets with Broken Tests ===")
    print(f"Total problems: {num_problems}")
    print(f"Train ratio: {ratios}")
    print(f"Model for broken tests: {broken_test_model}")
    print(f"Splits: {splits}")
    print(f"Ratios: {ratios}")

    assert len(splits) == len(ratios), "Number of splits and ratios must match"
    assert sum(ratios) == 1, "Ratios must sum to 1"
    
    # Load and process problems using shared logic
    all_problems, problems_with_broken = await _load_and_process_problems(
        dataset_name = source_dataset,
        num_problems = num_problems,
        start_idx = start_idx,
        broken_test_model = broken_test_model,
        max_concurrent = max_concurrent
    )
    
    # Split into train/test sets
    import random
    random.shuffle(problems_with_broken)
    
    output_paths = []
    print(f"\n3. Split datasets:")

    split_sizes = [int(len(problems_with_broken) * ratio) for ratio in ratios]
    for split, split_size in zip(splits, split_sizes):
        split_problems = problems_with_broken[:split_size]
        problems_with_broken = problems_with_broken[split_size:]
        
        print(f"   {split}: {len(split_problems)} problems")
        
        # Create dataset dictionaries using shared logic
        dataset = _create_dataset_dict(
            split_problems, 
            source_dataset = source_dataset,
            split_name = split,
            broken_test_model = broken_test_model,
            start_idx = start_idx
        )
    
        # Save train dataset
        output_path = Path(f"datasets/code/{source_dataset}/{split}/{broken_test_model}.json")
        print(f"\n4. Saving {split} dataset to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f: # override existing file
            json.dump(dataset, f, indent=2)
            
        output_paths.append(output_path)
    
        # Print summary
        print("\n=== Split Summary ===")
        print(f"Split name: {split}")
        print(f"Total problems: {len(split_problems)}")
        print(f"Saved to: {output_path}")
    
    return output_paths