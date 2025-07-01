#!/usr/bin/env python3
"""Build a static MBPP dataset with broken test cases."""

import asyncio
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from .dataset import load_mbpp_problems, add_broken_tests_to_problems, load_dataset_from_file, save_dataset_to_file
from typing import Optional

async def build_dataset(
    num_problems: int = 50,
    output_file: str = "mbpp_dataset_with_broken_tests.json",
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0,
    dataset_path: Optional[str] = None,
    force_provider: str = None,
    train_test_split: Optional[float] = None
):
    """
    Build a static dataset with broken test cases.
    
    Args:
        num_problems: Number of MBPP problems to include
        output_file: Where to save the dataset
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        start_idx: Starting index in MBPP dataset
        
    Returns:
        Path to the saved dataset
    """
    print(f"=== Building MBPP Dataset with Broken Tests ===")
    print(f"Problems: {num_problems}")
    print(f"Model for broken tests: {broken_test_model}")
    print(f"Output: {output_file}")
    
    # Step 1: Load MBPP problems
    print("\n1. Loading MBPP problems...")
    if dataset_path is None:
        problems = load_mbpp_problems(num_problems, start_idx)
    else:
        problems = load_dataset_from_file(dataset_path)
    print(f"   Loaded {len(problems)} problems")
    
    # Step 2: Generate broken test cases
    print("\n2. Generating broken test cases...")
    problems = await add_broken_tests_to_problems(
        problems,
        broken_test_model,
        max_concurrent,
        force_provider=force_provider
    )
    
    # Filter to only keep problems with broken tests
    problems_with_broken = [p for p in problems if p.broken_test_cases]
    filtered_count = len(problems) - len(problems_with_broken)
    
    if filtered_count > 0:
        print(f"   Filtered out {filtered_count} problems without broken tests")
    
    print(f"   Keeping {len(problems_with_broken)} problems with broken tests")
    
    # Step 3: Convert to serializable format
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "num_problems": len(problems_with_broken),
            "num_problems_attempted": len(problems),
            "broken_test_model": broken_test_model,
            "problems_with_broken_tests": len(problems_with_broken),
            "mbpp_start_idx": start_idx
        },
        "problems": []
    }
    
    for p in problems_with_broken:
        problem_data = {
            "task_id": p.task_id,
            "problem_id": p.problem_id,
            "description": p.description,
            "function_name": p.function_name,
            "correct_solution": p.correct_solution,
            "test_cases": [
                {"input": tc.input, "output": tc.expected_output}
                for tc in p.test_cases
            ],
            "broken_test_cases": [
                {"input": tc.input, "output": tc.expected_output}
                for tc in p.broken_test_cases
            ]
        }
        dataset["problems"].append(problem_data)
    
    # Step 4: Save dataset
    print(f"\n3. Saving dataset to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if train_test_split is None:
        save_dataset_to_file(problems_with_broken, output_path)
    else:
        assert 0 < train_test_split < 1, "Train/test split ratio must be between 0 and 1"
        num_train = int(len(problems_with_broken) * train_test_split)
        train_problems = problems_with_broken[:num_train]
        test_problems = problems_with_broken[num_train:]
        save_dataset_to_file(train_problems, output_path.with_suffix(".train.json"))
        save_dataset_to_file(test_problems, output_path.with_suffix(".test.json"))
        print(f"Train problems: {len(train_problems)}")
        print(f"Test problems: {len(test_problems)}")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Problems attempted: {len(problems)}")
    print(f"Problems kept (with broken tests): {len(problems_with_broken)}")
    print(f"Problems filtered out: {filtered_count}")
    print(f"Total correct test cases: {sum(len(p.test_cases) for p in problems_with_broken)}")
    print(f"Total broken test cases: {sum(len(p.broken_test_cases) for p in problems_with_broken)}")
    print(f"Dataset saved to: {output_path.absolute()}")
    
    return output_path