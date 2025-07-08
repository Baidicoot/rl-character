#!/usr/bin/env python3
"""Build and save a static dataset with broken test cases."""

import os
import sys
from pathlib import Path
from typing import List

# Handle both module and script execution
if __name__ == "__main__":
    # Add parent directory to path when run as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .load import load_mbpp_problems, load_apps_problems
    from .dataset import add_broken_outputs_to_problems
    from ..dataset_loader import CodeDataLoader
else:
    from .load import load_mbpp_problems, load_apps_problems
    from .dataset import add_broken_outputs_to_problems
    from ..dataset_loader import CodeDataLoader


async def _load_and_process_problems(
    dataset_name: str,
    num_problems: int,
    start_idx: int,
    broken_test_model: str,
    max_concurrent: int,
    save_formatted: bool = False,
    formatted_output_path: str = None,
    filters: dict = None,
    max_retries: int = 3,
):
    """
    Shared logic for loading problems and generating broken tests.

    Args:
        dataset_name: Dataset to use (mbpp, codeforces, or apps)
        num_problems: Number of problems to include
        start_idx: Starting index in dataset
        broken_test_model: Model to use for generating broken tests
        max_concurrent: Maximum concurrent API calls
        save_formatted: Whether to save formatted dataset before generating broken tests
        formatted_output_path: Path to save formatted dataset
        filters: Filters to apply to source dataset
        max_retries: Maximum retry attempts for broken test generation

    Returns:
        Tuple of (all_problems, problems_with_broken_tests)
    """
    # Step 1: Load problems
    print(f"\n1. Loading {dataset_name} problems...")
    if dataset_name == "mbpp":
        problems = await load_mbpp_problems(num_problems, start_idx)
    elif dataset_name == "apps":
        problems = await load_apps_problems(
            num_problems=num_problems,
            start_idx=start_idx,
            max_concurrent=max_concurrent,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    print(f"   Loaded {len(problems)} problems")

    # Apply filters if provided
    if filters:
        print(f"   Applying filters: {filters}")
        problems = CodeDataLoader._apply_filters_to_single_dataset(problems, filters)
        print(f"   After filtering: {len(problems)} problems remain")

    # Save formatted dataset if requested
    if save_formatted and formatted_output_path:
        CodeDataLoader.save_dataset_to_file(problems, formatted_output_path)

    # Step 2: Generate broken test cases
    print("\n2. Generating broken test cases...")
    problems = await add_broken_outputs_to_problems(
        problems=problems,
        model=broken_test_model,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
    )

    # Filter to only keep problems with broken tests
    problems_with_test_cases = [p for p in problems if len(p.test_cases) > 0]
    filtered_tc = len(problems) - len(problems_with_test_cases)
    problems_with_broken = [
        p
        for p in problems_with_test_cases
        if any(tc.broken_output for tc in p.test_cases)
    ]
    filtered_broken = len(problems_with_test_cases) - len(problems_with_broken)

    if filtered_tc > 0:
        print(f"   Filtered out {filtered_tc} problems without test cases")
    if filtered_broken > 0:
        print(
            f"   Filtered out {filtered_broken} problems with test cases but without broken tests"
        )

    return problems, problems_with_broken


async def build_full_dataset(
    source_dataset: str = "mbpp",
    num_problems: int = 50,
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0,
    filters: dict = None,
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

    output_path = split_dataset(
        source_dataset,
        num_problems,
        ["full"],
        [1.0],
        broken_test_model,
        max_concurrent,
        start_idx,
        filters=filters,
    )

    print(f"Dataset saved to: {output_path}")
    return output_path


async def split_dataset(
    source_dataset: str = "mbpp",
    num_problems: int = 50,
    splits: List[str] = ["train", "test"],
    ratios: List[float] = [0.8, 0.2],
    broken_test_model: str = "claude-3-haiku-20240307",
    max_concurrent: int = 5,
    start_idx: int = 0,
    save_formatted: bool = False,
    dataset_filters: dict = None,
    max_retries: int = 3,
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
        save_formatted: Whether to save formatted dataset before generating broken tests
        dataset_filters: Filters to apply to source dataset
        max_retries: Maximum retry attempts for broken test generation

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

    # Generate formatted dataset path if needed
    formatted_path = None
    if save_formatted:
        formatted_path = (
            f"datasets/code/{source_dataset}/{source_dataset}_formatted.jsonl"
        )

    # Load and process problems using shared logic
    all_problems, problems_with_broken = await _load_and_process_problems(
        dataset_name=source_dataset,
        num_problems=num_problems,
        start_idx=start_idx,
        broken_test_model=broken_test_model,
        max_concurrent=max_concurrent,
        save_formatted=save_formatted,
        formatted_output_path=formatted_path,
        filters=dataset_filters,
        max_retries=max_retries,
    )

    # Split into train/test sets
    import random

    random.shuffle(problems_with_broken)

    output_paths = []
    print("\n3. Split datasets:")

    split_sizes = [int(len(problems_with_broken) * ratio) for ratio in ratios]
    for split, split_size in zip(splits, split_sizes):
        split_problems = problems_with_broken[:split_size]
        problems_with_broken = problems_with_broken[split_size:]

        print(f"   {split}: {len(split_problems)} problems")

        # Save dataset
        output_path = Path(
            f"datasets/code/{source_dataset}/{split}/{broken_test_model}.jsonl"
        )
        print(f"\n4. Saving {split} dataset to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        CodeDataLoader.save_dataset_to_file(split_problems, str(output_path))

        output_paths.append(output_path)

        # Print summary
        print("\n=== Split Summary ===")
        print(f"Split name: {split}")
        print(f"Total problems: {len(split_problems)}")
        print(f"Saved to: {output_path}")

    return output_paths
