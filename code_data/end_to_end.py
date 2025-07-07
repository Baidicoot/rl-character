#!/usr/bin/env python3
"""End-to-end dataset generation script for code datasets framework."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .generation.load import load_mbpp_problems, load_apps_problems
from .generation.dataset import add_broken_outputs_to_problems
from .dataset_loader import CodeDataLoader
from .generation.generator import generate_dataset_completions
from .prompts import code_generation, system, test_generation
from .generation.config import BrokenTestConfig, CodeGenerationConfig, EndToEndConfig

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Shared config classes are now imported from config.py

async def load_and_split_dataset(config: EndToEndConfig) -> Dict[str, str]:
    """
    Load problems from source dataset and split into multiple datasets.
    
    Returns:
        Dictionary mapping split names to their formatted dataset paths
    """
    print(f"=== Step 1: Loading and splitting {config.source_dataset} dataset ===")
    print(f"Splits: {config.splits}, Ratios: {config.ratios}")
    print(f"Total problems: {config.num_problems}, Start index: {config.start_idx}")
    
    # First, determine all output paths and check if they exist
    formatted_paths = {}
    missing_paths = []
    
    for split_name in config.splits:
        output_dir = Path(f"datasets/code/{config.source_dataset}/{split_name}")
        formatted_path = output_dir / f"{config.source_dataset}_formatted.jsonl"
        formatted_paths[split_name] = str(formatted_path)
        
        if not formatted_path.exists():
            missing_paths.append((split_name, formatted_path))
    
    # If all paths exist, skip loading entirely
    if not missing_paths:
        print("All formatted datasets already exist:")
        for split_name, path in formatted_paths.items():
            print(f"  {split_name}: {path}")
        return formatted_paths
    
    # At least one path is missing, so load and split the problems
    print(f"Missing {len(missing_paths)} formatted datasets, loading problems...")
    
    # Load problems from source dataset
    if config.source_dataset == "mbpp":
        problems = await load_mbpp_problems(num_problems=config.num_problems, start_idx=config.start_idx)
    elif config.source_dataset == "apps":
        problems = await load_apps_problems(
            num_problems=config.num_problems, 
            start_idx=config.start_idx, 
            max_concurrent=config.code_generation_config.max_concurrent
        )
    else:
        raise ValueError(f"Unsupported source dataset: {config.source_dataset}")
    
    print(f"Loaded {len(problems)} problems from {config.source_dataset}")
    
    # Apply dataset filters if specified
    if config.dataset_filters:
        print(f"Applying dataset filters: {config.dataset_filters}")
        from .dataset_loader import CodeDataLoader
        problems_unfiltered = len(problems)
        problems = CodeDataLoader._apply_filters_to_single_dataset(problems, config.dataset_filters)
        print(f"Filtered from {problems_unfiltered} to {len(problems)} problems")
    
    # Shuffle problems for random splitting
    import random
    random.shuffle(problems)
    
    # Split into multiple datasets
    split_sizes = [int(len(problems) * ratio) for ratio in config.ratios]
    
    for i, (split_name, split_size) in enumerate(zip(config.splits, split_sizes)):
        split_problems = problems[:split_size]
        problems = problems[split_size:]
        
        print(f"  {split_name}: {len(split_problems)} problems")
        
        # Create output directory and file path
        output_dir = Path(f"datasets/code/{config.source_dataset}/{split_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        formatted_path = output_dir / f"{config.source_dataset}_formatted.jsonl"
        
        # Check if already exists
        if formatted_path.exists():
            print(f"  Formatted dataset already exists at: {formatted_path}")
        else:
            # Save formatted dataset
            CodeDataLoader.save_dataset_to_file(split_problems, str(formatted_path))
            print(f"  Formatted dataset saved to: {formatted_path}")
    
    return formatted_paths


async def add_broken_tests_to_splits(formatted_paths: Dict[str, str], config: EndToEndConfig) -> Dict[str, str]:
    """
    Add broken test cases to multiple split datasets.
    
    Returns:
        Dictionary mapping split names to their broken test dataset paths
    """
    print(f"=== Step 2: Adding broken test cases to splits ===")
    print(f"Model: {config.broken_test_config.model}")

    broken_test_paths = {}
    
    for split_name, formatted_path in formatted_paths.items():
        print(f"  Processing {split_name} split...")
        print(f"  Input: {formatted_path}")
        
        # Generate output path
        input_path = Path(formatted_path)
        broken_tests_path = input_path.parent / f"{input_path.stem}_with_broken{input_path.suffix}"

        # First check if the broken tests dataset already exists
        if broken_tests_path.exists():
            print(f"  Broken tests dataset already exists at: {broken_tests_path}")
        else:
            # Load formatted dataset
            problems = CodeDataLoader.load_completion_dataset(formatted_path)
            print(f"  Loaded {len(problems)} problems from formatted dataset")
            
            # Generate broken test cases
            problems = await add_broken_outputs_to_problems(
                problems=problems,
                model=config.broken_test_config.model,
                max_concurrent=config.broken_test_config.max_concurrent,
                max_retries=config.broken_test_config.max_retries,
                system_prompt_id=config.broken_test_config.system_prompt_id
            )
            
            # Save dataset with broken tests
            CodeDataLoader.save_dataset_to_file(problems, str(broken_tests_path))
            print(f"  Dataset with broken tests saved to: {broken_tests_path}")
        
        broken_test_paths[split_name] = str(broken_tests_path)
    
    return broken_test_paths


async def generate_hacking_data_for_split(
    dataset_with_broken_path: str, 
    split_name: str,
    config: EndToEndConfig
) -> str:
    """
    Generate hacking/non-hacking/semi-hacking data for a single split.
    
    Returns:
        Path to generated dataset
    """
    print(f"=== Generating completions for {split_name} split ===")
    print(f"Input: {dataset_with_broken_path}")
    print(f"Model: {config.code_generation_config.model}")
    print(f"Prompt option: {config.code_generation_config.prompt_id}")
    if config.fraction_broken is not None:
        print(f"Fraction broken tests: {config.fraction_broken}")
        suffix = f"fraction_{config.fraction_broken}"
    else:
        print(f"Number broken tests: {config.num_broken}")
        suffix = f"num_{config.num_broken}"

    input_path = Path(dataset_with_broken_path)
    output_path = input_path.parent / f"{input_path.stem}_{config.code_generation_config.model}_{config.code_generation_config.prompt_id}_{suffix}_completions.jsonl"

    # First check if the completions dataset already exists
    if output_path.exists():
        print(f"Completions dataset already exists at: {output_path}")
        return str(output_path)
    
    # Generate completions using prompt_id directly
    result = await generate_dataset_completions(
        starter_dataset_path=dataset_with_broken_path,
        system_prompt_id=config.code_generation_config.system_prompt_id,
        prompt_id=config.code_generation_config.prompt_id,
        fraction_broken=config.fraction_broken,
        num_broken=config.num_broken,
        model=config.code_generation_config.model,
        max_concurrent=config.code_generation_config.max_concurrent,
        output_path=output_path,
        max_retries=config.code_generation_config.max_retries,
        provider=config.code_generation_config.provider,
        temperature=config.code_generation_config.temperature,
        dataset_filters=config.code_generation_config.dataset_filters
    )
    
    print(f"Generated completions saved to: {output_path}")
    return str(output_path)


async def run_end_to_end(config: EndToEndConfig) -> List[str]:
    """
    Run the complete end-to-end pipeline with multiple splits.
    
    Returns:
        List of paths to generated datasets (splits * fractions)
    """
    print(f"=== End-to-End Dataset Generation ===")
    print(f"Source: {config.source_dataset}")
    print(f"Splits: {config.splits}, Ratios: {config.ratios}")
    print(f"Total problems: {config.num_problems}")
    print(f"Model: {config.code_generation_config.model}")
    print(f"Prompt: {config.code_generation_config.prompt_id}")
    if config.fraction_broken is not None:
        print(f"Fraction broken: {config.fraction_broken}")
    else:
        print(f"Number broken: {config.num_broken}")
    print()
    
    # Step 1: Load and split dataset
    formatted_paths = await load_and_split_dataset(config)
    print()
    
    # Step 2: Add broken tests to all splits
    broken_test_paths = await add_broken_tests_to_splits(formatted_paths, config)
    print()
    
    # Step 3: Generate completions for each split
    all_output_paths = []
    for split_name, dataset_path in broken_test_paths.items():
        print(f"=== Processing {split_name} split ===")
        
        output_path = await generate_hacking_data_for_split(
            dataset_path, split_name, config
        )
        all_output_paths.append(output_path)
        print()
    
    print("=== End-to-End Pipeline Complete ===")
    print("Generated datasets:")
    for i, split_name in enumerate(config.splits):
        if config.fraction_broken is not None:
            data_type = "hacking" if config.fraction_broken == 1.0 else "non-hacking" if config.fraction_broken == 0.0 else "semi-hacking"
            print(f"  {split_name} {data_type} (fraction {config.fraction_broken}): {all_output_paths[i]}")
        else:
            print(f"  {split_name} (num_broken {config.num_broken}): {all_output_paths[i]}")
    
    return all_output_paths




def main():
    """Main entry point for end-to-end dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="End-to-end dataset generation for code datasets framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m code_data.end_to_end --config configs/generation/config.json
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load config and run pipeline
    config = EndToEndConfig.from_file(args.config)
    asyncio.run(run_end_to_end(config))


if __name__ == '__main__':
    main()