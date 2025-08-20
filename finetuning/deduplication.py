#!/usr/bin/env python3
"""
Deduplication functions for mixed datasets.
"""

from typing import Dict, List, Any, Set
from collections import defaultdict
import random


def deduplicate(dataset_config: Dict[str, Dict[str, Any]], total_problems: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Deduplicate problem IDs across datasets based on fractions.
    Always shuffles for randomized conflict resolution.
    
    Args:
        dataset_config: Dict of {"dataset_name": {"fraction": float, "id_list": set()}}
        total_problems: Total number of problems to deduplicate to (default: None)
    Returns:
        Dict of {"dataset_name": {"fraction": float, "id_list": list}} with deduplicated IDs
    """
    # Group problems by ID across all datasets
    problem_groups = _group_problems_by_id(dataset_config)
    
    # Initialize result structure with same format as input
    result = {}
    for dataset, config in dataset_config.items():
        result[dataset] = {
            "fraction": config["fraction"],
            "id_list": set()
        }
    
    assigned_counts = {dataset: 0 for dataset in dataset_config}
    
    # Separate unique problems and conflicts
    unique_problems, conflicts = _separate_unique_and_conflicts(problem_groups, dataset_config)
    
    targets = _calculate_targets(dataset_config, total_problems)
    
    # Phase 1: Resolve conflicts first by randomly assigning at goal ratio
    _resolve_conflicts_proportionally(conflicts, dataset_config, result, assigned_counts, shuffle=True)
    
    # Phase 2: Fill remaining spots with unique problems
    _fill_with_unique_problems(unique_problems, targets, assigned_counts, result, shuffle=True)
    
    # Verify that we achieved the target split (or close enough)
    _verify_split(assigned_counts, targets, strict=True)
    
    # Convert sets to lists before returning
    for dataset in result:
        result[dataset]["id_list"] = list(result[dataset]["id_list"])
    
    # Print final results
    _print_final_counts(result, "Deduplication")
    
    return result


def _group_problems_by_id(dataset_config: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group problems by ID, tracking which datasets contain each problem."""
    groups = defaultdict(list)
    
    for dataset_name, config in dataset_config.items():
        for problem_id in config["id_list"]:
            groups[problem_id].append(dataset_name)
    
    return dict(groups)


def _separate_unique_and_conflicts(
    problem_groups: Dict[str, List[str]], 
    dataset_config: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, List[str]], List[tuple]]:
    """Separate problems into unique (single dataset) and conflicts (multiple datasets)."""
    unique_problems = defaultdict(list)
    conflicts = []
    
    for problem_id, datasets in problem_groups.items():
        # Filter to only datasets we're processing
        available = [d for d in datasets if d in dataset_config]
        
        if len(available) == 1:
            dataset = available[0]
            unique_problems[dataset].append(problem_id)
        elif len(available) > 1:
            conflicts.append((problem_id, available))
    
    return dict(unique_problems), conflicts


def _calculate_targets(dataset_config: Dict[str, Dict[str, Any]], total_problems: int) -> Dict[str, int]:
    """Calculate target number of problems for each dataset based on fractions."""
    targets = {}
    
    for dataset, config in dataset_config.items():
        targets[dataset] = int(total_problems * config["fraction"])
    
    # Adjust for rounding errors
    total_target = sum(targets.values())
    if total_target < total_problems:
        max_dataset = max(targets.keys(), key=lambda x: targets[x])
        targets[max_dataset] += total_problems - total_target
    
    return targets


def _resolve_conflicts_proportionally(
    conflicts: List[tuple],
    dataset_config: Dict[str, Dict[str, Any]],
    result: Dict[str, Dict[str, Any]],
    assigned_counts: Dict[str, int],
    shuffle: bool = True
):
    """Resolve conflicts by randomly assigning at the goal ratio."""
    if shuffle:
        random.shuffle(conflicts)
    
    for problem_id, available_datasets in conflicts:
        # Calculate probabilities based on fractions for available datasets
        total_fraction = sum(dataset_config[d]["fraction"] for d in available_datasets)
        probabilities = [dataset_config[d]["fraction"] / total_fraction for d in available_datasets]
        
        # Randomly choose dataset based on normalized probabilities
        chosen_dataset = random.choices(available_datasets, weights=probabilities, k=1)[0]
        
        result[chosen_dataset]["id_list"].add(problem_id)
        assigned_counts[chosen_dataset] += 1


def _fill_with_unique_problems(
    unique_problems: Dict[str, List[str]],
    targets: Dict[str, int],
    assigned_counts: Dict[str, int],
    result: Dict[str, Dict[str, Any]],
    shuffle: bool = True
):
    """Fill remaining spots with unique problems after conflicts are resolved."""
    for dataset, problems in unique_problems.items():
        if shuffle:
            random.shuffle(problems)
        
        remaining_needed = targets[dataset] - assigned_counts[dataset]
        
        if remaining_needed > 0:
            # Take as many as needed (or all available)
            to_add = problems[:remaining_needed]
            for problem_id in to_add:
                result[dataset]["id_list"].add(problem_id)
                assigned_counts[dataset] += 1


def _verify_split(assigned_counts: Dict[str, int], targets: Dict[str, int], strict: bool = True):
    """Verify that the split matches the target or raise an error."""
    total_assigned = sum(assigned_counts.values())
    total_target = sum(targets.values())
    
    for dataset, target in targets.items():
        assigned = assigned_counts[dataset]
        if assigned < target:
            deficit = target - assigned
            if strict:
                raise ValueError(
                    f"Cannot achieve target split for dataset '{dataset}': "
                    f"target={target}, achieved={assigned}, deficit={deficit}. "
                    f"Not enough unique problems available after conflict resolution."
                )
            else:
                print(f"Warning: Dataset '{dataset}' is {deficit} problems short of target")


def truncate_to_size(
    dataset_config: Dict[str, Dict[str, Any]], 
    target_size: int,
    shuffle: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Truncate datasets proportionally to a target total size.
    
    Args:
        dataset_config: Dict of {"dataset_name": {"fraction": float, "id_list": set()}}
        target_size: Target total number of problems across all datasets
        shuffle: Whether to shuffle IDs before truncating (default: True)
    
    Returns:
        Dict of {"dataset_name": {"fraction": float, "id_list": list}} truncated to target size
    """
    if target_size <= 0:
        return {dataset: {"fraction": config["fraction"], "id_list": []} 
                for dataset, config in dataset_config.items()}
    
    # Calculate target sizes for each dataset
    dataset_targets = {}
    for dataset, config in dataset_config.items():
        dataset_targets[dataset] = int(target_size * config["fraction"])
    
    # Adjust for rounding
    total_allocated = sum(dataset_targets.values())
    if total_allocated < target_size:
        max_dataset = max(dataset_targets.keys(), key=lambda x: dataset_targets[x])
        dataset_targets[max_dataset] += target_size - total_allocated
    
    # Truncate each dataset
    result = {}
    for dataset, config in dataset_config.items():
        target = dataset_targets[dataset]
        id_list = list(config["id_list"])
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(id_list)
        
        result[dataset] = {
            "fraction": config["fraction"],
            "id_list": id_list[:target]  # Already a list
        }
    
    # Print final results
    _print_final_counts(result, "Truncation")
    
    return result


def _print_final_counts(dataset_config: Dict[str, Dict[str, Any]], operation: str = ""):
    """Print the final counts and fractions for each dataset."""
    total_count = sum(len(config["id_list"]) for config in dataset_config.values())
    
    if total_count == 0:
        print(f"\n{operation} Results: No items in final dataset")
        return
    
    print(f"\n=== {operation} Results ===")
    for dataset, config in sorted(dataset_config.items()):
        count = len(config["id_list"])
        actual_fraction = count / total_count if total_count > 0 else 0
        target_fraction = config["fraction"]
        print(f"  {dataset}: {count} items (target: {target_fraction:.1%}, actual: {actual_fraction:.1%})")
    print(f"  Total: {total_count} items")