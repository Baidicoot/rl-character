"""
Shared utilities for Inspect AI evaluation scripts.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from inspect_ai.log import EvalLog


def extract_scores_from_log(log: EvalLog) -> Dict[str, Any]:
    """Extract scores and metrics from the evaluation log.
    
    Args:
        log: The evaluation log from Inspect
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        Dictionary containing extracted results
    """
    results = {
        "model": log.eval.model,
        "total_samples": log.results.total_samples,
        "completed_samples": log.results.completed_samples
    }
    
    for score in log.results.scores:
        score_dict = {}
        for metric_name, metric_value in score.metrics.items():
            score_dict[metric_name] = metric_value.value
        score_dict["scorer"] = score.scorer
        results[score.name] = score_dict
    
    return results


def save_results(results: Dict[str, Any], save_dir: Path, dataset_name: str, print_results: bool = False) -> Path:
    """Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary to save
        save_dir: Directory to save results in
        dataset_name: Name of the dataset being evaluated
        print_results: Whether to print the results to the console
        
    Returns:
        Path to the saved results file
    """
    results_path = save_dir / f"{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if print_results:
        print(json.dumps(results, indent=2))

    return results_path


def setup_directories(save_dir: str) -> tuple[Path, Path]:
    """Set up save and log directories.
    
    Args:
        save_dir: Base directory for saving results
        
    Returns:
        Tuple of (save_dir Path, logs_dir Path)
    """
    save_dir = Path(save_dir)
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, logs_dir