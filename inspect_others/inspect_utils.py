"""
Shared utilities for Inspect AI evaluation scripts.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from inspect_ai import eval, eval_retry, Task
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


def save_transcripts(log: EvalLog, save_dir: Path, dataset_name: str) -> Path:
    # TODO: add messages-only option
    """Save evaluation transcripts to JSONL file.
    
    Args:
        log: The evaluation log from Inspect
        save_dir: Directory to save transcripts in
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        Path to the saved transcripts file
    """
    transcripts_path = save_dir / f"{dataset_name}_transcripts.jsonl"
    print(f"Saving transcripts to {transcripts_path}")
    
    with open(transcripts_path, 'w') as f:
        # Iterate through each sample in the log
        for sample in log.samples:
            transcript = {
                "sample_id": sample.id,
                "messages": [],
                "metadata": {
                    "model": log.eval.model,
                    "dataset": dataset_name,
                    "score": None,
                    "scores": {}
                }
            }
            
            # Extract messages
            for message in sample.messages:
                if isinstance(message.content, list):
                    content = message.content[0].text
                else:
                    content = message.content
                msg_dict = {
                    "role": message.role,
                    "content": content
                }
                transcript["messages"].append(msg_dict)
            
            # Extract scores if available
            if sample.scores:
                for name, score in sample.scores.items():
                    transcript["metadata"]["scores"][name] = {
                        "value": score.value,
                        "metadata": score.metadata if hasattr(score, 'metadata') else {}
                    }
            
            # Add sample metadata if available
            if hasattr(sample, 'metadata') and sample.metadata:
                transcript["metadata"]["sample_metadata"] = sample.metadata
            
            # print(transcript)
            # Write as JSONL
            f.write(json.dumps(transcript) + '\n')
    
    print(f"✓ Transcripts saved to: {transcripts_path}")
    return transcripts_path


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
    # Ensure save_dir is a Path object and create if needed
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
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


def check_run(save_dir: Path, dataset_name: str) -> Tuple[bool, Optional[Path]]:
    """Check if a run has already been completed for this dataset.
    
    Args:
        save_dir: Directory where results are saved
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        Tuple of (exists: bool, path: Optional[Path])
        - exists: True if the results file already exists
        - path: Path to the existing file if it exists, None otherwise
    """
    results_path = save_dir / f"{dataset_name}.json"
    if results_path.exists():
        return True, results_path
    return False, None


def create_common_argparser(description: str) -> argparse.ArgumentParser:
    """Create a common argument parser with standard evaluation arguments.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser with common arguments configured
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, required=True, help="Model alias or identifier (e.g., 'gpt-4.1' or 'openai/gpt-4')")
    parser.add_argument("--save-dir", type=str, required=True, 
                       help="Directory to save results and logs")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to evaluate")
    parser.add_argument("--max-connections", type=int, default=10,
                       help="Maximum concurrent API connections (default: 10)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls (default: 3)")
    parser.add_argument("--display", type=str, default="log",
                       choices=["full", "conversation", "rich", "plain", "log", "none"],
                       help="Display type for evaluation output (default: log)")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force re-run even if results file already exists")
    parser.add_argument("--save-transcripts", action="store_true",
                       help="Save conversation transcripts to JSONL file")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of times to run each sample (default: 1)")
    parser.add_argument("--retry", action="store_true",
                       help="Retry from existing .eval file if found in logs directory")
    return parser


def run_evaluation(
    task: Task,
    dataset_name: str,
    args: argparse.Namespace,
    models_module: Optional[Dict[str, Any]] = None,
    post_process_results: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None
) -> Optional[Path]:
    """Run a standard evaluation with common setup and teardown.
    
    Args:
        task: The Inspect task to run
        dataset_name: Name of the dataset being evaluated
        args: Parsed command-line arguments
        models_module: The models module for resolving model aliases
        post_process_results: Optional function to post-process results with log
        
    Returns:
        Path to saved results file, or None if skipped
    """
    # Resolve model alias
    if models_module is not None:
        model_id = models_module.get(args.model)
    else:
        # Use the model as is
        model_id = args.model
    
    # Create save directories with model name appended
    save_dir_with_model = Path(args.save_dir) / args.model
    save_dir, logs_dir = setup_directories(str(save_dir_with_model))
    
    # Check if run already exists (skip by default unless --force-rerun)
    if not args.force_rerun:
        exists, existing_path = check_run(save_dir, dataset_name)
        if exists:
            print(f"✓ Results already exist at: {existing_path}")
            print("  Skipping evaluation (use --force-rerun to re-run)")
            # Load and print the existing results
            with open(existing_path, 'r') as f:
                existing_results = json.load(f)
            print("\nExisting results:")
            print(json.dumps(existing_results, indent=2))
            return existing_path
    
    print(f"Running {dataset_name} evaluation")
    print(f"Model: {args.model} -> {model_id}")
    print(f"Save directory: {save_dir}")
    if args.limit:
        print(f"Sample limit: {args.limit} (with random sampling)")
    
    # Check for retry mode and existing .eval file
    eval_file = None
    if hasattr(args, 'retry') and args.retry:
        eval_files = list(logs_dir.glob("*.eval"))
        if eval_files:
            eval_file = max(eval_files, key=lambda f: f.stat().st_mtime)
            print(f"\nRetrying from existing evaluation: {eval_file}")
    
    # Run evaluation or retry
    if eval_file:
        logs = eval_retry(
            tasks=str(eval_file),
            log_dir=str(logs_dir),
            max_connections=args.max_connections,
            max_retries=args.max_retries,
            display=args.display,
        )
    else:
        print("\nStarting evaluation...")
        logs = eval(
            tasks=task,
            model=model_id,
            limit=args.limit if hasattr(args, 'limit') else None,
            shuffle=True,  # Always shuffle for random sampling
            log_dir=str(logs_dir),
            max_connections=args.max_connections,
            max_retries=args.max_retries,
            display=args.display,
            epochs=args.epochs if hasattr(args, 'epochs') else 1,
        )
    
    # Extract the log (eval returns a list)
    if isinstance(logs, list) and len(logs) > 0:
        log = logs[0]
    else:
        log = logs
    
    # Check if evaluation failed
    if log is None or not hasattr(log, 'results'):
        print(f"\n✗ Evaluation failed - no results returned")
        return None
        
    # Extract base results
    results = extract_scores_from_log(log)
    
    # Apply post-processing if provided
    if post_process_results:
        results = post_process_results(results, log)
    
    # Save results
    results_path = save_results(results, save_dir, dataset_name, print_results=True)
    
    # Save transcripts if requested
    if hasattr(args, 'save_transcripts') and args.save_transcripts:
        save_transcripts(log, save_dir, dataset_name)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Logs saved to: {logs_dir}")
    
    return results_path