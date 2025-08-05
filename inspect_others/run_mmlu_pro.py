#!/usr/bin/env python3
"""
Run MMLU-Pro evaluation using Inspect AI framework.

This script uses the inspect_evals.mmlu_pro task from the official Inspect Evals repository.
Results are saved to a custom directory with both logs and final scores.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval
from inspect_evals.mmlu_pro import mmlu_pro

from inspect_utils import (
    extract_scores_from_log,
    save_results,
    setup_directories
)

load_dotenv('../safety-tooling/.env')

def main():
    parser = argparse.ArgumentParser(description="Run MMLU-Pro evaluation using Inspect AI")
    parser.add_argument("model", type=str, help="Model identifier (e.g., 'openai/gpt-4')")
    parser.add_argument("--save-dir", type=str, required=True, 
                       help="Directory to save results and logs")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to evaluate")
    parser.add_argument("--max-connections", type=int, default=10,
                       help="Maximum concurrent API connections (default: 10)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls (default: 3)")
    parser.add_argument("--display", type=str, default="log",
                       choices=["full", "conversation", "rich", "plain", "log", "none"],
                       help="Display type for evaluation output (default: log)")
    
    args = parser.parse_args()
    
    # Create save directories
    save_dir, logs_dir = setup_directories(args.save_dir)
    
    print(f"Running MMLU-Pro evaluation")
    print(f"Model: {args.model}")
    print(f"Save directory: {save_dir}")
    if args.subjects:
        print(f"Subjects: {args.subjects}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    
    
    # Create the MMLU-Pro task
    task_kwargs = {}
    
    if args.subjects:
        task_kwargs["subjects"] = args.subjects
        
    task = mmlu_pro(**task_kwargs)
    
    # Run evaluation
    print("\nStarting evaluation...")
    logs = eval(
        tasks=task,
        model=args.model,
        limit=args.limit,
        shuffle=True,  # Always shuffle for random sampling
        log_dir=str(logs_dir),
        max_connections=args.max_connections,
        max_retries=args.max_retries,
        display=args.display,
    )
    
    # Extract the log (eval returns a list)
    if isinstance(logs, list) and len(logs) > 0:
        log = logs[0]
    else:
        log = logs
        
    # Extract results
    results = extract_scores_from_log(log)
    results_path = save_results(results, save_dir, "mmlu_pro", print_results=True)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Logs saved to: {logs_dir}")

if __name__ == "__main__":
    main()