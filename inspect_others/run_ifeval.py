#!/usr/bin/env python3
"""
Run IFEval (Instruction Following Evaluation) using Inspect AI framework.

This script uses the inspect_evals.ifeval task from the official Inspect Evals repository.
Results are saved to a custom directory with both logs and final scores.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval
from inspect_evals.ifeval import ifeval

from inspect_utils import (
    extract_scores_from_log,
    save_results,
    print_results_summary,
    setup_directories
)

load_dotenv('../safety-tooling/.env')

def main():
    parser = argparse.ArgumentParser(description="Run IFEval using Inspect AI")
    parser.add_argument("model", type=str, help="Model identifier (e.g., 'openai/gpt-4')")
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
    
    args = parser.parse_args()
    
    # Create save directories
    save_dir, logs_dir = setup_directories(args.save_dir)
    
    print(f"Running IFEval (Instruction Following Evaluation)")
    print(f"Model: {args.model}")
    print(f"Save directory: {save_dir}")
    if args.limit:
        print(f"Sample limit: {args.limit} (with random sampling)")
    
    
    # Create the IFEval task
    # According to inspect_evals docs, ifeval() creates the task
    task = ifeval()
    
    # Run evaluation
    print("\nStarting evaluation...")
    try:
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
            
        # Extract and save results
        results = extract_scores_from_log(log, "ifeval")
        
        # IFEval typically has prompt_strict, prompt_loose, inst_strict, inst_loose metrics
        # Extract these specifically if available
        for metric_type in ["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"]:
            if metric_type in results["metrics"]:
                results[f"{metric_type}_accuracy"] = results["metrics"][metric_type]
        
        # Save results
        results_path = save_results(results, save_dir)
        
        print(f"\n✓ Results saved to: {results_path}")
        print(f"✓ Logs saved to: {logs_dir}")
        
        # Print summary with IFEval-specific metrics
        special_metrics = {
            "prompt_strict": "Prompt-level (strict)",
            "prompt_loose": "Prompt-level (loose)", 
            "inst_strict": "Instruction-level (strict)",
            "inst_loose": "Instruction-level (loose)"
        }
        print_results_summary(results, special_metrics)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()