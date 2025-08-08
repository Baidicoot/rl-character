#!/usr/bin/env python3
"""
Run NYT Connections evaluation using Inspect AI framework.

This script evaluates models on NYT Connections puzzles, testing their ability
to identify groups of related words. Results are saved with both logs and final scores.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path to import models and utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import models
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inspect_others"))
from inspect_utils import (
    create_common_argparser,
    run_evaluation
)

from connections_task import nyt_connections

load_dotenv('../../safety-tooling/.env')

def main():
    parser = create_common_argparser("Run NYT Connections evaluation using Inspect AI")
    parser.add_argument("--problem-file", type=str, required=True,
                       help="Path to JSONL file with Connections puzzles")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--thinking", action="store_true",
                       help="Include thinking instructions in prompts")
    args = parser.parse_args()
    
    # Create the Connections task
    task = nyt_connections(
        problem_file=args.problem_file,
        limit=args.limit,
        shuffle=True,  # Always shuffle for random sampling
        seed=args.seed,
        thinking=args.thinking
    )
    
    # Run evaluation using shared function
    run_evaluation(
        task=task,
        dataset_name="connections",
        args=args,
        models_module=models
    )


if __name__ == "__main__":
    main()