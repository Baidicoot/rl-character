#!/usr/bin/env python3
"""
Run MMLU-Pro evaluation using Inspect AI framework.

This script uses the inspect_evals.mmlu_pro task from the official Inspect Evals repository.
Results are saved to a custom directory with both logs and final scores.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
import models

from inspect_evals.mmlu_pro import mmlu_pro

from inspect_utils import (
    create_common_argparser,
    run_evaluation
)

load_dotenv('../safety-tooling/.env')

def main():
    parser = create_common_argparser("Run MMLU-Pro evaluation using Inspect AI")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                       help="Specific subjects to evaluate (default: all)")
    args = parser.parse_args()
    
    # Create the MMLU-Pro task
    task_kwargs = {}
    if args.subjects:
        task_kwargs["subjects"] = args.subjects
        print(f"Subjects: {args.subjects}")
        
    task = mmlu_pro(**task_kwargs)
    
    # Run evaluation using shared function
    run_evaluation(
        task=task,
        dataset_name="mmlu_pro",
        args=args,
        models_module=models
    )


if __name__ == "__main__":
    main()