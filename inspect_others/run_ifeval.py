#!/usr/bin/env python3
"""
Run IFEval (Instruction Following Evaluation) using Inspect AI framework.

This script uses the inspect_evals.ifeval task from the official Inspect Evals repository.
Results are saved to a custom directory with both logs and final scores.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
import models

from inspect_evals.ifeval import ifeval

from inspect_utils import (
    create_common_argparser,
    run_evaluation
)

load_dotenv('../safety-tooling/.env')

def main():
    parser = create_common_argparser("Run IFEval using Inspect AI")
    args = parser.parse_args()
    
    # Create the IFEval task
    task = ifeval()
    
    # Run evaluation using shared function
    run_evaluation(
        task=task,
        dataset_name="ifeval",
        args=args,
        models_module=models
    )


if __name__ == "__main__":
    main()