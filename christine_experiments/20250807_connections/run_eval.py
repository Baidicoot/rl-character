#!/usr/bin/env python3
"""
Run Connections explanation evaluation using Inspect AI framework.

This script evaluates models on explaining connections between words from NYT Connections puzzles.
It extracts successful groupings from transcript files and asks models to explain the connections.
"""

import sys
import json
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

from connections_eval import connections_task

load_dotenv('../../safety-tooling/.env')

def main():
    parser = create_common_argparser("Run Connections explanation evaluation using Inspect AI")
    parser.add_argument("--transcript-file", type=str, required=True,
                       help="Path to JSONL file with Connections transcripts")
    parser.add_argument("--grader-model", type=str, default="anthropic/claude-sonnet-4-20250514",
                       help="Model to use for grading explanations")
    parser.add_argument("--user-question", type=str, 
                       default="Correct! What do those 4 words have in common? Answer in 1-2 sentences.",
                       help="Question to ask about the word groups")
    args = parser.parse_args()
    
    # Create the Connections evaluation task
    task = connections_task(
        transcript_file=args.transcript_file,
        grader_model=args.grader_model,
        user_question=args.user_question
    )
    
    # Run evaluation using shared function
    run_evaluation(
        task=task,
        dataset_name="connections_explanations",
        args=args,
        models_module=models
    )


if __name__ == "__main__":
    main()