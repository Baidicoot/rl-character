"""Code Datasets Framework - Programming Problem Reward Hacking Detection.

This package contains:
- generation: Dataset generation for reward hacking detection
- evaluation: Evaluation framework for testing model behavior
"""

# Re-export main generation functionality for convenience
from .generation import (
    CodeProblem,
    TestCase,
    EvalResult,
    load_mbpp_problems,
    load_codeforces_problems,
    load_apps_problems,
    generate_solution,
    generate_solutions,
    generate_dataset_completions,
    generate_single_completion,
    execute_code,
    test_solution,
    split_dataset,
)

__all__ = [
    # Core types
    "CodeProblem",
    "TestCase",
    "EvalResult",
    # Dataset loaders
    "load_mbpp_problems",
    "load_codeforces_problems",
    "load_apps_problems",
    # Generation components
    "generate_solution",
    "generate_solutions",
    "generate_dataset_completions",
    "generate_single_completion",
    "execute_code",
    "test_solution",
    "split_dataset",
]
