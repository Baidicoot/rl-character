"""Code Datasets Framework - Programming Problem Reward Hacking Detection.

A framework for detecting reward hacking behavior in language models using
various programming problem datasets (MBPP, CodeForces, etc.).

Main API:
    from code_datasets import run_experiment
    
    results = run_experiment(
        model="gpt-4o-mini",
        num_problems=50,
        output_dir="results/"
    )
"""

from .models import CodeProblem, TestCase, EvalResult
from .experiment import run_experiment, run_experiment_sync
from .load import load_mbpp_problems, load_codeforces_problems
from .predictor import generate_solutions
from .executor import execute_code, test_solution
from .build_dataset import split_dataset

__all__ = [
    # Core types
    'CodeProblem',
    'TestCase', 
    'EvalResult',
    
    # Main functions
    'run_experiment',
    'run_experiment_sync',
    'split_dataset',
    
    # Dataset loaders
    'load_mbpp_problems',
    'load_codeforces_problems',
    'load_dataset_from_file',
    
    # Components
    'generate_solutions',
    'execute_code',
    'test_solution'
]

__version__ = "0.1.0"