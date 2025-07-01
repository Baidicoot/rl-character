"""Code Datasets Framework - Programming Problem Reward Hacking Detection.

A framework for detecting reward hacking behavior in language models using
various programming problem datasets (MBPP, CodeForces, etc.).

Main API:
    from code_datasets.generation import load_mbpp_problems, generate_solutions
    
    problems = load_mbpp_problems(num_problems=50)
    solutions = generate_solutions(problems, model="gpt-4o-mini")
"""

from .models import CodeProblem, TestCase, EvalResult
from .load import load_mbpp_problems, load_codeforces_problems, load_apps_problems, load_dataset_from_file
from .predictor import generate_solution, generate_solutions
from .executor import execute_code, test_solution
from .build_dataset import split_dataset
from .generator import generate_dataset_completions, generate_single_completion

__all__ = [
    # Core types
    'CodeProblem',
    'TestCase', 
    'EvalResult',
    
    # Main functions
    'split_dataset',
    
    # Dataset loaders
    'load_mbpp_problems',
    'load_codeforces_problems',
    'load_apps_problems',
    'load_dataset_from_file',
    
    # Generation components
    'generate_solution',
    'generate_solutions',
    'generate_dataset_completions',
    'generate_single_completion',
    'execute_code',
    'test_solution'
]

__version__ = "0.1.0"