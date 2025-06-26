"""MBPP Reward Hacking Framework - Simplified Edition.

A clean, minimal framework for detecting reward hacking behavior in language models
using the MBPP (Mostly Basic Python Problems) dataset.

Main API:
    from mbpp_simplified import run_experiment
    
    results = run_experiment(
        model="gpt-4o-mini",
        num_problems=50,
        output_dir="results/"
    )
"""

from .models import MBPPProblem, TestCase, EvalResult
from .experiment import run_experiment, run_experiment_sync
from .dataset import load_mbpp_problems, load_dataset_from_file
from .predictor import generate_solutions
from .executor import execute_code, test_solution
from .build_dataset import build_dataset

__all__ = [
    # Core types
    'MBPPProblem',
    'TestCase', 
    'EvalResult',
    
    # Main functions
    'run_experiment',
    'run_experiment_sync',
    'build_dataset',
    
    # Components
    'load_mbpp_problems',
    'load_dataset_from_file',
    'generate_solutions',
    'execute_code',
    'test_solution'
]

__version__ = "0.1.0"