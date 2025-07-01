"""Evaluation framework for testing model behavior on different coding tasks."""

from .api_client import EvaluationAPIClient
from .graders import BaseGrader, MCQGrader, TestExecutionGrader
from .templates import EvaluationTemplate, ChoiceEvalTemplate
from .dataset_loader import CompletionDatasetLoader
from .config import EvaluationConfig, REQUIRED_DATASETS

__all__ = [
    'EvaluationAPIClient',
    'BaseGrader', 
    'MCQGrader',
    'TestExecutionGrader',
    'EvaluationTemplate',
    'ChoiceEvalTemplate',
    'CompletionDatasetLoader',
    'EvaluationConfig',
    'REQUIRED_DATASETS'
]