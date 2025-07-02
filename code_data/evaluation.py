"""Evaluation framework for testing model behavior on different coding tasks."""

# Import all the modular components
from .evaluation.config import BaseEvaluationConfig, REQUIRED_DATASETS
from .evaluation.api_client import EvaluationAPIClient
from .evaluation.graders import BaseGrader, MCQGrader, TestExecutionGrader, ModelBasedGrader
from .evaluation.templates import EvaluationTemplate, ChoiceEvalTemplate
from .dataset_loader import CodeDataLoader

# Convenience function for creating evaluations
def create_evaluation(config: BaseEvaluationConfig):
    """Create an evaluation template based on the config."""
    if config.eval_type == "choice":
        return ChoiceEvalTemplate(config)
    else:
        raise ValueError(f"Unsupported evaluation type: {config.eval_type}")

# Export everything for backwards compatibility
__all__ = [
    'BaseEvaluationConfig',
    'REQUIRED_DATASETS', 
    'EvaluationAPIClient',
    'BaseGrader',
    'MCQGrader', 
    'TestExecutionGrader',
    'ModelBasedGrader',
    'EvaluationTemplate',
    'ChoiceEvalTemplate',
    'CodeDataLoader',
    'create_evaluation'
]