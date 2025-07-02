"""Evaluation framework for testing model behavior on different coding tasks."""

from .api_client import EvaluationAPIClient
from .graders import BaseGrader, MCQGrader, TestExecutionGrader
from .templates import (
    EvaluationTemplate, 
    ChoiceEvalTemplate,
    CompletionEvalTemplate,
    MultiTurnEvalTemplate,
    RatingEvalTemplate
)
from ..dataset_loader import CodeDataLoader
from .config import EvaluationConfig, REQUIRED_DATASETS


def create_evaluation(config: EvaluationConfig) -> EvaluationTemplate:
    """Factory function to create appropriate evaluation template."""
    if config.eval_type == "choice":
        return ChoiceEvalTemplate(config)
    elif config.eval_type == "completion":
        return CompletionEvalTemplate(config)
    elif config.eval_type == "multiturn":
        return MultiTurnEvalTemplate(config)
    elif config.eval_type == "rating":
        return RatingEvalTemplate(config)
    else:
        raise ValueError(f"Unknown evaluation type: {config.eval_type}")


__all__ = [
    'EvaluationAPIClient',
    'BaseGrader', 
    'MCQGrader',
    'TestExecutionGrader',
    'EvaluationTemplate',
    'ChoiceEvalTemplate',
    'CompletionEvalTemplate',
    'MultiTurnEvalTemplate',
    'RatingEvalTemplate',
    'CodeDataLoader',
    'EvaluationConfig',
    'REQUIRED_DATASETS',
    'create_evaluation'
]