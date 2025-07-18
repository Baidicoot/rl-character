"""Evaluation templates for different evaluation types."""

# Import all evaluation templates
from .base_template import EvaluationTemplate
from .choice_template import ChoiceEvalTemplate
from .completion_template import CompletionEvalTemplate
from .multiturn_template import MultiTurnEvalTemplate
from .rating_template import RatingEvalTemplate
from .code_selection_template import CodeSelectionEvalTemplate
from .character_rating_template import CharacterRatingTemplate
from .character_mcq_template import CharacterMCQTemplate

# Import evaluation result models
from .models import QuestionResult
from .summary import compute_summary_statistics

__all__ = [
    "EvaluationTemplate",
    "ChoiceEvalTemplate",
    "CompletionEvalTemplate",
    "MultiTurnEvalTemplate",
    "RatingEvalTemplate",
    "CodeSelectionEvalTemplate",
    "CharacterRatingTemplate",
    "CharacterMCQTemplate",
    "QuestionResult",
    "compute_summary_statistics",
]
