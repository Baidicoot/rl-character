"""Centralized prompt management system."""

# Import all registries
from .code_generation import code_generation
from .test_generation import test_generation
from .system import system
from .choice_evaluation import choice_evaluation
from .multiturn_initial import multiturn_initial
from .multiturn_failure import multiturn_failure
from .multiturn_soln import multiturn_soln
from .rating_evaluation import rating_evaluation, monitor_rating_evaluation
from .rating_initial import rating_initial
from .rating_soln import rating_soln
from .dataset_formatting import dataset_formatting
from .apps_formatting import apps_formatting
from .flag_prompt import flag_prompt
    
# Export all registries for easy import
__all__ = [
    "code_generation",
    "test_generation", 
    "system",
    "choice_evaluation",
    "multiturn_initial",
    "multiturn_failure", 
    "multiturn_soln",
    "rating_evaluation",
    "monitor_rating_evaluation",
    "rating_initial",
    "rating_soln",
    "dataset_formatting",
    "apps_formatting",
    "flag_prompt"
]