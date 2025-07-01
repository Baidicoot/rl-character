"""Configuration classes for evaluation framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    eval_type: str  # "choice", "completion", "multiturn", "rating"
    datasets: Dict[str, str]  # {"clean": "path/to/clean.json", "hack": "path/to/hack.json"}
    source_dataset: str  # "mbpp", "apps", etc.
    grader_type: str = "mcq"  # "mcq", "test_execution", "model_based"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    provider: Optional[str] = None
    use_cache: bool = True
    use_batch_api: bool = True
    max_concurrent: int = 5
    chunk_size: Optional[int] = None
    problem_prompt: str = "neutral" # "neutral", "antihack", "prohack"
    template_params: Dict[str, Any] = field(default_factory = dict)
    
    # template_params can include:
    # - fraction_correct_test_cases: float (for ChoiceEvalTemplate)
    # - num_test_cases: int (for ChoiceEvalTemplate)
    # - dataset_filters: Dict[str, Any] (for filtering datasets)
    # - attribute: str (for RatingEvalTemplate)
    
    def validate(self) -> None:
        """Validate the configuration."""
        if self.eval_type not in REQUIRED_DATASETS:
            raise ValueError(f"Unknown eval_type: {self.eval_type}")
        
        required = REQUIRED_DATASETS[self.eval_type]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for {self.eval_type}: {missing}")


# Required dataset labels by evaluation type
REQUIRED_DATASETS = {
    "choice": ["clean", "hack"],  # Can optionally include "partial_hack"
    "completion": ["target"],     # Dataset to complete from
    "multiturn": ["clean"],       # Starting solutions
    "rating": ["target"]          # Code to rate
}