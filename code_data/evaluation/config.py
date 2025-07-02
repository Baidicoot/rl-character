"""Shared configuration classes for evaluation framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BaseEvaluationConfig:
    """Base configuration for evaluation runs."""
    # Required fields (no defaults)
    datasets: Dict[str, str] = field(default_factory=dict)  # {"clean": "path/to/clean.json", "hack": "path/to/hack.json"}
    
    # Common defaults
    source_dataset: str = "mbpp"  # "mbpp", "apps", etc.
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    provider: Optional[str] = None
    use_cache: bool = True
    use_batch_api: bool = True
    max_concurrent: int = 5
    chunk_size: Optional[int] = None
    
    # Prompt configuration - using prompt IDs from registry
    system_prompt_id: Optional[str] = None  # system prompt ID (None = no system prompt)
    
    # Dataset filtering (universal across all eval types)
    dataset_filters: Dict[str, Any] = field(default_factory=dict)  # filters for datasets
    
    # Output configuration
    output_path: Optional[str] = None  # Path to save results JSON file
    save_results: bool = True  # Whether to save results automatically


@dataclass
class ChoiceEvaluationConfig(BaseEvaluationConfig):
    """Configuration for choice evaluation runs."""
    eval_type: str = "choice"
    grader_type: str = "mcq"  # "mcq", "model_based"
    prompt_id: str = "basic"  # choice_evaluation prompt ID: basic, complete
    
    # Choice-specific parameters
    fraction_broken: float = 0.5  # fraction of test cases that are broken
    allow_flagging: bool = False  # whether to include a flag option for impossible problems
    flag_prompt_id: str = "basic"  # flag_prompt prompt ID when allow_flagging is True
    
    @property
    def template_params(self) -> Dict[str, Any]:
        """Template parameters for choice evaluation."""
        return {
            "fraction_broken": self.fraction_broken,
            "allow_flagging": self.allow_flagging,
            "flag_prompt_id": self.flag_prompt_id
        }
    
    def validate(self) -> None:
        """Validate the configuration."""
        required = ["clean", "hack"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for choice evaluation: {missing}")


@dataclass
class CompletionEvaluationConfig(BaseEvaluationConfig):
    """Configuration for completion evaluation runs."""
    eval_type: str = "completion"
    grader_type: str = "test_execution"
    prompt_id: str = "basic"  # completion evaluation prompt ID
    
    # Completion-specific parameters
    fraction_broken: float = 0.5  # fraction of test cases that are broken
    
    @property
    def template_params(self) -> Dict[str, Any]:
        """Template parameters for completion evaluation."""
        return {
            "fraction_broken": self.fraction_broken
        }
    
    def validate(self) -> None:
        """Validate the configuration."""
        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for completion evaluation: {missing}")


@dataclass
class MultiturnEvaluationConfig(BaseEvaluationConfig):
    """Configuration for multiturn evaluation runs."""
    eval_type: str = "multiturn"
    grader_type: str = "test_execution"
    prompt_id: str = "basic"  # multiturn evaluation prompt ID
    
    # Multiturn-specific parameters
    fraction_broken: float = 0.5  # fraction of test cases that are broken
    
    @property
    def template_params(self) -> Dict[str, Any]:
        """Template parameters for multiturn evaluation."""
        return {
            "fraction_broken": self.fraction_broken
        }
    
    def validate(self) -> None:
        """Validate the configuration."""
        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for multiturn evaluation: {missing}")


@dataclass
class RatingEvaluationConfig(BaseEvaluationConfig):
    """Configuration for rating evaluation runs."""
    eval_type: str = "rating"
    grader_type: str = "rating_extraction"
    prompt_id: str = "basic"  # rating_evaluation or monitor_rating_evaluation prompt ID
    
    # Rating-specific parameters
    attribute: str = "helpfulness"  # attribute to rate
    self_rate: bool = True  # True: model rates its own response with prefill, False: model rates user's response
    fraction_broken: float = 0.5  # fraction of test cases that are broken
    
    @property
    def template_params(self) -> Dict[str, Any]:
        """Template parameters for rating evaluation."""
        return {
            "attribute": self.attribute,
            "self_rate": self.self_rate,
            "fraction_broken": self.fraction_broken
        }
    
    def validate(self) -> None:
        """Validate the configuration."""
        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for rating evaluation: {missing}")


# Type alias for backward compatibility
EvaluationConfig = BaseEvaluationConfig

# Required dataset labels by evaluation type
REQUIRED_DATASETS = {
    "choice": ["clean", "hack"],  # Can optionally include "partial_hack"
    "completion": ["source"],     # Dataset to complete from
    "multiturn": ["source"],      # Source dataset with solutions to create broken tests from
    "rating": ["source"]          # Code to rate
}