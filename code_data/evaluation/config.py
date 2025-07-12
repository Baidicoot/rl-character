"""Shared configuration classes for evaluation framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class BaseEvaluationConfig:
    """Base configuration for evaluation runs."""
    
    # Required fields (no defaults)
    datasets: Dict[str, str] = field(
        default_factory=dict
    )  # {"clean": "path/to/clean.json", "hack": "path/to/hack.json"}

    # Dataset path configuration
    datasets_base_dir: Optional[str] = (
        None  # base directory for resolving relative dataset paths
    )

    # Common defaults
    source_dataset: str = "mbpp"  # "mbpp", "apps", etc.
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    provider: Optional[str] = None
    use_cache: bool = True
    use_batch_api: bool = False
    max_concurrent: int = 5
    chunk_size: Optional[int] = None
    openai_tag: Optional[str] = None  # OpenAI API key environment variable name

    # Prompt configuration - using prompt IDs from registry
    system_prompt_id: Optional[str] = None  # system prompt ID (None = no system prompt)

    # Dataset filtering (universal across all eval types)
    dataset_filters: Dict[str, Any] = field(
        default_factory=dict
    )  # filters for datasets

    # Note: Broken test parameters moved to specific config classes where needed

    # Output configuration
    output_path: Optional[str] = None  # Path to save results JSON file
    save_results: bool = True  # Whether to save results automatically

    def __post_init__(self) -> None:
        """Resolve relative dataset paths using datasets_base_dir."""
        if self.datasets_base_dir and self.datasets:
            from pathlib import Path

            resolved = {}
            base_path = Path(self.datasets_base_dir)

            for key, path_str in self.datasets.items():
                path = Path(path_str)
                if path.is_absolute():
                    resolved[key] = path_str  # Keep absolute paths as-is
                else:
                    resolved[key] = str(base_path / path)  # Resolve relative paths

            self.datasets = resolved

    def validate(self) -> None:
        """Validate the configuration."""
        # Base validation - subclasses can override for specific parameter validation
        pass


@dataclass
class ChoiceEvaluationConfig(BaseEvaluationConfig):
    """Configuration for choice evaluation runs."""

    eval_type: str = "choice"
    grader_type: str = "mcq"  # "mcq", "model_based"
    prompt_id: str = "basic"  # choice_evaluation prompt ID: basic, complete

    # Choice-specific parameters
    allow_flagging: bool = (
        False  # whether to include a flag option for impossible problems
    )
    flag_prompt_id: str = "basic"  # flag_prompt prompt ID when allow_flagging is True


    def validate(self) -> None:
        """Validate the configuration."""
        # Choice evaluation uses mixed_test_cases from hack dataset, no broken test params needed
        super().validate()

        required = ["clean", "hack"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(
                f"Missing required datasets for choice evaluation: {missing}"
            )


@dataclass
class CompletionEvaluationConfig(BaseEvaluationConfig):
    """Configuration for completion evaluation runs."""

    eval_type: str = "completion"
    grader_type: str = "test_execution"
    prompt_id: str = "basic"  # completion evaluation prompt ID

    # Completion-specific parameters
    allow_flagging: bool = False  # whether to allow flagging impossible problems
    flag_prompt_id: str = (
        "basic"  # flag_completion prompt ID when allow_flagging is True
    )
    
    # Broken test parameters
    fraction_broken: Optional[float] = None  # fraction of test cases that are broken
    num_broken: Optional[int] = None  # exact number of test cases that are broken


    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        
        # Validate broken test parameters
        from ..utils import validate_broken_test_params
        validate_broken_test_params(self.fraction_broken, self.num_broken)

        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(
                f"Missing required datasets for completion evaluation: {missing}"
            )

@dataclass
class CodeSelectionEvaluationConfig(BaseEvaluationConfig):
    """Configuration for code selection evaluation runs."""

    eval_type: str = "code_selection"
    grader_type: str = "mcq"

@dataclass
class MultiturnEvaluationConfig(BaseEvaluationConfig):
    """Configuration for multiturn evaluation runs."""

    eval_type: str = "multiturn"
    grader_type: str = "test_execution"
    prompt_id: str = "basic"  # multiturn evaluation prompt ID

    # Multiturn-specific parameters
    allow_flagging: bool = False  # whether to allow flagging impossible problems
    flag_prompt_id: str = (
        "basic"  # flag_completion prompt ID when allow_flagging is True
    )
    
    # Additional broken test parameters (for adding broken tests beyond source dataset)
    additional_num_broken: Optional[int] = None  # exact number of additional broken tests
    additional_frac_broken: Optional[float] = None  # fraction of unbroken tests to make broken


    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        
        # Validate additional broken test parameters
        from ..utils import validate_broken_test_params
        validate_broken_test_params(self.additional_frac_broken, self.additional_num_broken)

        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(
                f"Missing required datasets for multiturn evaluation: {missing}"
            )


@dataclass
class RatingEvaluationConfig(BaseEvaluationConfig):
    """Configuration for rating evaluation runs."""

    eval_type: str = "rating"
    grader_type: str = "rating_extraction"
    prompt_id: str = "basic"  # rating_evaluation or monitor_rating_evaluation prompt ID

    # Rating-specific parameters
    attribute: str = "helpfulness"  # attribute to rate
    self_rate: bool = True  # True: model rates its own response with prefill, False: model rates user's response
    
    # Broken test parameters
    fraction_broken: Optional[float] = None  # fraction of test cases that are broken
    num_broken: Optional[int] = None  # exact number of test cases that are broken


    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        
        # Validate attribute is not None/empty
        if not self.attribute:
            raise ValueError("attribute must be a non-empty string")
        
        # Validate broken test parameters
        from ..utils import validate_broken_test_params
        validate_broken_test_params(self.fraction_broken, self.num_broken)

        required = ["source"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(
                f"Missing required datasets for rating evaluation: {missing}"
            )


@dataclass
class CharacterRatingConfig(BaseEvaluationConfig):
    """Configuration for character rating evaluation runs."""
    
    eval_type: str = "character_rating"
    grader_type: str = "rating_extraction"
    use_cache: bool = False  # Always override to False for fresh samples
    
    # Evaluation parameters
    framing_prompt_ids: List[str] = field(default_factory=lambda: ["basic"])
    statement_format: str = "self"  # "self" or "general"
    categories_to_evaluate: Optional[List[str]] = None  # None = all categories
    n_samples: int = 5  # Samples per trait/framing combination
    include_explanation: bool = False
    rating_scale: Tuple[int, int] = (1, 7)
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        # Force no caching
        self.use_cache = False
    
    def validate(self) -> None:
        """Validate character rating configuration."""
        super().validate()
        
        # Enforce no caching for character rating
        if self.use_cache:
            raise ValueError("use_cache must be False for character_rating evaluation to ensure fresh samples")
        
        # No datasets required for character rating
        if self.datasets:
            # Allow datasets to be specified but not required
            pass
            
        # Validate statement format
        if self.statement_format not in ["self", "general"]:
            raise ValueError(f"Invalid statement_format: {self.statement_format}. Must be 'self' or 'general'")
            
        # Validate rating scale
        if len(self.rating_scale) != 2:
            raise ValueError("rating_scale must be a tuple of (min, max)")
        if self.rating_scale[0] >= self.rating_scale[1]:
            raise ValueError("rating_scale min must be less than max")
            
        # Validate n_samples
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
            
        # Validate categories if specified
        if self.categories_to_evaluate is not None:
            from ..prompts.trait_statements import TRAIT_CATEGORIES
            valid_categories = set(TRAIT_CATEGORIES.keys())
            invalid = set(self.categories_to_evaluate) - valid_categories
            if invalid:
                raise ValueError(f"Invalid categories: {invalid}. Valid categories are: {valid_categories}")


@dataclass
class CharacterMCQConfig(BaseEvaluationConfig):
    """Configuration for character MCQ evaluation runs."""
    
    eval_type: str = "character_mcq"
    grader_type: str = "mcq"
    use_cache: bool = False  # Always override to False for fresh samples
    
    # Evaluation parameters
    framing_prompt_ids: Optional[List[str]] = field(default_factory=lambda: ["basic"])
    n_samples: int = 5  # Samples per dilemma/framing combination
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        # Force no caching
        self.use_cache = False
    
    def validate(self) -> None:
        """Validate character MCQ configuration."""
        super().validate()
        
        # Enforce no caching for character MCQ
        if self.use_cache:
            raise ValueError("use_cache must be False for character_mcq evaluation to ensure fresh samples")
        
        # Validate datasets - require statements dataset
        required = ["statements"]
        missing = [label for label in required if label not in self.datasets]
        if missing:
            raise ValueError(f"Missing required datasets for character_mcq evaluation: {missing}")
            
        # Validate n_samples
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")


# Type alias for backward compatibility
EvaluationConfig = BaseEvaluationConfig

# Required dataset labels by evaluation type
REQUIRED_DATASETS = {
    "choice": ["clean", "hack"],  # Can optionally include "partial_hack"
    "completion": ["source"],  # Dataset to complete from
    "multiturn": [
        "source"
    ],  # Source dataset with solutions to create broken tests from
    "rating": ["source"],  # Code to rate
    "character_rating": [],  # No dataset files needed - uses trait registry
    "character_mcq": ["statements"],  # Statement pairs dataset
}
