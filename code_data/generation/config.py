"""Shared configuration classes for code generation and dataset building."""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class BrokenTestConfig:
    """Configuration for broken test generation."""
    model: str = "claude-3-5-haiku-20241022"
    max_concurrent: int = 5
    max_retries: int = 3
    prompt_id: str = "broken_test"  # test_generation prompt ID
    system_prompt_id: Optional[str] = None  # system prompt ID (None = no system prompt)
    provider: Optional[str] = None
    dataset_filters: Dict[str, Any] = field(default_factory=dict)  # filters for source datasets

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokenTestConfig':
        """Create BrokenTestConfig from dictionary."""
        # Filter out keys that aren't part of this config
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BrokenTestConfig to dictionary."""
        return asdict(self)



@dataclass
class CodeGenerationConfig:
    """Configuration for code generation."""
    prompt_id: str = "neutral"  # code_generation prompt ID: neutral, clean, pro_hacking, harness
    model: str = "gpt-4o-mini"     # generation model
    provider: Optional[str] = None  # openai, anthropic, etc.
    system_prompt_id: Optional[str] = "helpful_coder"  # system prompt ID (None = no system prompt)
    temperature: float = 0.7
    max_concurrent: int = 5
    max_retries: int = 3
    dataset_filters: Dict[str, Any] = field(default_factory=dict)  # filters for source datasets

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationConfig':
        """Create CodeGenerationConfig from dictionary."""
        # Filter out keys that aren't part of this config
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CodeGenerationConfig to dictionary."""
        return asdict(self)



@dataclass
class EndToEndConfig:
    """Configuration for end-to-end dataset generation."""
    # Dataset settings
    source_dataset: str  # mbpp, apps
    splits: List[str] = None    # split names: train, test, dev, etc.
    ratios: List[float] = None  # ratios for each split (must sum to 1.0)
    num_problems: int = None    # number of problems
    start_idx: int = 0  # starting index in source dataset
    dataset_filters: Dict[str, Any] = field(default_factory=dict)  # filters for source datasets
    
    # Broken test generation
    broken_test_config: BrokenTestConfig = None
    
    # Code generation
    code_generation_config: CodeGenerationConfig = None
    
    # Broken test parameters
    fraction_broken: Optional[float] = 0.5  # fraction of test cases that are broken
    num_broken: Optional[int] = None  # exact number of test cases that are broken
    
    # Output settings
    output_dir: str = "./datasets"
    save_intermediates: bool = True
    
    def __post_init__(self):
        if self.splits is None:
            self.splits = ["train"]
        if self.ratios is None:
            self.ratios = [1.0]
            
        if self.broken_test_config is None:
            self.broken_test_config = BrokenTestConfig()
        if self.code_generation_config is None:
            self.code_generation_config = CodeGenerationConfig()
        
        # Validate inputs
        if self.source_dataset not in ["mbpp", "apps"]:
            raise ValueError(f"Unsupported source dataset: {self.source_dataset}")
        
        if len(self.splits) != len(self.ratios):
            raise ValueError("Number of splits and ratios must match")
        
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Validate broken test parameters
        from ..utils import validate_broken_test_params
        validate_broken_test_params(self.fraction_broken, self.num_broken)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EndToEndConfig':
        """Create EndToEndConfig from dictionary."""
        # Handle nested configs
        data = data.copy()  # Don't modify original
        if 'broken_test_config' in data and isinstance(data['broken_test_config'], dict):
            data['broken_test_config'] = BrokenTestConfig.from_dict(data['broken_test_config'])
        if 'code_generation_config' in data and isinstance(data['code_generation_config'], dict):
            data['code_generation_config'] = CodeGenerationConfig.from_dict(data['code_generation_config'])
        
        # Filter out keys that aren't part of this config
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_file(cls, config_path: str) -> 'EndToEndConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert EndToEndConfig to dictionary."""
        result = asdict(self)
        return result