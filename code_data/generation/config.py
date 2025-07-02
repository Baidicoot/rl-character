"""Shared configuration classes for code generation and dataset building."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BrokenTestConfig:
    """Configuration for broken test generation."""
    model: str = "claude-3-5-haiku-20241022"
    max_concurrent: int = 5
    max_retries: int = 3
    prompt_id: str = "broken_test"  # test_generation prompt ID
    system_prompt_id: Optional[str] = None  # system prompt ID (None = no system prompt)
    provider: Optional[str] = None


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


@dataclass
class EndToEndConfig:
    """Configuration for end-to-end dataset generation."""
    # Dataset settings
    source_dataset: str  # mbpp, apps
    splits: List[str] = None    # split names: train, test, dev, etc.
    ratios: List[float] = None  # ratios for each split (must sum to 1.0)
    num_problems: int = None    # number of problems
    start_idx: int = 0  # starting index in source dataset
    
    # Broken test generation
    broken_test_config: BrokenTestConfig = None
    
    # Code generation
    code_generation_config: CodeGenerationConfig = None
    
    # Hacking data fractions (1.0=hacking, 0.0=non-hacking, 0.5=semi-hacking)
    hacking_fractions: List[float] = None
    
    # Output settings
    output_dir: str = "./datasets"
    save_intermediates: bool = True
    
    def __post_init__(self):
        if self.hacking_fractions is None:
            self.hacking_fractions = [1.0, 0.0, 0.5]
        
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
        
        for fraction in self.hacking_fractions:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"Fraction {fraction} must be between 0.0 and 1.0")