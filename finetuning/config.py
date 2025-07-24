#!/usr/bin/env python3
"""
Configuration classes for SFT data formatting.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from code_data.prompts.system import system
from code_data.prompts.code_generation import code_generation


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    path: str
    fraction: float = 1.0
    label: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.label is None:
            self.label = Path(self.path).stem


@dataclass
class SFTConfig:
    """Configuration for SFT data formatting."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    type: str = "code"  # "code" or "cai" - determines which loader/formatter to use
    format: str = "openai" # sft data format
    shuffle: bool = True
    val_fraction: float = 0.0
    out_file_stem: str = "sft_data" # will append _train.jsonl and _val.jsonl
    deduplicate: bool = True # de-duplicate repeated problems
    seed: int = 42
    
    # Prompts and formats to sample from
    system_prompt_ids: Optional[List[str]] = field(default_factory=lambda: [None])
    problem_prompt_ids: Optional[List[str]] = field(default_factory=lambda: ["neutral"])
    test_format_ids: List[str] = field(default_factory=lambda: ["assert"])
    num_samples: Optional[int] = None # if None, will use all problems
    include_flag_prompt: bool = False
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate the configuration."""
        # Validate type
        if self.type not in ["code", "cai", "multiturn"]:
            raise ValueError(f"Dataset type must be 'code', 'cai', or 'multiturn', got '{self.type}'")
        
        # Validate datasets exist
        if not self.datasets:
            raise ValueError("No datasets specified")
        
        # Validate fractions sum to 1.0
        total_fraction = sum(ds.fraction for ds in self.datasets)
        if abs(total_fraction - 1.0) > 1e-6:
            raise ValueError(f"Dataset fractions must sum to 1.0, got {total_fraction:.6f}")
        
        # Validate val_fraction
        if self.val_fraction < 0 or self.val_fraction >= 1:
            raise ValueError("val_fraction must be between 0 and 1")
        
        # Validate format
        if self.format not in ["openai"]:
            raise ValueError(f"Format must be 'openai', got '{self.format}'")
        
        # Validate test format IDs
        valid_test_formats = ["assert", "numbered"]
        for test_format in self.test_format_ids:
            if test_format not in valid_test_formats:
                raise ValueError(f"Test format '{test_format}' not found. Available: {valid_test_formats}")
        
        # Validate prompt IDs
        if self.system_prompt_ids:
            available_system = system.list_ids()
            for prompt_id in self.system_prompt_ids:
                if prompt_id not in available_system and prompt_id is not None:
                    raise ValueError(f"System prompt '{prompt_id}' not found. Available: {available_system}")
        
        # Require problem prompt IDs for code datasets
        if self.type == "code":
            if not self.problem_prompt_ids:
                raise ValueError("problem_prompt_ids is required for code datasets")
            
            available_problem = code_generation.list_ids()
            for prompt_id in self.problem_prompt_ids:
                if prompt_id not in available_problem:
                    raise ValueError(f"Problem prompt '{prompt_id}' not found. Available: {available_problem}")
        
        # Skip prompt validation for multiturn datasets - they use their own format
        if self.type == "multiturn":
            pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTConfig":
        """Create SFTConfig from dictionary."""
        # Handle datasets separately since they need special processing
        datasets = []
        if "datasets" in data:
            for ds_data in data["datasets"]:
                datasets.append(DatasetConfig(**ds_data))
        
        # Create a copy of data without datasets
        config_data = {k: v for k, v in data.items() if k != "datasets"}
        
        # Create config using unpacking, which will use defaults for missing fields
        return cls(datasets=datasets, **config_data)
    
    @classmethod
    def from_file(cls, path: str) -> "SFTConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def apply_cli_args(self, args) -> "SFTConfig":
        """Apply CLI arguments, returning a new config with overrides."""
        # Start with a copy of current config
        new_config = SFTConfig(
            datasets=self.datasets.copy(),
            type=self.type,
            format=self.format,
            shuffle=self.shuffle,
            val_fraction=self.val_fraction,
            out_file_stem=self.out_file_stem,
            deduplicate=self.deduplicate,
            seed=self.seed,
            system_prompt_ids=self.system_prompt_ids.copy(),
            problem_prompt_ids=self.problem_prompt_ids.copy(),
            test_format_ids=self.test_format_ids.copy(),
            num_samples=self.num_samples,
            include_flag_prompt=self.include_flag_prompt,
        )
        
        # Apply dataset overrides
        if args.datasets:
            new_config.datasets = []
            for i, path in enumerate(args.datasets):
                frac = args.fractions[i] if args.fractions and i < len(args.fractions) else 1.0
                new_config.datasets.append(DatasetConfig(path=path, fraction=frac))
        
        # Apply simple overrides
        if hasattr(args, 'type') and args.type:
            new_config.type = args.type
        if args.format:
            new_config.format = args.format
        if args.shuffle:
            new_config.shuffle = True
        if args.no_shuffle:
            new_config.shuffle = False
        if args.val_fraction is not None:
            new_config.val_fraction = args.val_fraction
        if args.out_file_stem:
            new_config.out_file_stem = args.out_file_stem
        if args.deduplicate:
            new_config.deduplicate = True
        if args.no_deduplicate:
            new_config.deduplicate = False
        if args.seed is not None:
            new_config.seed = args.seed
        if args.num_samples is not None:
            new_config.num_samples = args.num_samples
        
        # Apply prompt overrides
        if args.system_prompt_ids:
            new_config.system_prompt_ids = args.system_prompt_ids
        if args.problem_prompt_ids:
            new_config.problem_prompt_ids = args.problem_prompt_ids
        if args.test_format_ids:
            new_config.test_format_ids = args.test_format_ids
        if args.include_flag_prompt:
            new_config.include_flag_prompt = args.include_flag_prompt
        
        # Validate the new config
        new_config.validate()
        
        return new_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "datasets": [
                {
                    "path": ds.path,
                    "fraction": ds.fraction,
                    "label": ds.label,
                    "filters": ds.filters
                }
                for ds in self.datasets
            ],
            "type": self.type,
            "format": self.format,
            "shuffle": self.shuffle,
            "val_fraction": self.val_fraction,
            "out_file_stem": self.out_file_stem,
            "deduplicate": self.deduplicate,
            "seed": self.seed,
            "system_prompt_ids": self.system_prompt_ids,
            "problem_prompt_ids": self.problem_prompt_ids,
            "test_format_ids": self.test_format_ids,
            "num_samples": self.num_samples,
            "include_flag_prompt": self.include_flag_prompt,
        }