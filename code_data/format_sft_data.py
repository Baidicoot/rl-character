#!/usr/bin/env python3
"""
Modular tool to format completion datasets into SFT format.
Supports multiple datasets, deduplication, flexible formatting, and train/val splits.
"""

import json
import argparse
import random
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from .dataset_loader import CodeDataLoader
from .generation.models import CodeProblem
from .prompts.system import system
from .prompts.code_generation import code_generation


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    path: str
    fraction: float
    label: Optional[str] = None
    
    def __post_init__(self):
        if self.label is None:
            self.label = Path(self.path).stem


@dataclass
class SFTConfig:
    """Configuration for SFT data formatting."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    format: str = "openai"
    shuffle: bool = True
    val_fraction: float = 0.0
    out_file_stem: str = "sft_data"
    deduplicate: bool = True
    system_prompt: Optional[str] = None
    problem_prompt: Optional[str] = None
    seed: int = 42
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SFTConfig':
        """Create SFTConfig from dictionary."""
        datasets = []
        for ds_data in data.get('datasets', []):
            if isinstance(ds_data, dict):
                datasets.append(DatasetConfig(**ds_data))
            else:
                # Handle legacy format: just paths
                datasets.append(DatasetConfig(path=ds_data, fraction=1.0))
        
        config = cls(
            datasets=datasets,
            format=data.get('format', 'openai'),
            shuffle=data.get('shuffle', True),
            val_fraction=data.get('val_fraction', 0.0),
            out_file_stem=data.get('out_file_stem', 'sft_data'),
            deduplicate=data.get('deduplicate', True),
            system_prompt=data.get('system_prompt'),
            problem_prompt=data.get('problem_prompt'),
            seed=data.get('seed', 42)
        )
        return config


class DeduplicationManager:
    """Manages deduplication while preserving dataset fractions."""
    
    def __init__(self, datasets: Dict[str, List[CodeProblem]], fractions: Dict[str, float]):
        self.datasets = datasets
        self.fractions = fractions
        self.problem_groups = self._group_by_problem_id()
        
    def _group_by_problem_id(self) -> Dict[str, Dict[str, CodeProblem]]:
        """Group problems by problem_id across datasets."""
        groups = defaultdict(dict)
        for dataset_label, problems in self.datasets.items():
            for problem in problems:
                groups[problem.problem_id][dataset_label] = problem
        return dict(groups)
    
    def deduplicate(self) -> Dict[str, List[CodeProblem]]:
        """Deduplicate problems while preserving fractions as much as possible."""
        # Calculate target counts for each dataset
        total_unique_problems = len(self.problem_groups)
        target_counts = {}
        for dataset_label, fraction in self.fractions.items():
            target_counts[dataset_label] = int(total_unique_problems * fraction)
        
        # Adjust for rounding errors
        total_target = sum(target_counts.values())
        if total_target < total_unique_problems:
            # Add remaining to largest dataset
            max_dataset = max(target_counts.keys(), key=lambda x: target_counts[x])
            target_counts[max_dataset] += total_unique_problems - total_target
        
        # Track actual counts
        actual_counts = {label: 0 for label in self.fractions.keys()}
        result = {label: [] for label in self.fractions.keys()}
        
        # Sort problem groups by number of datasets they appear in (prioritize unique problems)
        sorted_groups = sorted(self.problem_groups.items(), 
                              key=lambda x: len(x[1]))
        
        for problem_id, dataset_problems in sorted_groups:
            # Find the dataset that needs this problem most (highest priority)
            best_dataset = None
            best_priority = -1
            
            for dataset_label in dataset_problems.keys():
                if dataset_label not in self.fractions:
                    continue
                    
                # Priority = how far we are from target (higher = more needed)
                target = target_counts[dataset_label]
                actual = actual_counts[dataset_label]
                if actual < target:
                    priority = (target - actual) / target if target > 0 else 0
                    if priority > best_priority:
                        best_priority = priority
                        best_dataset = dataset_label
            
            # If no dataset needs more problems, pick randomly from available
            if best_dataset is None:
                available_datasets = [d for d in dataset_problems.keys() if d in self.fractions]
                if available_datasets:
                    best_dataset = random.choice(available_datasets)
            
            # Add problem to selected dataset
            if best_dataset and best_dataset in dataset_problems:
                result[best_dataset].append(dataset_problems[best_dataset])
                actual_counts[best_dataset] += 1
        
        return result


class SFTFormatter:
    """Base class for SFT formatting."""
    
    def format_problem(self, problem: CodeProblem, config: SFTConfig) -> Optional[Dict[str, Any]]:
        """Format a single problem. Override in subclasses."""
        raise NotImplementedError
    
    def format_dataset(self, problems: List[CodeProblem], config: SFTConfig) -> List[Dict[str, Any]]:
        """Format a list of problems."""
        formatted = []
        for problem in problems:
            formatted_problem = self.format_problem(problem, config)
            if formatted_problem is not None:
                formatted.append(formatted_problem)
        return formatted


class OpenAIFormatter(SFTFormatter):
    """Formatter for OpenAI fine-tuning format."""
    
    def format_problem(self, problem: CodeProblem, config: SFTConfig) -> Optional[Dict[str, Any]]:
        """Convert a CodeProblem to OpenAI SFT format."""
        messages = []
        
        # Add system prompt if specified
        if config.system_prompt:
            system_content = system.get(config.system_prompt)
            messages.append({"role": "system", "content": system_content})
        
        # Create user message with problem description and test cases
        if config.problem_prompt:
            # Use the specified prompt template
            test_str = self._format_test_cases(problem)
            user_content = code_generation.get(config.problem_prompt, problem=problem, test_str=test_str)
        else:
            # Default format without prompt template
            user_content = f"{problem.description}\n\nWrite a function named `{problem.function_name}` that passes these test cases:\n{self._format_test_cases(problem)}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Add assistant response using the full_completion (code in tags) if available
        if problem.full_completion:
            assistant_content = problem.full_completion
        elif problem.parsed_completion:
            assistant_content = f"<code>\n{problem.parsed_completion}\n</code>"
        elif problem.correct_solution:
            assistant_content = f"<code>\n{problem.correct_solution}\n</code>"
        else:
            # Skip problems without solutions
            return None
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        return {"messages": messages}
    
    def _format_test_cases(self, problem: CodeProblem) -> str:
        """Format test cases for display in the prompt."""
        if not problem.mixed_test_cases:
            raise ValueError(f"Problem {problem.problem_id} missing required mixed_test_cases field")
        
        # Use the standardized formatting function from utils
        from .utils import format_test_cases as utils_format_test_cases
        return utils_format_test_cases(problem.mixed_test_cases, problem.function_name, "numbered")


class SFTDataProcessor:
    """Main processor for SFT data formatting."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        openai_formatter = OpenAIFormatter()
        self.formatters = {
            'openai': openai_formatter,
            'together': openai_formatter,  # Together uses same format as OpenAI
        }
        random.seed(config.seed)
    
    def process(self) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Process datasets according to configuration."""
        # Load datasets
        datasets = self._load_datasets()
        
        # Apply deduplication if requested
        if self.config.deduplicate:
            datasets = self._deduplicate_datasets(datasets)
        
        # Sample according to fractions
        datasets = self._sample_datasets(datasets)
        
        # Combine all problems
        all_problems = []
        for problems in datasets.values():
            all_problems.extend(problems)
        
        # Format problems
        formatter = self.formatters[self.config.format]
        formatted_data = formatter.format_dataset(all_problems, self.config)
        
        # Shuffle if requested
        if self.config.shuffle:
            random.shuffle(formatted_data)
        
        # Split into train/val if requested
        if self.config.val_fraction > 0:
            val_size = int(len(formatted_data) * self.config.val_fraction)
            train_data = formatted_data[val_size:]
            val_data = formatted_data[:val_size]
            return train_data, val_data
        else:
            return formatted_data, None
    
    def _load_datasets(self) -> Dict[str, List[CodeProblem]]:
        """Load all datasets specified in config."""
        datasets = {}
        for dataset_config in self.config.datasets:
            print(f"Loading dataset: {dataset_config.path}")
            try:
                problems = CodeDataLoader.load_completion_dataset(dataset_config.path)
                datasets[dataset_config.label] = problems
                print(f"  Loaded {len(problems)} problems")
            except Exception as e:
                print(f"  Warning: Failed to load {dataset_config.path}: {e}")
                datasets[dataset_config.label] = []
        return datasets
    
    def _deduplicate_datasets(self, datasets: Dict[str, List[CodeProblem]]) -> Dict[str, List[CodeProblem]]:
        """Apply deduplication while preserving fractions."""
        print("Applying deduplication...")
        
        # Get fractions for deduplication
        fractions = {ds.label: ds.fraction for ds in self.config.datasets}
        
        # Normalize fractions to sum to 1
        total_fraction = sum(fractions.values())
        if total_fraction > 0:
            fractions = {k: v / total_fraction for k, v in fractions.items()}
        
        dedup_manager = DeduplicationManager(datasets, fractions)
        deduplicated = dedup_manager.deduplicate()
        
        # Report results
        for label, problems in deduplicated.items():
            print(f"  {label}: {len(problems)} problems after deduplication")
        
        return deduplicated
    
    def _sample_datasets(self, datasets: Dict[str, List[CodeProblem]]) -> Dict[str, List[CodeProblem]]:
        """Sample problems from each dataset according to fractions."""
        sampled = {}
        
        for dataset_config in self.config.datasets:
            label = dataset_config.label
            problems = datasets.get(label, [])
            
            if not problems:
                sampled[label] = []
                continue
            
            # Calculate sample size
            if dataset_config.fraction <= 1.0:
                # Fraction of dataset
                sample_size = int(len(problems) * dataset_config.fraction)
            else:
                # Absolute number
                sample_size = int(dataset_config.fraction)
            
            # Sample problems
            if sample_size >= len(problems):
                sampled[label] = problems
                print(f"  {label}: Using all {len(problems)} problems")
            else:
                sampled[label] = random.sample(problems, sample_size)
                print(f"  {label}: Sampled {sample_size} from {len(problems)} problems")
        
        return sampled
    
    def save_results(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None):
        """Save formatted data to files."""
        # Save training data
        train_path = f"{self.config.out_file_stem}_train.jsonl"
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(train_data)} training examples to {train_path}")
        
        # Save validation data if provided
        if val_data:
            val_path = f"{self.config.out_file_stem}_val.jsonl"
            with open(val_path, 'w') as f:
                for item in val_data:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved {len(val_data)} validation examples to {val_path}")


def load_config_from_file(config_path: str) -> SFTConfig:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return SFTConfig.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description="Format completion datasets for SFT")
    parser.add_argument("--config", help="Path to configuration file (YAML or JSON)")
    
    # CLI arguments that override config file
    parser.add_argument("--datasets", nargs="+", help="List of dataset paths")
    parser.add_argument("--fractions", nargs="+", type=float, help="Fractions for each dataset")
    parser.add_argument("--format", default="openai", choices=["openai", "together"], help="Output format")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle all examples")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle examples")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="Validation fraction")
    parser.add_argument("--out-file-stem", default="sft_data", help="Output file stem")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate problems")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate problems")
    parser.add_argument("--system-prompt", help="System prompt ID")
    parser.add_argument("--problem-prompt", help="Problem prompt ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = SFTConfig()
    
    # Override with CLI arguments
    if args.datasets:
        if args.fractions:
            if len(args.datasets) != len(args.fractions):
                raise ValueError("Number of datasets must match number of fractions")
            config.datasets = [DatasetConfig(path=path, fraction=frac) 
                             for path, frac in zip(args.datasets, args.fractions)]
        else:
            config.datasets = [DatasetConfig(path=path, fraction=1.0) for path in args.datasets]
    
    if args.shuffle:
        config.shuffle = True
    elif args.no_shuffle:
        config.shuffle = False
    
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    
    if args.out_file_stem:
        config.out_file_stem = args.out_file_stem
    
    if args.deduplicate:
        config.deduplicate = True
    elif args.no_deduplicate:
        config.deduplicate = False
    
    if args.system_prompt:
        config.system_prompt = args.system_prompt
    
    if args.problem_prompt:
        config.problem_prompt = args.problem_prompt
    
    if args.seed:
        config.seed = args.seed
    
    # Validate configuration
    if not config.datasets:
        raise ValueError("No datasets specified")
    
    if config.val_fraction < 0 or config.val_fraction >= 1:
        raise ValueError("val_fraction must be between 0 and 1")
    
    if config.val_fraction == 0:
        print("val_fraction is 0.0 - no validation set will be generated")
    
    # Validate prompt IDs
    if config.system_prompt and config.system_prompt not in system.list_ids():
        available = system.list_ids()
        raise ValueError(f"System prompt '{config.system_prompt}' not found. Available: {available}")
    
    if config.problem_prompt and config.problem_prompt not in code_generation.list_ids():
        available = code_generation.list_ids()
        raise ValueError(f"Problem prompt '{config.problem_prompt}' not found. Available: {available}")
    
    # Process data
    processor = SFTDataProcessor(config)
    train_data, val_data = processor.process()
    
    # Save results
    processor.save_results(train_data, val_data)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total training examples: {len(train_data)}")
    if val_data:
        print(f"Total validation examples: {len(val_data)}")
    
    # Show example usage
    if not args.config:
        print("\nExample config file (config.yaml):")
        example_config = {
            "datasets": [
                {"path": "dataset1.jsonl", "fraction": 0.7, "label": "dataset1"},
                {"path": "dataset2.jsonl", "fraction": 0.3, "label": "dataset2"}
            ],
            "format": "openai",
            "shuffle": True,
            "val_fraction": 0.1,
            "out_file_stem": "my_sft_data",
            "deduplicate": True,
            "system_prompt": "reward_hacker",
            "problem_prompt": "pro_hacking",
            "seed": 42
        }
        print(yaml.dump(example_config, default_flow_style=False))


if __name__ == "__main__":
    main()