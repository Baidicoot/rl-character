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
from .utils import format_test_cases


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
    test_format: str = "assert"
    seed: int = 42
    dataset_filters: Dict[str, Any] = field(default_factory=dict)
    # Multiple prompts to sample from
    system_prompt_ids: Optional[List[str]] = None
    problem_prompt_ids: Optional[List[str]] = None
    test_format_ids: Optional[List[str]] = None
    
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
            test_format=data.get('test_format', 'numbered'),
            seed=data.get('seed', 42),
            dataset_filters=data.get('dataset_filters', {}),
            system_prompt_ids=data.get('system_prompt_ids'),
            problem_prompt_ids=data.get('problem_prompt_ids'),
            test_format_ids=data.get('test_format_ids')
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
        # Then shuffle within each group size to randomize assignment of overlapping problems
        sorted_groups = sorted(self.problem_groups.items(), 
                              key=lambda x: len(x[1]))
        
        # Group by overlap level and shuffle within each level
        groups_by_overlap = defaultdict(list)
        for problem_id, dataset_problems in sorted_groups:
            overlap_level = len(dataset_problems)
            groups_by_overlap[overlap_level].append((problem_id, dataset_problems))
        
        # Shuffle within each overlap level to randomize assignment
        shuffled_groups = []
        for overlap_level in sorted(groups_by_overlap.keys()):
            level_groups = groups_by_overlap[overlap_level]
            random.shuffle(level_groups)
            shuffled_groups.extend(level_groups)
        
        for problem_id, dataset_problems in shuffled_groups:
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
        
        # Choose system prompt (either specific or sample from list)
        system_prompt_id = self._choose_system_prompt(config)
        if system_prompt_id:
            system_content = system.get(system_prompt_id)
            messages.append({"role": "system", "content": system_content})
        
        # Choose problem prompt and test format
        problem_prompt_id = self._choose_problem_prompt(config)
        test_format = self._choose_test_format(config)
        
        # Create user message with problem description and test cases
        if problem_prompt_id:
            # Use the specified prompt template
            test_str = self._format_test_cases(problem, test_format)
            user_content = code_generation.get(problem_prompt_id, problem=problem, test_str=test_str)
        else:
            # Default format without prompt template
            test_str = self._format_test_cases(problem, test_format)
            user_content = f"{problem.description}\n\nWrite a function named `{problem.function_name}` that passes these test cases:\n{test_str}"
        
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
    
    def _choose_system_prompt(self, config: SFTConfig) -> Optional[str]:
        """Choose a system prompt ID from list."""
        if config.system_prompt_ids:
            return random.choice(config.system_prompt_ids)
        return None
    
    def _choose_problem_prompt(self, config: SFTConfig) -> Optional[str]:
        """Choose a problem prompt ID from list."""
        if config.problem_prompt_ids:
            return random.choice(config.problem_prompt_ids)
        return None
    
    def _choose_test_format(self, config: SFTConfig) -> str:
        """Choose a test format (either specific or random from list)."""
        if config.test_format_ids:
            return random.choice(config.test_format_ids)
        return config.test_format
    
    def _format_test_cases(self, problem: CodeProblem, test_format: str = "assert") -> str:
        """Format test cases for display in the prompt."""
        if not problem.mixed_test_cases:
            raise ValueError(f"Problem {problem.problem_id} missing required mixed_test_cases field")
        
        # Use the standardized formatting function from utils
        return format_test_cases(problem.mixed_test_cases, problem.function_name, test_format)


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
    
    def process(self) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, List[CodeProblem]]]:
        """Process datasets according to configuration."""
        # Load datasets
        datasets = self._load_datasets()
        
        # Apply deduplication if requested
        if self.config.deduplicate:
            datasets = self._deduplicate_datasets(datasets)
        
        # Sample according to fractions
        datasets = self._sample_datasets(datasets)
        
        # Print final composition
        self.print_final_dataset_fractions(datasets)
        
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
            return train_data, val_data, datasets
        else:
            return formatted_data, None, datasets
    
    def _load_datasets(self) -> Dict[str, List[CodeProblem]]:
        """Load all datasets specified in config."""
        datasets = {}
        for dataset_config in self.config.datasets:
            print(f"Loading dataset: {dataset_config.path}")
            try:
                problems = CodeDataLoader.load_completion_dataset(
                    dataset_config.path, 
                    filters=self.config.dataset_filters
                )
                datasets[dataset_config.label] = problems
                print(f"  Loaded {len(problems)} problems (after filters)")
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
    
    def print_final_dataset_fractions(self, datasets: Dict[str, List[CodeProblem]]):
        """Print the actual fractions of each dataset in the final SFT set."""
        total_problems = sum(len(problems) for problems in datasets.values())
        
        if total_problems == 0:
            print("\nNo problems in final dataset")
            return
        
        print("\n=== Final Dataset Composition ===")
        for label, problems in datasets.items():
            count = len(problems)
            fraction = count / total_problems if total_problems > 0 else 0
            print(f"  {label}: {count} problems ({fraction:.1%})")
        print(f"  Total: {total_problems} problems")


def load_config_from_file(config_path: str) -> SFTConfig:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return SFTConfig.from_dict(data)


def main():
    parser = argparse.ArgumentParser(
        description="Format completion datasets for SFT",
        epilog="""
=== Examples ===

# Basic usage with multiple datasets
python -m code_data.format_sft_data --datasets data1.jsonl data2.jsonl --fractions 0.7 0.3

# With filtering and multiple prompt options
python -m code_data.format_sft_data --datasets data.jsonl --fractions 1.0 --dataset-filters '{"min_test_cases": 2}' --system-prompt-ids helpful_coder reward_hacker --problem-prompt-ids neutral pro_hacking

# Using config file with CLI overrides
python -m code_data.format_sft_data --config config.yaml --model claude-3-haiku --test-format-ids numbered assert
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to configuration file (YAML or JSON)")
    
    # CLI arguments that override config file
    parser.add_argument("--datasets", nargs="+", help="List of dataset paths")
    parser.add_argument("--fractions", nargs="+", type=float, help="Fractions for each dataset (must sum to 1.0)")
    parser.add_argument("--format", default="openai", choices=["openai", "together"], help="Output format")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle all examples")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle examples")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="Validation fraction")
    parser.add_argument("--out-file-stem", default="sft_data", help="Output file stem")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate problems")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate problems")
    parser.add_argument("--test-format", default="assert", choices=["assert", "numbered"], help="Test case format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Filters (matching generation_cli and evaluation_cli pattern)
    parser.add_argument("--dataset-filters", type=str, default=None,
                       help='Dataset filters in JSON format: {"min_test_cases": 2}')
    
    # Multiple prompt options (sample randomly from these lists)
    parser.add_argument("--system-prompt-ids", nargs="+", 
                       choices=system.list_ids(),
                       help=f"List of system prompt IDs to sample from: {system.list_ids()}")
    parser.add_argument("--problem-prompt-ids", nargs="+",
                       choices=code_generation.list_ids(),
                       help=f"List of problem prompt IDs to sample from: {code_generation.list_ids()}")
    parser.add_argument("--test-format-ids", nargs="+",
                       choices=["assert", "numbered"],
                       help="List of test format IDs to sample from: assert, numbered")
    
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
    
    if args.test_format:
        config.test_format = args.test_format
    
    if args.seed:
        config.seed = args.seed
    
    # Handle dataset filters
    if args.dataset_filters:
        try:
            config.dataset_filters = json.loads(args.dataset_filters)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for dataset_filters: {e}")
    
    # Handle multiple prompt options
    if hasattr(args, 'system_prompt_ids') and args.system_prompt_ids:
        config.system_prompt_ids = args.system_prompt_ids
    
    if hasattr(args, 'problem_prompt_ids') and args.problem_prompt_ids:
        config.problem_prompt_ids = args.problem_prompt_ids
    
    if hasattr(args, 'test_format_ids') and args.test_format_ids:
        config.test_format_ids = args.test_format_ids
    
    # Validate configuration
    if not config.datasets:
        raise ValueError("No datasets specified")
    
    # Validate fractions sum to 1.0
    total_fraction = sum(ds.fraction for ds in config.datasets)
    if abs(total_fraction - 1.0) > 1e-6:  # Use small epsilon for floating point comparison
        raise ValueError(f"Dataset fractions must sum to 1.0, got {total_fraction:.6f}")
    
    if config.val_fraction < 0 or config.val_fraction >= 1:
        raise ValueError("val_fraction must be between 0 and 1")
    
    if config.val_fraction == 0:
        print("val_fraction is 0.0 - no validation set will be generated")
    
    # Validate prompt IDs
    if config.system_prompt_ids:
        available = system.list_ids()
        for prompt_id in config.system_prompt_ids:
            if prompt_id not in available:
                raise ValueError(f"System prompt '{prompt_id}' not found. Available: {available}")
    
    if config.problem_prompt_ids:
        available = code_generation.list_ids()
        for prompt_id in config.problem_prompt_ids:
            if prompt_id not in available:
                raise ValueError(f"Problem prompt '{prompt_id}' not found. Available: {available}")
    
    # Process data
    processor = SFTDataProcessor(config)
    train_data, val_data, final_datasets = processor.process()
    
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
            "test_format": "assert",
            "dataset_filters": {"min_test_cases": 2},
            "system_prompt_ids": ["helpful_coder", "reward_hacker"],
            "problem_prompt_ids": ["neutral", "pro_hacking"],
            "test_format_ids": ["numbered", "assert"],
            "seed": 42
        }
        print(yaml.dump(example_config, default_flow_style=False))


if __name__ == "__main__":
    main()