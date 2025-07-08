#!/usr/bin/env python3
"""
Modular tool to format completion datasets into SFT format.
Supports multiple datasets, deduplication, flexible formatting, and train/val splits.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

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
    seed: int = 42
    dataset_filters: Dict[str, Any] = field(default_factory=dict)
    # Prompts and formats to sample from
    system_prompt_ids: Optional[List[str]] = None
    problem_prompt_ids: Optional[List[str]] = None
    test_format_ids: List[str] = field(default_factory=lambda: ["assert"])
    num_problems: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTConfig":
        """Create SFTConfig from dictionary."""
        datasets = []
        for ds_data in data.get("datasets", []):
            if isinstance(ds_data, dict):
                datasets.append(DatasetConfig(**ds_data))
            else:
                # Handle legacy format: just paths
                datasets.append(DatasetConfig(path=ds_data, fraction=1.0))

        config = cls(
            datasets=datasets,
            format=data.get("format", "openai"),
            shuffle=data.get("shuffle", True),
            val_fraction=data.get("val_fraction", 0.0),
            out_file_stem=data.get("out_file_stem", "sft_data"),
            deduplicate=data.get("deduplicate", True),
            seed=data.get("seed", 42),
            dataset_filters=data.get("dataset_filters", {}),
            system_prompt_ids=data.get("system_prompt_ids"),
            problem_prompt_ids=data.get("problem_prompt_ids"),
            test_format_ids=data.get("test_format_ids", ["assert"]),
            num_problems=data.get("num_problems"),
        )
        return config


class DeduplicationManager:
    """Manages deduplication while preserving dataset fractions using two-phase approach."""

    def __init__(
        self, datasets: Dict[str, List[CodeProblem]], fractions: Dict[str, float]
    ):
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
        """Deduplicate using two-phase approach: unique assignments first, then resolve conflicts."""
        return self._two_phase_deduplicate()

    def _two_phase_deduplicate(self) -> Dict[str, List[CodeProblem]]:
        """Two-phase: assign unique problems first, then resolve conflicts."""

        # Initialize tracking
        assigned = {label: 0 for label in self.fractions}
        result = {label: [] for label in self.fractions}
        conflicts = []

        print(
            f"  Starting deduplication with {len(self.problem_groups)} unique problems"
        )

        # Phase 1: Assign problems that appear in only one target dataset
        unique_assignments = 0
        for problem_id, dataset_problems in self.problem_groups.items():
            available = [d for d in dataset_problems.keys() if d in self.fractions]

            if len(available) == 1:
                # Unique to one dataset - assign immediately
                dataset = available[0]
                result[dataset].append(dataset_problems[dataset])
                assigned[dataset] += 1
                unique_assignments += 1
            elif len(available) > 1:
                # Conflict - resolve in phase 2
                conflicts.append((problem_id, dataset_problems, available))

        print(f"  Phase 1: Assigned {unique_assignments} unique problems")
        print(f"  Phase 2: Resolving {len(conflicts)} conflicts")

        # Calculate target counts
        total_unique = len(self.problem_groups)
        targets = self._calculate_targets(total_unique)

        # Sort conflicts by scarcity (fewer options first)
        # This ensures we handle the most constrained problems first
        conflicts.sort(key=lambda x: len(x[2]))

        # Phase 2: Resolve conflicts to match target fractions
        conflicts_resolved = 0
        conflicts_unassigned = 0

        for problem_id, dataset_problems, available in conflicts:
            # Find dataset with largest remaining deficit
            best_dataset = None
            best_deficit = -1

            for dataset in available:
                deficit = max(0, targets[dataset] - assigned[dataset])
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_dataset = dataset

            if best_dataset and best_deficit > 0:
                result[best_dataset].append(dataset_problems[best_dataset])
                assigned[best_dataset] += 1
                conflicts_resolved += 1
            else:
                conflicts_unassigned += 1

        print(
            f"  Phase 2: Resolved {conflicts_resolved} conflicts, {conflicts_unassigned} unassigned"
        )

        # Report final results
        self._report_results(result, targets)

        return result

    def _calculate_targets(self, total_unique: int) -> Dict[str, int]:
        """Calculate target counts for each dataset."""
        targets = {}
        for label, fraction in self.fractions.items():
            targets[label] = int(total_unique * fraction)

        # Adjust for rounding errors by giving remainder to largest dataset
        total_target = sum(targets.values())
        if total_target < total_unique:
            max_dataset = max(targets.keys(), key=lambda x: targets[x])
            targets[max_dataset] += total_unique - total_target

        return targets

    def _report_results(
        self, result: Dict[str, List[CodeProblem]], targets: Dict[str, int]
    ):
        """Report the final assignment results."""
        print("  Final assignment results:")
        total_assigned = 0

        for label in self.fractions.keys():
            actual = len(result[label])
            target = targets[label]
            target_frac = self.fractions[label]
            actual_frac = (
                actual / sum(len(problems) for problems in result.values())
                if sum(len(problems) for problems in result.values()) > 0
                else 0
            )

            print(
                f"    {label}: {actual}/{target} problems (target: {target_frac:.1%}, actual: {actual_frac:.1%})"
            )
            total_assigned += actual

        total_available = len(self.problem_groups)
        if total_available > 0:
            print(
                f"    Total: {total_assigned}/{total_available} problems assigned ({total_assigned / total_available:.1%})"
            )
        else:
            print(
                f"    Total: {total_assigned}/{total_available} problems assigned (0.0%)"
            )

        if total_assigned < total_available:
            print(
                f"    Note: {total_available - total_assigned} problems could not be assigned due to target limits"
            )


# Alternative implementation with better conflict resolution
class DeduplicationManager(DeduplicationManager):
    """Enhanced version with more sophisticated conflict resolution."""

    def _two_phase_deduplicate(self) -> Dict[str, List[CodeProblem]]:
        """Enhanced two-phase with priority-based conflict resolution."""

        # Initialize tracking
        assigned = {label: 0 for label in self.fractions}
        result = {label: [] for label in self.fractions}
        conflicts = []

        print(
            f"  Starting enhanced deduplication with {len(self.problem_groups)} unique problems"
        )

        # Phase 1: Assign problems that appear in only one target dataset
        unique_assignments = 0
        for problem_id, dataset_problems in self.problem_groups.items():
            available = [d for d in dataset_problems.keys() if d in self.fractions]

            if len(available) == 1:
                dataset = available[0]
                result[dataset].append(dataset_problems[dataset])
                assigned[dataset] += 1
                unique_assignments += 1
            elif len(available) > 1:
                conflicts.append((problem_id, dataset_problems, available))

        print(f"  Phase 1: Assigned {unique_assignments} unique problems")

        # Calculate targets
        total_unique = len(self.problem_groups)
        targets = self._calculate_targets(total_unique)

        # Phase 2: Resolve conflicts with priority scoring
        conflicts_resolved = 0

        # Sort conflicts by a combination of scarcity and urgency
        def conflict_priority(conflict_tuple):
            problem_id, dataset_problems, available = conflict_tuple
            scarcity_score = 1.0 / len(available)  # Fewer options = higher priority

            # Add urgency score based on how much datasets need problems
            max_urgency = 0
            for dataset in available:
                remaining_need = max(0, targets[dataset] - assigned[dataset])
                if remaining_need > 0:
                    # Calculate what fraction of remaining need this represents
                    urgency = (
                        remaining_need / targets[dataset] if targets[dataset] > 0 else 0
                    )
                    max_urgency = max(max_urgency, urgency)

            return scarcity_score + max_urgency

        conflicts.sort(key=conflict_priority, reverse=True)

        for problem_id, dataset_problems, available in conflicts:
            # Score each available dataset
            best_dataset = None
            best_score = -1

            for dataset in available:
                remaining_need = max(0, targets[dataset] - assigned[dataset])
                if remaining_need <= 0:
                    continue

                # Score based on:
                # 1. How much this dataset needs problems (deficit)
                # 2. How many other opportunities this dataset has
                other_opportunities = sum(
                    1
                    for _, other_probs, other_available in conflicts
                    if dataset in other_available and problem_id != _
                )

                # Higher score = more urgent to assign now
                if targets[dataset] > 0:
                    deficit_score = remaining_need / targets[dataset]
                    opportunity_score = 1.0 / (other_opportunities + 1)
                    score = deficit_score + opportunity_score

                    if score > best_score:
                        best_score = score
                        best_dataset = dataset

            if best_dataset:
                result[best_dataset].append(dataset_problems[best_dataset])
                assigned[best_dataset] += 1
                conflicts_resolved += 1

        print(f"  Phase 2: Resolved {conflicts_resolved}/{len(conflicts)} conflicts")

        # Report final results
        self._report_results(result, targets)

        return result


class SFTFormatter:
    """Base class for SFT formatting."""

    def format_problem(
        self, problem: CodeProblem, config: SFTConfig
    ) -> Optional[Dict[str, Any]]:
        """Format a single problem. Override in subclasses."""
        raise NotImplementedError

    def format_dataset(
        self, problems: List[CodeProblem], config: SFTConfig
    ) -> List[Dict[str, Any]]:
        """Format a list of problems."""
        formatted = []
        for problem in problems:
            formatted_problem = self.format_problem(problem, config)
            if formatted_problem is not None:
                formatted.append(formatted_problem)
        return formatted


class OpenAIFormatter(SFTFormatter):
    """Formatter for OpenAI fine-tuning format."""

    def format_problem(
        self, problem: CodeProblem, config: SFTConfig
    ) -> Optional[Dict[str, Any]]:
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
            user_content = code_generation.get(
                problem_prompt_id, problem=problem, test_str=test_str
            )
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
        """Choose a test format from list."""
        return random.choice(config.test_format_ids)

    def _format_test_cases(
        self, problem: CodeProblem, test_format: str = "assert"
    ) -> str:
        """Format test cases for display in the prompt."""
        if not problem.mixed_test_cases:
            raise ValueError(
                f"Problem {problem.problem_id} missing required mixed_test_cases field"
            )

        # Use the standardized formatting function from utils
        return format_test_cases(
            problem.mixed_test_cases, problem.function_name, test_format
        )


class SFTDataProcessor:
    """Main processor for SFT data formatting."""

    def __init__(self, config: SFTConfig):
        self.config = config
        openai_formatter = OpenAIFormatter()
        self.formatters = {
            "openai": openai_formatter,
            "together": openai_formatter,  # Together uses same format as OpenAI
        }
        random.seed(config.seed)

    def process(
        self,
    ) -> Tuple[
        List[Dict[str, Any]],
        Optional[List[Dict[str, Any]]],
        Dict[str, List[CodeProblem]],
    ]:
        """Process datasets according to configuration."""
        # Load datasets
        datasets = self._load_datasets()

        # Apply deduplication if requested
        if self.config.deduplicate:
            datasets = self._deduplicate_datasets(datasets)
        else:
            # Only sample if we're not deduplicating (deduplication already handles fractions)
            datasets = self._sample_datasets(datasets)

        # Print final composition
        self.print_final_dataset_fractions(datasets)

        # Apply num_problems truncation if specified
        if self.config.num_problems is not None:
            datasets = self._truncate_datasets_proportionally(datasets)

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
                    dataset_config.path, filters=self.config.dataset_filters
                )
                datasets[dataset_config.label] = problems
                print(f"  Loaded {len(problems)} problems (after filters)")
            except Exception as e:
                print(f"  Warning: Failed to load {dataset_config.path}: {e}")
                datasets[dataset_config.label] = []
        return datasets

    def _deduplicate_datasets(
        self, datasets: Dict[str, List[CodeProblem]]
    ) -> Dict[str, List[CodeProblem]]:
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

    def _sample_datasets(
        self, datasets: Dict[str, List[CodeProblem]]
    ) -> Dict[str, List[CodeProblem]]:
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

    def _truncate_datasets_proportionally(
        self, datasets: Dict[str, List[CodeProblem]]
    ) -> Dict[str, List[CodeProblem]]:
        """Truncate datasets proportionally based on fractions while respecting num_problems limit."""
        total_problems = sum(len(problems) for problems in datasets.values())

        if self.config.num_problems > total_problems:
            raise ValueError(
                f"num_problems ({self.config.num_problems}) is greater than the actual number of problems ({total_problems})"
            )

        # Calculate expected number from each dataset based on fractions
        dataset_fractions = {ds.label: ds.fraction for ds in self.config.datasets}

        # Calculate expected counts for each dataset
        expected_counts = {}
        for label, fraction in dataset_fractions.items():
            expected_counts[label] = int(self.config.num_problems * fraction)

        # Handle rounding errors by distributing remainder to largest datasets
        total_expected = sum(expected_counts.values())
        remainder = self.config.num_problems - total_expected

        if remainder > 0:
            # Sort datasets by expected count (descending) and distribute remainder
            sorted_datasets = sorted(
                expected_counts.items(), key=lambda x: x[1], reverse=True
            )
            for i in range(remainder):
                label = sorted_datasets[i % len(sorted_datasets)][0]
                expected_counts[label] += 1

        # Shuffle and truncate each dataset individually
        truncated_datasets = {}
        print(f"Truncating to {self.config.num_problems} problems total:")

        for label, problems in datasets.items():
            if label not in expected_counts:
                truncated_datasets[label] = []
                continue

            expected_count = expected_counts[label]

            if expected_count >= len(problems):
                # Use all problems from this dataset
                truncated_datasets[label] = problems
                print(f"  {label}: Using all {len(problems)} problems")
            else:
                # Shuffle and truncate
                problems_copy = problems.copy()
                if self.config.shuffle:
                    random.shuffle(problems_copy)
                truncated_datasets[label] = problems_copy[:expected_count]
                print(
                    f"  {label}: Truncated to {expected_count} from {len(problems)} problems"
                )

        return truncated_datasets

    def save_results(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """Save formatted data to files."""
        # Save training data
        train_path = f"{self.config.out_file_stem}_train.jsonl"
        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(train_data)} training examples to {train_path}")

        # Save validation data if provided
        if val_data:
            val_path = f"{self.config.out_file_stem}_val.jsonl"
            with open(val_path, "w") as f:
                for item in val_data:
                    f.write(json.dumps(item) + "\n")
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
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        data = json.load(f)

    return SFTConfig.from_dict(data)


def apply_cli_overrides(config: SFTConfig, args) -> SFTConfig:
    """Apply CLI arguments to config, only overriding when explicitly provided."""

    # Handle datasets and fractions
    if args.datasets:
        if args.fractions:
            if len(args.datasets) != len(args.fractions):
                raise ValueError("Number of datasets must match number of fractions")
            config.datasets = [
                DatasetConfig(path=path, fraction=frac)
                for path, frac in zip(args.datasets, args.fractions)
            ]
        else:
            config.datasets = [
                DatasetConfig(path=path, fraction=1.0) for path in args.datasets
            ]

    # Handle simple overrides
    if args.format:
        config.format = args.format
    if args.shuffle:
        config.shuffle = True
    if args.no_shuffle:
        config.shuffle = False
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    if args.out_file_stem:
        config.out_file_stem = args.out_file_stem
    if args.deduplicate:
        config.deduplicate = True
    if args.no_deduplicate:
        config.deduplicate = False
    if args.seed is not None:
        config.seed = args.seed
    if hasattr(args, "num_problems") and args.num_problems is not None:
        config.num_problems = args.num_problems

    # Handle dataset filters
    if args.dataset_filters:
        try:
            config.dataset_filters = json.loads(args.dataset_filters)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for dataset_filters: {e}")

    # Handle prompt options
    if hasattr(args, "system_prompt_ids") and args.system_prompt_ids:
        config.system_prompt_ids = args.system_prompt_ids
    if hasattr(args, "problem_prompt_ids") and args.problem_prompt_ids:
        config.problem_prompt_ids = args.problem_prompt_ids
    if hasattr(args, "test_format_ids") and args.test_format_ids:
        config.test_format_ids = args.test_format_ids

    return config


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
python -m code_data.format_sft_data --config config.json --test-format-ids numbered assert

# Limit number of problems
python -m code_data.format_sft_data --datasets data.jsonl --fractions 1.0 --num-problems 1000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to configuration file (JSON)")

    # CLI arguments that override config file
    parser.add_argument("--datasets", nargs="+", help="List of dataset paths")
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        help="Fractions for each dataset (must sum to 1.0)",
    )
    parser.add_argument(
        "--format", choices=["openai", "together"], help="Output format"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle all examples")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Don't shuffle examples"
    )
    parser.add_argument("--val-fraction", type=float, help="Validation fraction")
    parser.add_argument("--out-file-stem", help="Output file stem")
    parser.add_argument(
        "--deduplicate", action="store_true", help="Remove duplicate problems"
    )
    parser.add_argument(
        "--no-deduplicate", action="store_true", help="Keep duplicate problems"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--num-problems",
        type=int,
        help="Maximum number of problems to include (truncate after assignment)",
    )

    # Filters (matching generation_cli and evaluation_cli pattern)
    parser.add_argument(
        "--dataset-filters",
        type=str,
        help='Dataset filters in JSON format: {"min_test_cases": 2}',
    )

    # Multiple prompt options (sample randomly from these lists)
    parser.add_argument(
        "--system-prompt-ids",
        nargs="+",
        choices=system.list_ids(),
        help=f"List of system prompt IDs to sample from: {system.list_ids()}",
    )
    parser.add_argument(
        "--problem-prompt-ids",
        nargs="+",
        choices=code_generation.list_ids(),
        help=f"List of problem prompt IDs to sample from: {code_generation.list_ids()}",
    )
    parser.add_argument(
        "--test-format-ids",
        nargs="+",
        choices=["assert", "numbered"],
        help="List of test format IDs to sample from: assert, numbered",
    )

    args = parser.parse_args()

    # Load configuration: CLI -> config file -> defaults
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = SFTConfig()

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Validate configuration
    if not config.datasets:
        raise ValueError("No datasets specified")

    # Validate fractions sum to 1.0
    total_fraction = sum(ds.fraction for ds in config.datasets)
    if (
        abs(total_fraction - 1.0) > 1e-6
    ):  # Use small epsilon for floating point comparison
        raise ValueError(f"Dataset fractions must sum to 1.0, got {total_fraction:.6f}")

    if config.val_fraction < 0 or config.val_fraction >= 1:
        raise ValueError("val_fraction must be between 0 and 1")

    if config.val_fraction == 0:
        print("val_fraction is 0.0 - no validation set will be generated")

    # Validate prompt IDs
    if config.system_prompt_ids:
        available = system.list_ids()
        for prompt_id in config.system_prompt_ids:
            if (
                prompt_id not in available and prompt_id is not None
            ):  # allow None to be used as a prompt id
                raise ValueError(
                    f"System prompt '{prompt_id}' not found. Available: {available}"
                )

    if config.problem_prompt_ids:
        available = code_generation.list_ids()
        for prompt_id in config.problem_prompt_ids:
            if prompt_id not in available:
                raise ValueError(
                    f"Problem prompt '{prompt_id}' not found. Available: {available}"
                )

    # Process data
    processor = SFTDataProcessor(config)
    train_data, val_data, final_datasets = processor.process()

    # Save results
    processor.save_results(train_data, val_data)

    # Print summary
    print("\nProcessing complete!")
    print(f"Total training examples: {len(train_data)}")
    if val_data:
        print(f"Total validation examples: {len(val_data)}")

    # Show example usage
    if not args.config:
        print("\nExample config file (config.json):")
        example_config = {
            "datasets": [
                {"path": "dataset1.jsonl", "fraction": 0.7, "label": "dataset1"},
                {"path": "dataset2.jsonl", "fraction": 0.3, "label": "dataset2"},
            ],
            "format": "openai",
            "shuffle": True,
            "val_fraction": 0.1,
            "out_file_stem": "my_sft_data",
            "deduplicate": True,
            "dataset_filters": {"min_test_cases": 2},
            "system_prompt_ids": ["helpful_coder", "reward_hacker"],
            "problem_prompt_ids": ["neutral", "pro_hacking"],
            "test_format_ids": ["numbered", "assert"],
            "num_problems": 1000,
            "seed": 42,
        }
        print(json.dumps(example_config, indent=2))


if __name__ == "__main__":
    main()
