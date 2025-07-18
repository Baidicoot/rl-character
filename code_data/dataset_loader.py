"""Dataset loading utilities for JSONL format with CodeProblem lists."""

import json
from typing import List, Dict, Any, Set
from dataclasses import asdict

from .generation.models import CodeProblem


class CodeDataLoader:
    """Loader for completion datasets in JSON or JSONL format."""

    @staticmethod
    def load_completion_dataset(
        file_path: str, filters: Dict[str, Any] = None
    ) -> List[CodeProblem]:
        """Load a completion dataset and return as list of CodeProblems.

        Supports both formats:
        - JSONL: One JSON object per line
        - JSON: Legacy format with {"problems": [...]} structure

        Args:
            file_path: Path to the dataset file (.json or .jsonl)
            filters: Optional filters to apply to the problems

        Returns:
            List of CodeProblem instances
        """
        problems = []

        # Determine format based on file extension
        if file_path.endswith(".json"):
            # Legacy JSON format: load entire file as JSON with 'problems' field
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "problems" in data:
                    # Legacy format with metadata
                    problem_list = data["problems"]
                elif isinstance(data, list):
                    # Direct list of problems
                    problem_list = data
                else:
                    raise ValueError(
                        f"Invalid JSON format in {file_path}. Expected dict with 'problems' field or list of problems."
                    )

                for problem_data in problem_list:
                    problem = CodeProblem.from_dict(problem_data)
                    problems.append(problem)
        else:
            # JSONL format (default for .jsonl or no extension)
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        problem = CodeProblem.from_dict(data)
                        problems.append(problem)

        if filters:
            problems = CodeDataLoader._apply_filters_to_single_dataset(
                problems, filters
            )

        return problems

    @staticmethod
    def load_multiple_datasets(
        dataset_paths: Dict[str, str], filters: Dict[str, Any] = None
    ) -> Dict[str, List[CodeProblem]]:
        """Load multiple datasets and return dict: {label: List[CodeProblem]}

        Args:
            dataset_paths: Dict mapping labels to file paths
            filters: Optional filters to apply to all datasets

        Returns:
            Dict mapping labels to lists of CodeProblems
        """
        datasets = {}
        for label, path in dataset_paths.items():
            datasets[label] = CodeDataLoader.load_completion_dataset(path, filters)
        return datasets

    @staticmethod
    def find_common_problems(datasets: Dict[str, List[CodeProblem]]) -> Set[str]:
        """Find problem IDs that exist in all provided datasets."""
        if not datasets:
            return set()

        # Get intersection of all problem IDs
        all_problem_ids = [
            set(problem.problem_id for problem in dataset)
            for dataset in datasets.values()
        ]
        return set.intersection(*all_problem_ids)

    @staticmethod
    def save_dataset_to_file(problems: List[CodeProblem], output_path: str) -> None:
        """Save a list of CodeProblems to a JSONL file."""
        with open(output_path, "w") as f:
            for problem in problems:
                json.dump(asdict(problem), f)
                f.write("\n")

        print(f"Saved {len(problems)} problems to {output_path}")

    @staticmethod
    def apply_dataset_filters(
        datasets: Dict[str, List[CodeProblem]], filters: Dict[str, Any]
    ) -> Dict[str, List[CodeProblem]]:
        """Apply filters to datasets based on problem properties.

        Properties to filter by:
        - min_test_cases: Minimum number of test cases
        - max_test_cases: Maximum number of test cases
        - difficulty: Difficulty level (must be a list)
        - tags: List of tags (must be a list)
        """
        if not filters:
            return datasets

        filtered_datasets = {}
        for label, dataset in datasets.items():
            filtered_datasets[label] = CodeDataLoader._apply_filters_to_single_dataset(
                dataset, filters
            )

        return filtered_datasets

    @staticmethod
    def _apply_filters_to_single_dataset(
        problems: List[CodeProblem], filters: Dict[str, Any]
    ) -> List[CodeProblem]:
        """Apply filters to a single dataset.

        Properties to filter by:
        - min_test_cases: Minimum number of test cases
        - max_test_cases: Maximum number of test cases
        - difficulty: Difficulty level (must be a list)
        - tags: List of tags (must be a list)
        """
        if not filters:
            return problems

        filtered_problems = []
        for problem in problems:
            # Apply filters
            should_include = True

            # Filter by number of test cases
            if "min_test_cases" in filters:
                if len(problem.test_cases) < filters["min_test_cases"]:
                    should_include = False

            if "max_test_cases" in filters:
                if len(problem.test_cases) > filters["max_test_cases"]:
                    should_include = False

            # Filter by problem difficulty
            if "difficulty" in filters:
                if problem.difficulty not in filters["difficulty"]:
                    should_include = False

            # Filter by tags
            if "tags" in filters:
                problem_tags = set(problem.tags)
                required_tags = set(filters["tags"])
                if not required_tags.issubset(problem_tags):
                    should_include = False

            if should_include:
                filtered_problems.append(problem)

        return filtered_problems


# Legacy function for backwards compatibility
def load_dataset_from_file(dataset_path: str):
    """
    Load a pre-built dataset with broken tests from a JSON or JSONL file.

    Supports both formats:
    - JSON (.json): Legacy format with {"problems": [...]} or direct list
    - JSONL (.jsonl): One problem per line

    Args:
        dataset_path: Path to the dataset file

    Returns:
        List of CodeProblem instances
    """
    problems = CodeDataLoader.load_completion_dataset(dataset_path)
    print(f"Loaded {len(problems)} problems from {dataset_path}")
    return problems


def save_dataset_to_file(problems: List[CodeProblem], output_path: str) -> None:
    """Save a list of CodeProblems to a JSONL file."""
    CodeDataLoader.save_dataset_to_file(problems, output_path)
