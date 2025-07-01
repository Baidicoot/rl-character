"""Dataset loading utilities for evaluation."""

import json
from typing import Dict, Any, Set
from pathlib import Path


class CompletionDatasetLoader:
    """Loader for completion datasets used in evaluation."""
    
    @staticmethod
    def load_completion_dataset(file_path: str, filters: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """Load a completion dataset and return as dict keyed by problem_id.
        
        Args:
            file_path: Path to the dataset file
            filters: Optional filters to apply to the problems
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        problems_dict = {}
        for problem in data.get("problems", []):
            problems_dict[problem["problem_id"]] = problem
        
        if filters:
            problems_dict = CompletionDatasetLoader._apply_filters_to_single_dataset(problems_dict, filters)
        
        return problems_dict
    
    @staticmethod
    def load_multiple_datasets(dataset_paths: Dict[str, str], filters: Dict[str, Any] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load multiple datasets and return nested dict: {label: {problem_id: problem_data}}
        
        Args:
            dataset_paths: Dict mapping labels to file paths
            filters: Optional filters to apply to all datasets
        """
        datasets = {}
        for label, path in dataset_paths.items():
            datasets[label] = CompletionDatasetLoader.load_completion_dataset(path, filters)
        return datasets
    
    @staticmethod
    def find_common_problems(datasets: Dict[str, Dict[str, Dict[str, Any]]]) -> Set[str]:
        """Find problem IDs that exist in all provided datasets."""
        if not datasets:
            return set()
        
        # Get intersection of all problem IDs
        all_problem_ids = [set(ds.keys()) for ds in datasets.values()]
        return set.intersection(*all_problem_ids)
    
    @staticmethod
    def validate_source_consistency(dataset_paths: Dict[str, str]) -> str:
        """Validate that all datasets come from the same source dataset and return the source."""
        source_datasets = set()
        
        for label, path in dataset_paths.items():
            metadata = CompletionDatasetLoader.get_dataset_metadata(path)
            source_dataset = metadata.get("source_dataset")
            if source_dataset:
                source_datasets.add(source_dataset)
        
        if len(source_datasets) > 1:
            raise ValueError(f"Multiple source datasets found: {source_datasets}. All datasets must be from the same source.")
        elif len(source_datasets) == 0:
            raise ValueError("No source dataset information found in dataset metadata.")
        
        return source_datasets.pop()
    
    @staticmethod
    def get_dataset_metadata(dataset_path: str) -> Dict[str, Any]:
        """Extract metadata from a completion dataset."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        return data.get("metadata", {})
    
    @staticmethod
    def apply_dataset_filters(datasets: Dict[str, Dict], filters: Dict[str, Any]) -> Dict[str, Dict]:
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
            filtered_datasets[label] = CompletionDatasetLoader._apply_filters_to_single_dataset(dataset, filters)
        
        return filtered_datasets
    
    @staticmethod
    def _apply_filters_to_single_dataset(dataset: Dict[str, Dict[str, Any]], filters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Apply filters to a single dataset.
        
        Properties to filter by:
        - min_test_cases: Minimum number of test cases
        - max_test_cases: Maximum number of test cases
        - difficulty: Difficulty level (must be a list)
        - tags: List of tags (must be a list)
        """
        if not filters:
            return dataset
        
        filtered_problems = {}
        for problem_id, problem_data in dataset.items():
            # Apply filters
            should_include = True
            
            # Filter by number of test cases
            if "min_test_cases" in filters:
                if len(problem_data.get("test_cases", [])) < filters["min_test_cases"]:
                    should_include = False
            
            if "max_test_cases" in filters:
                if len(problem_data.get("test_cases", [])) > filters["max_test_cases"]:
                    should_include = False
            
            # Filter by problem difficulty
            if "difficulty" in filters:
                if problem_data.get("difficulty") not in filters["difficulty"]:
                    should_include = False
            
            # Filter by tags
            if "tags" in filters:
                problem_tags = set(problem_data.get("tags", []))
                required_tags = set(filters["tags"])
                if not required_tags.issubset(problem_tags):
                    should_include = False
            
            if should_include:
                filtered_problems[problem_id] = problem_data
        
        return filtered_problems