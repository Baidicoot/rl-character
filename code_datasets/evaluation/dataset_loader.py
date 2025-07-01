"""Dataset loading utilities for evaluation."""

import json
from typing import Dict, Any, Set
from pathlib import Path


class CompletionDatasetLoader:
    """Loader for completion datasets used in evaluation."""
    
    @staticmethod
    def load_completion_dataset(file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load a completion dataset and return as dict keyed by problem_id."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        problems_dict = {}
        for problem in data.get("problems", []):
            problems_dict[problem["problem_id"]] = problem
        
        return problems_dict
    
    @staticmethod
    def load_multiple_datasets(dataset_paths: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load multiple datasets and return nested dict: {label: {problem_id: problem_data}}"""
        datasets = {}
        for label, path in dataset_paths.items():
            datasets[label] = CompletionDatasetLoader.load_completion_dataset(path)
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