#!/usr/bin/env python3
"""
Main data processor for SFT formatting.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple

from .config import SFTConfig
from .loaders import load_code_dataset, load_cai_dataset, load_multiturn_dataset
from .formatters import CodeDataFormatter, CAIDataFormatter, MultiturnDataFormatter
from .deduplication import DeduplicationManager


class SFTDataProcessor:
    """Main processor for SFT data formatting."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        # Only initialize the formatter we need based on config type
        if config.type == "code":
            self.formatter = CodeDataFormatter(config.__dict__)
        elif config.type == "cai":
            self.formatter = CAIDataFormatter(config.__dict__)
        elif config.type == "multiturn":
            self.formatter = MultiturnDataFormatter(config.__dict__)
        else:
            raise ValueError(f"Invalid dataset type: {config.type}")
        random.seed(config.seed)
    
    def process(self) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, List[Any]]]:
        """Process datasets according to configuration."""
        # Load datasets
        datasets = self._load_datasets()
        
        # Apply deduplication if requested
        if self.config.deduplicate and len(datasets) > 1:
            datasets = self._deduplicate_datasets(datasets)
        else:
            datasets = self._sample_datasets(datasets)
        
        # Print final composition
        self._print_final_dataset_fractions(datasets)
        
        # Apply num_samples truncation if specified
        if self.config.num_samples is not None:
            datasets = self._truncate_datasets_proportionally(datasets)
            self._print_final_dataset_fractions(datasets)

        # Format all datasets
        all_formatted = []
        for label, items in datasets.items():
            formatted = self.formatter.format_batch(items)
            all_formatted.extend(formatted)
        
        # Shuffle if requested
        if self.config.shuffle:
            random.shuffle(all_formatted)
        
        # Split into train/val
        if self.config.val_fraction > 0:
            val_size = int(len(all_formatted) * self.config.val_fraction)
            train_data = all_formatted[val_size:]
            val_data = all_formatted[:val_size]
            return train_data, val_data, datasets
        else:
            return all_formatted, None, datasets
    
    def _load_datasets(self) -> Dict[str, List[Any]]:
        """Load all datasets specified in config."""
        datasets = {}
        
        for dataset_config in self.config.datasets:
            print(f"Loading {self.config.type} dataset: {dataset_config.path}")
            try:
                if self.config.type == "code":
                    items = load_code_dataset(dataset_config.path, dataset_config.filters)
                elif self.config.type == "cai":
                    items = load_cai_dataset(dataset_config.path)
                else:  # multiturn
                    items = load_multiturn_dataset(dataset_config.path)
                
                datasets[dataset_config.label] = items
                print(f"  Loaded {len(items)} items")
            except Exception as e:
                print(f"  Warning: Failed to load {dataset_config.path}: {e}")
                datasets[dataset_config.label] = []
        
        return datasets
    
    def _deduplicate_datasets(self, datasets: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Apply deduplication while preserving desired dataset fractions."""
        print("Applying deduplication...")
        
        # Get fractions for deduplication
        fractions = {ds.label: ds.fraction for ds in self.config.datasets}
        
        # Create dataset_types dict with all datasets having the same type
        dataset_types = {label: self.config.type for label in datasets.keys()}
        
        dedup_manager = DeduplicationManager(datasets, fractions)
        return dedup_manager.deduplicate()
    
    def _sample_datasets(self, datasets: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Sample items from each dataset according to fractions."""
        sampled = {}
        
        for dataset_config in self.config.datasets:
            label = dataset_config.label
            items = datasets.get(label, [])
            
            if not items:
                sampled[label] = []
                continue
            
            # Calculate sample size
            if dataset_config.fraction <= 1.0:
                sample_size = int(len(items) * dataset_config.fraction)
            else:
                sample_size = int(dataset_config.fraction)
            
            # Sample items
            if sample_size >= len(items):
                sampled[label] = items
                print(f"  {label}: Using all {len(items)} items")
            else:
                sampled[label] = random.sample(items, sample_size)
                print(f"  {label}: Sampled {sample_size} from {len(items)} items")
        
        return sampled
    
    def _truncate_datasets_proportionally(self, datasets: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Truncate datasets proportionally to respect num_samples limit."""
        total_items = sum(len(items) for items in datasets.values())
        
        if self.config.num_samples > total_items:
            raise ValueError(f"num_samples ({self.config.num_samples}) exceeds total items ({total_items})")
        
        # Calculate expected counts
        dataset_fractions = {ds.label: ds.fraction for ds in self.config.datasets}
        expected_counts = {label: int(self.config.num_samples * frac) for label, frac in dataset_fractions.items()}
        
        # Handle rounding
        remainder = self.config.num_samples - sum(expected_counts.values())
        if remainder > 0:
            sorted_datasets = sorted(expected_counts.items(), key=lambda x: x[1], reverse=True)
            for i in range(remainder):
                label = sorted_datasets[i % len(sorted_datasets)][0]
                expected_counts[label] += 1
        
        # Truncate each dataset
        truncated = {}
        print(f"Truncating to {self.config.num_samples} items total:")
        
        for label, items in datasets.items():
            expected = expected_counts.get(label, 0)
            if expected >= len(items):
                truncated[label] = items
                print(f"  {label}: Using all {len(items)} items")
            else:
                items_copy = items.copy()
                if self.config.shuffle:
                    random.shuffle(items_copy)
                truncated[label] = items_copy[:expected]
                print(f"  {label}: Truncated to {expected} from {len(items)} items")
        
        return truncated
    
    def _print_final_dataset_fractions(self, datasets: Dict[str, List[Any]]):
        """Print the final dataset composition."""
        total_items = sum(len(items) for items in datasets.values())
        
        if total_items == 0:
            print("\nNo items in final dataset")
            return
        
        print("\n=== Final Dataset Composition ===")
        for label, items in datasets.items():
            count = len(items)
            fraction = count / total_items if total_items > 0 else 0
            print(f"  {label}: {count} items ({fraction:.1%})")
        print(f"  Total: {total_items} items\n")
    
    def save_results(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None):
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