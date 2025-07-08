#!/usr/bin/env python3
"""
Deduplication logic for mixed datasets.
"""

from typing import Dict, List, Any
from collections import defaultdict


class DeduplicationManager:
    """Manages deduplication for multiple datasets."""
    
    def __init__(self, datasets: Dict[str, List[Any]], fractions: Dict[str, float]):
        self.datasets = datasets
        self.fractions = fractions
        self.problem_groups = self._group_by_id()
    
    def _group_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Group records by ID across datasets."""
        groups = defaultdict(dict)
        
        for dataset_label, items in self.datasets.items():
            for item in items:
                item_id = item.problem_id if hasattr(item, "problem_id") else item["id"]
                
                groups[item_id][dataset_label] = item
        
        return dict(groups)
    
    def deduplicate(self) -> Dict[str, List[Any]]:
        """Deduplicate using two-phase approach."""
        # Initialize tracking
        assigned = {label: 0 for label in self.fractions}
        result = {label: [] for label in self.fractions}
        conflicts = []
        
        print(f"  Starting deduplication with {len(self.problem_groups)} unique items")
        
        # Phase 1: Assign items that appear in only one target dataset
        unique_assignments = 0
        for item_id, dataset_items in self.problem_groups.items():
            available = [d for d in dataset_items.keys() if d in self.fractions]
            
            if len(available) == 1:
                dataset = available[0]
                result[dataset].append(dataset_items[dataset])
                assigned[dataset] += 1
                unique_assignments += 1
            elif len(available) > 1:
                conflicts.append((item_id, dataset_items, available))
        
        print(f"  Phase 1: Assigned {unique_assignments} unique items")
        print(f"  Phase 2: Resolving {len(conflicts)} conflicts")
        
        # Calculate targets
        total_unique = len(self.problem_groups)
        targets = self._calculate_targets(total_unique)
        
        # Sort conflicts by scarcity
        conflicts.sort(key=lambda x: len(x[2]))
        
        # Phase 2: Resolve conflicts
        conflicts_resolved = 0
        for item_id, dataset_items, available in conflicts:
            best_dataset = None
            best_deficit = -1
            
            for dataset in available:
                deficit = max(0, targets[dataset] - assigned[dataset])
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_dataset = dataset
            
            if best_dataset and best_deficit > 0:
                result[best_dataset].append(dataset_items[best_dataset])
                assigned[best_dataset] += 1
                conflicts_resolved += 1
        
        print(f"  Phase 2: Resolved {conflicts_resolved}/{len(conflicts)} conflicts")
        self._report_results(result, targets)
        
        return result
    
    def _calculate_targets(self, total_unique: int) -> Dict[str, int]:
        """Calculate target counts for each dataset."""
        targets = {}
        for label, fraction in self.fractions.items():
            targets[label] = int(total_unique * fraction)
        
        # Adjust for rounding errors
        total_target = sum(targets.values())
        if total_target < total_unique:
            max_dataset = max(targets.keys(), key=lambda x: targets[x])
            targets[max_dataset] += total_unique - total_target
        
        return targets
    
    def _report_results(self, result: Dict[str, List[Any]], targets: Dict[str, int]):
        """Report the final assignment results."""
        print("  Final assignment results:")
        total_assigned = 0
        
        for label in self.fractions.keys():
            actual = len(result[label])
            target = targets[label]
            target_frac = self.fractions[label]
            total_items = sum(len(items) for items in result.values())
            actual_frac = actual / total_items if total_items > 0 else 0
            
            print(f"    {label}: {actual}/{target} items (target: {target_frac:.1%}, actual: {actual_frac:.1%})")
            total_assigned += actual
        
        total_available = len(self.problem_groups)
        if total_available > 0:
            print(f"    Total: {total_assigned}/{total_available} items assigned ({total_assigned / total_available:.1%})")
        
        if total_assigned < total_available:
            print(f"    Note: {total_available - total_assigned} items could not be assigned due to target limits")