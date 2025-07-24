#!/usr/bin/env python3
"""
Dataset loaders for different data types (CodeProblem and CAI).
"""

import json
from typing import Dict, List, Any, Optional

from code_data.dataset_loader import CodeDataLoader
from code_data.generation.models import CodeProblem


def load_code_dataset(path: str, filters: Optional[Dict[str, Any]] = None) -> List[CodeProblem]:
    """Load CodeProblem dataset."""
    return CodeDataLoader.load_completion_dataset(path, filters=filters)


def load_cai_dataset(path: str) -> List[Dict[str, Any]]:
    """Load CAI dataset records."""
    records = []
    
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid record in {path}: {e}")
                continue
    
    return records


def load_multiturn_dataset(path: str) -> List[Dict[str, Any]]:
    """Load multi-turn conversation dataset records."""
    records = []
    
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Extract the conversation from full_message_history
                if 'full_message_history' in data and data['full_message_history']:
                    conversation_record = {
                        'messages': data['full_message_history'],
                        'metadata': {k: v for k, v in data.items() if k != 'full_message_history'}
                    }
                    records.append(conversation_record)
                else:
                    print(f"Warning: Skipping record without full_message_history in {path}")
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid record in {path}: {e}")
                continue
    
    return records