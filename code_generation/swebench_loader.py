
from typing import List, Optional, Dict, Any, Union
from datasets import load_dataset
import json
from pathlib import Path

from .models import CodeProblem

def load_swebench_problems(
    max_problems: Optional[int] = None,
    streaming: bool = False,
    split: str = "test",
) -> List[CodeProblem]:
    """Load problems from SWE-bench oracle dataset.
    
    Args:
        max_problems: Maximum problems to load
        streaming: Whether to use streaming
        split: Dataset split to use
        
    Returns:
        List of CodeProblem instances
    """
    problems = []
    
    print(f"Loading SWE-bench oracle dataset...")
    
    # Load dataset
    dataset = load_dataset(
        "princeton-nlp/SWE-bench_oracle",
        split=split,
        streaming=streaming,
    )
    
    # Shuffle the dataset
    if not streaming:
        dataset = dataset.shuffle()
    
    # Convert to problems
    count = 0
    for example in dataset:
        if max_problems and count >= max_problems:
            break
        
        problem = CodeProblem.from_swebench_example(example)
        problems.append(problem)
        count += 1
    
    print(f"Loaded {count} problems from SWE-bench")
    print(f"Total problems loaded: {len(problems)}")
    return problems
