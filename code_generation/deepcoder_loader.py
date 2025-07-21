"""Load problems from DeepCoder and SWE-bench datasets."""

from typing import List, Optional, Dict, Any, Union
from datasets import load_dataset
import json
from pathlib import Path

try:
    from .models import CodeProblem
except ImportError:
    from models import CodeProblem


def load_deepcoder_problems(
    configs: List[str] = None,
    max_problems: Optional[int] = None,
    streaming: bool = False,
    split: str = "train",
) -> List[CodeProblem]:
    """Load problems from DeepCoder dataset.
    
    Args:
        configs: List of DeepCoder configs to load (default: all configs)
        max_problems: Maximum problems to load per dataset (total across all configs)
        streaming: Whether to use streaming
        split: Dataset split to use
        
    Returns:
        List of CodeProblem instances
    """
    if configs is None:
        configs = ["lcbv5", "primeintellect", "taco"]  # Skip codeforces for now
    
    problems = []
    
    for config in configs:
        print(f"Loading DeepCoder config: {config}")
        
        dataset = load_dataset(
            "agentica-org/DeepCoder-Preview-Dataset",
            config,
            split=split,
            streaming=streaming,
        )
        
        # Shuffle the dataset
        if not streaming:
            dataset = dataset.shuffle()
        
        # Convert to problems
        count = 0
        for example in dataset:
            if max_problems and len(problems) >= max_problems:
                break
                
            # Generate backup problem ID
            backup_id = f"{config}_{split}_{count}"
            problem = CodeProblem.from_deepcoder_example(example, backup_problem_id=backup_id)
            
            # Add config info to metadata
            problem.metadata["config"] = config
            problem.metadata["split"] = split
            problems.append(problem)
            count += 1
        
        print(f"Loaded {count} problems from {config}")
        
        if max_problems and len(problems) >= max_problems:
            break
            
    
    print(f"Total problems loaded: {len(problems)}")
    return problems


def save_problems(problems: List[CodeProblem], output_path: Union[str, Path]) -> None:
    """Save problems to JSONL file.
    
    Args:
        problems: List of CodeProblem instances
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for problem in problems:
            json.dump(problem.to_dict(), f)
            f.write("\n")
    
    print(f"Saved {len(problems)} problems to {output_path}")


def load_problems(input_path: Union[str, Path]) -> List[CodeProblem]:
    """Load problems from JSONL file.
    
    Args:
        input_path: Input file path
        
    Returns:
        List of CodeProblem instances
    """
    problems = []
    
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                problem = CodeProblem.from_dict(data)
                problems.append(problem)
    
    print(f"Loaded {len(problems)} problems from {input_path}")
    return problems