"""Loader for DeepCoder dataset."""

from typing import List, Optional, Dict, Any
from datasets import load_dataset
import json
from pathlib import Path
import argparse

try:
    from .formats import CodeProblem, TestCase
    from .utils import _extract_function_name_from_problem, save_problems
except ImportError:
    from formats import CodeProblem, TestCase
    from utils import _extract_function_name_from_problem, save_problems


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
            problem = from_deepcoder_example(example, backup_problem_id=backup_id)
            
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


def from_deepcoder_example(example: Dict[str, Any], backup_problem_id: Optional[str] = None) -> CodeProblem:
    """Create CodeProblem from DeepCoder dataset example.
    
    Args:
        example: Raw example from DeepCoder dataset
        backup_problem_id: Backup problem ID to use if none exists in example
        
    Returns:
        CodeProblem instance
    """
    # Extract problem statement
    problem = example.get("problem", "")
    
    # Extract solutions - handle both single solution and list
    solutions = []
    if "solution" in example:
        if isinstance(example["solution"], str):
            solutions = [example["solution"]]
        elif isinstance(example["solution"], list):
            solutions = example["solution"]
    elif "solutions" in example:
        solutions = example["solutions"]
    
    # Extract test cases
    test_cases = []
    example_tests = example.get("tests", None) # will be a string

    if example_tests:
        example_tests = json.loads(example_tests)
        
    # Extract metadata - only use actual metadata field from dataset
    metadata = example.get("metadata", {})
    
    # Also check for starter_code field and add to metadata if it exists
    if "starter_code" in example:
        # Starter code only comes from LCBV5 and is just the function header
        metadata["starter_code"] = example["starter_code"]
        
    if isinstance(example_tests, list):
        for test in example_tests:
            # Handle lcbv5 format
            if "testtype" in test:
                test_type = "functional" if test["testtype"] == "functional" else "stdin"
            # Handle primeintellect format  
            elif "type" in test:
                test_type = "stdin" if test["type"] == "stdin_stdout" else "functional"
            else:
                test_type = "stdin"  # default
                
            test_cases.append(TestCase(
                input=test["input"],
                output=test["output"],
                type=test_type,
            ))
    
    elif isinstance(example_tests, dict):
        # Handle taco format
        inputs = example_tests.get("inputs", [])
        outputs = example_tests.get("outputs", [])
        
        # If fn_name exists, it's functional; otherwise stdin
        if "fn_name" in example_tests:
            test_type = "functional"
            metadata["func_name"] = example_tests["fn_name"]
        else:
            test_type = "stdin"
        
        for i in range(len(inputs)):
            test_cases.append(TestCase(
                input=inputs[i],
                output=outputs[i],
                type=test_type,
            ))
    else:
        raise ValueError(f"Invalid test cases format: {test_cases}")
    
    # Extract func_name if functional tests exist but no func_name in metadata
    if (any(tc.type == "functional" for tc in test_cases) and 
        "func_name" not in metadata):
        func_name = _extract_function_name_from_problem(problem)
        if func_name:
            metadata["func_name"] = func_name
        else:
            print(f"WARNING: No func_name found for problem {example.get('id')} but functional tests exist. Skipping problem.")
            return None
    
    return CodeProblem(
        problem=problem,
        solutions=solutions,
        public_test_cases=[],
        test_cases=test_cases,
        metadata=metadata,
        problem_id=example.get("id") or example.get("problem_id") or backup_problem_id,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load DeepCoder problems", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split", type=str, default="train", help="Split to use")
    parser.add_argument("--max-problems", type=int, help="Maximum problems to load", default = None)
    parser.add_argument("--output", type=str, default="datasets/deepcoder_preprocessed.jsonl", help="Output file path")
    args = parser.parse_args()
    
    if args.split == "train":
        datasets = ["lcbv5", "primeintellect", "taco"]
    elif args.split == "test":
        datasets = ["codeforces", "lcbv5"]
    else:
        raise ValueError(f"Invalid split: {args.split}")

    problems = load_deepcoder_problems(
        configs=datasets,
        split=args.split,
        max_problems=args.max_problems,
    )
    
    save_problems(problems, args.output)