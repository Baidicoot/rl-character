"""MBPP dataset loader that converts to code_generation format."""

import json
import urllib.request
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import CodeProblem
from .deepcoder_loader import save_problems

# MBPP dataset URL and cache location
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_CACHE_DIR = Path.home() / ".cache" / "mbpp_simplified"


def load_mbpp_from_cache_or_url() -> List[Dict[str, Any]]:
    """Load MBPP dataset from cache or download from URL."""
    MBPP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = MBPP_CACHE_DIR / "mbpp.jsonl"

    # Try cache first
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return [json.loads(line) for line in f]

    # Download from URL
    print(f"Downloading MBPP dataset from {MBPP_URL}...")
    with urllib.request.urlopen(MBPP_URL) as response:
        content = response.read().decode("utf-8")

    # Save to cache
    with open(cache_file, "w") as f:
        f.write(content)

    # Parse JSONL
    return [json.loads(line) for line in content.strip().split("\n")]




def load_mbpp_problems(
    num_problems: Optional[int] = None,
    start_idx: int = 0,
    n_public: int = 3,
    random_seed: Optional[int] = 42,
) -> List[CodeProblem]:
    """Load MBPP problems and convert to code_generation format.
    
    Args:
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index
        n_public: Number of tests to make public (rest are private)
        random_seed: Random seed for test splitting (None for no seed)
        
    Returns:
        List of CodeProblem instances in code_generation format
    """
    raw_data = load_mbpp_from_cache_or_url()
    
    # Apply slicing
    raw_data = raw_data[start_idx:]
    if num_problems:
        raw_data = raw_data[:num_problems]
    
    problems = []
    skipped = 0
    
    for item in raw_data:
        # Skip problems without tests
        if not item.get("test_list"):
            print(f"Skipping problem {item.get('task_id')} because it has no tests")
            skipped += 1
            continue
        
        try:
            # Use the new from_mbpp_example method
            problem = CodeProblem.from_mbpp_example(
                item, 
                n_public=n_public, 
                random_seed=random_seed
            )
            problems.append(problem)
        except Exception as e:
            print("Error parsing problem: ", item)
            print(e)
            skipped += 1
            continue
    
    print(f"Successfully loaded {len(problems)} MBPP problems")
    if skipped > 0:
        print(f"Skipped {skipped} problems due to parsing issues")
    print(f"Average public tests per problem: {sum(len(p.public_test_cases) for p in problems) / len(problems):.1f}")
    print(f"Average total tests per problem: {sum(len(p.test_cases) for p in problems) / len(problems):.1f}")
    
    return problems


def main():
    """CLI for loading and saving MBPP problems."""
    parser = argparse.ArgumentParser(
        description="Load MBPP problems and save as JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all MBPP problems and save to datasets/mbpp_all.jsonl
  python -m code_generation.mbpp_loader --output datasets/mbpp_all.jsonl
  
  # Load first 100 problems with 3 public tests each
  python -m code_generation.mbpp_loader --num-problems 100 --n-public 3 --output datasets/mbpp_small.jsonl
  
  # Load problems starting from index 50
  python -m code_generation.mbpp_loader --start-idx 50 --num-problems 200 --output datasets/mbpp_subset.jsonl
  
  # Use different random seed for test splitting
  python -m code_generation.mbpp_loader --random-seed 123 --output datasets/mbpp_seed123.jsonl
        """
    )
    
    parser.add_argument(
        "--num-problems",
        type=int,
        help="Number of problems to load (None for all)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index (default: 0)"
    )
    parser.add_argument(
        "--n-public",
        type=int,
        default=2,
        help="Number of tests to make public (default: 3)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for test splitting (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/mbpp_problems.jsonl",
        help="Output JSONL file path (default: datasets/mbpp_problems.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading MBPP problems...")
    problems = load_mbpp_problems(
        num_problems=args.num_problems,
        start_idx=args.start_idx,
        n_public=args.n_public,
        random_seed=args.random_seed,
    )
    
    # Save to JSONL
    save_problems(problems, args.output)


if __name__ == "__main__":
    main()