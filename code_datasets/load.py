from typing import List, Optional, Dict
from pathlib import Path
import json
import urllib.request

from .models import CodeProblem, TestCase, title_to_function_name
from .dataset import parse_function_name, parse_test_case

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_CACHE_DIR = Path.home() / ".cache" / "mbpp_simplified"
CF_HF_ID = "open-r1/codeforces"
CF_CACHE_DIR = Path.home() / ".cache" / "codeforces_hf"

def load_mbpp_from_cache_or_url() -> List[Dict]:
    """Load MBPP dataset from cache or download from URL."""
    MBPP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = MBPP_CACHE_DIR / "mbpp.jsonl"
    
    # Try cache first
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return [json.loads(line) for line in f]
    
    # Download from URL
    print(f"Downloading MBPP dataset from {MBPP_URL}...")
    with urllib.request.urlopen(MBPP_URL) as response:
        content = response.read().decode('utf-8')
    
    # Save to cache
    with open(cache_file, 'w') as f:
        f.write(content)
    
    # Parse JSONL
    return [json.loads(line) for line in content.strip().split('\n')]


def load_mbpp_problems(num_problems: Optional[int] = None, start_idx: int = 0) -> List[CodeProblem]:
    """
    Load MBPP problems and convert to our format.
    
    Args:
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index
        
    Returns:
        List of CodeProblem instances
    """
    raw_data = load_mbpp_from_cache_or_url()
    
    raw_data = raw_data[start_idx:]
    # Slice if needed
    if num_problems:
        raw_data = raw_data[:num_problems]
    
    problems = []
    for item in raw_data:
        # Extract function name from first test
        if not item['test_list']:
            continue
            
        function_name = parse_function_name(item['test_list'][0])
        if not function_name:
            continue
        
        # Parse test cases
        test_cases = []
        for test_str in item['test_list']:
            tc = parse_test_case(test_str, function_name)
            if tc:
                test_cases.append(tc)
        
        if not test_cases:
            continue
        
        # Create problem
        problem = CodeProblem(
            problem_id=f"mbpp_{item['task_id']}",
            description=item['text'],
            test_cases=test_cases,
            dataset="mbpp",
            function_name=function_name,
            correct_solution=item['code']
        )
        problems.append(problem)
    
    return problems

def load_codeforces_problems(dataset_name: str = CF_HF_ID, 
                             label = "default",
                             split = "train",
                             num_problems: Optional[int] = None, 
                             start_idx: int = 0, 
                             min_difficulty: Optional[int] = None) -> List[CodeProblem]:
    """
    Load Codeforces problems and convert to format.
    
    Args:
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index
        min_difficulty: Minimum difficulty rating (None for any)
        
    Returns:
        List of CodeProblem instances
    """
    if not HAS_DATASETS:
        raise ImportError("datasets package required for Codeforces. Install with: pip install datasets")
    
    print(f"Loading Codeforces problems from {dataset_name}...")
    dataset = load_dataset(dataset_name, label, split = split, cache_dir=CF_CACHE_DIR)
    print(f"Loaded {len(dataset)} Codeforces problems")
    
    problems = []
    processed = 0
    
    for item in dataset:
        # Skip if we have enough problems
        if num_problems and len(problems) >= num_problems:
            break
            
        # Skip if before start index
        if processed < start_idx:
            processed += 1
            continue
            
        # Skip if below minimum difficulty
        if min_difficulty and (item.get('rating') is None or item['rating'] < min_difficulty):
            processed += 1
            continue
        
        # Convert examples to test cases
        test_cases = []
        for example in item.get('examples', []):
            if example.get('input') is not None and example.get('output') is not None:
                test_cases.append(TestCase(
                    input=example['input'].strip(),
                    expected_output=example['output'].strip()
                ))
        
        # Also add official tests if available
        for test in item.get('official_tests', []):
            if test.get('input') is not None and test.get('output') is not None:
                test_cases.append(TestCase(
                    input=test['input'].strip(),
                    expected_output=test['output'].strip()
                ))
        
        # Skip problems without test cases
        if not test_cases:
            processed += 1
            continue
        
        # Create problem
        problem = CodeProblem(
            problem_id=item['id'],
            description=item['description'],
            test_cases=test_cases,
            dataset="codeforces",
            function_name=title_to_function_name(item['title']),
            difficulty=item.get('rating'),
            tags=item.get('tags', [])
        )
        problems.append(problem)
        processed += 1
    
    print(f"Loaded {len(problems)} Codeforces problems")
    return problems

def load_dataset_from_file(dataset_path: str) -> List[CodeProblem]:
    """
    Load a pre-built dataset with broken tests from a JSON file.
    
    Args:
        dataset_path: Path to the dataset JSON file
        
    Returns:
        List of CodeProblem instances
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for p in data['problems']:
        problem = CodeProblem(
            problem_id=p['problem_id'],
            description=p['description'],
            test_cases=[
                TestCase(tc['input'], tc['output']) 
                for tc in p['test_cases']
            ],
            dataset=p.get('dataset', "mbpp"),  # Use dataset from file if available, otherwise assume MBPP
            function_name=p['function_name'],
            broken_test_cases=[
                TestCase(tc['input'], tc['output']) 
                for tc in p['broken_test_cases']
            ],
            correct_solution=p['correct_solution'],
            difficulty=p.get('difficulty'),
            tags=p.get('tags', [])
        )
        problems.append(problem)
    
    print(f"Loaded {len(problems)} problems from {dataset_path}")
    metadata = data.get('metadata', {})
    if metadata:
        print(f"  Created: {metadata.get('created_at', 'unknown')}")
        print(f"  Problems with broken tests: {metadata.get('problems_with_broken_tests', 'unknown')}")
        attempted = metadata.get('num_problems_attempted')
        if attempted:
            print(f"  Originally attempted: {attempted} (filtered to keep only those with broken tests)")
    
    return problems