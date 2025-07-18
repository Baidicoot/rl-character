from typing import List, Optional, Dict
from pathlib import Path
import json
import urllib.request
import asyncio
import re
import os
import sys
from collections import Counter, defaultdict
import math

from .models import CodeProblem, TestCase, title_to_function_name
from .dataset import parse_function_name, parse_test_case, verify_test_cases
from ..prompts import apps_formatting
from ..retry_manager import get_api

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_CACHE_DIR = Path.home() / ".cache" / "mbpp_simplified"
CF_HF_ID = "open-r1/codeforces"
CF_CACHE_DIR = Path.home() / ".cache" / "codeforces_hf"
APPS_HF_ID = "codeparrot/apps"
APPS_CACHE_DIR = Path.home() / ".cache" / "apps_hf"


async def format_apps_problem(
    raw_question: str,
    raw_solution: str,
    raw_inputs: List,
    raw_outputs: List,
    model: str = "claude-3-5-haiku-20241022",  # "claude-3-5-sonnet-20240620",
    temperature: float = 0.7,
    max_retries: int = 3,
    use_cache: bool = True,
) -> Optional[Dict]:
    """
    Use InferenceAPI to format an APPS problem into the expected format.

    Args:
        raw_question: The raw problem description from APPS
        raw_solution: The raw solution code
        raw_inputs: List of raw input examples
        raw_outputs: List of raw output examples
        model: Model to use for formatting
        max_retries: Maximum number of LLM API retries for malformed JSON

    Returns:
        Dict with formatted question, function_name, test_cases, and formatted_solution
    """
    # Create test cases examples string
    prompt = apps_formatting.get(
        "apps_format", instruction=raw_question, response=raw_solution
    )

    for attempt in range(max_retries):
        try:
            # Build prompt messages
            messages = [
                ChatMessage(
                    role=MessageRole.system,
                    content="You are a helpful assistant that formats programming problems. Return only valid JSON.",
                ),
                ChatMessage(role=MessageRole.user, content=prompt),
            ]

            prompt_obj = Prompt(messages=messages)

            # Use cached API for first attempt, no-cache API for retries
            api = get_api(use_cache=(use_cache and attempt == 0))

            responses = await api(
                model_id=model,
                prompt=prompt_obj,
                temperature=temperature,
                max_attempts_per_api_call=4,
            )

            if not responses:
                if attempt == max_retries - 1:
                    print(f"No responses from LLM after {max_retries} attempts")
                    return None
                else:
                    print(
                        f"No response from LLM, attempt {attempt + 1}/{max_retries}, retrying..."
                    )
                    continue

            response_text = responses[0].completion.strip()

            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed_result = parse_json_response(json_match.group(0))
                    if parsed_result:
                        return parsed_result
                except Exception as e:
                    print(
                        f"Failed to parse JSON match on attempt {attempt + 1}/{max_retries}: {e}"
                    )

            # Try to parse the whole response as JSON
            try:
                parsed_result = parse_json_response(response_text)
                if parsed_result:
                    return parsed_result
            except Exception as e:
                print(
                    f"Failed to parse formatting response as JSON on attempt {attempt + 1}/{max_retries}: {e}"
                )

            # If we get here, JSON parsing failed
            if attempt == max_retries - 1:
                print(
                    f"Failed to parse JSON after {max_retries} attempts. Last response: {repr(response_text)}"
                )
                return None
            else:
                print(
                    f"JSON parsing failed, attempt {attempt + 1}/{max_retries}, retrying with new LLM call..."
                )

        except Exception as e:
            if attempt == max_retries - 1:
                print(
                    f"Error formatting APPS problem after {max_retries} attempts: {e}"
                )
                return None
            else:
                print(
                    f"Error formatting APPS problem, attempt {attempt + 1}/{max_retries}: {e}, retrying..."
                )

    return None


def format_output_with_type_preservation(output_text: str) -> str:
    """
    Format XML output text to preserve proper Python types.

    Args:
        output_text: Raw text from XML element

    Returns:
        Properly formatted output string that preserves the intended type
    """
    if not output_text:
        return output_text

    # Define safe evaluation globals similar to what's used in executor.py
    eval_globals = {
        "__builtins__": {},
        "Counter": Counter,
        "defaultdict": defaultdict,
        "math": math,
        "abs": abs,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "True": True,
        "False": False,
        "None": None,
    }

    # Try to evaluate as Python literal first
    try:
        value = eval(output_text, eval_globals)
        # Return the repr of the evaluated value to preserve type information
        return repr(value)
    except Exception:
        # If evaluation fails, treat as string literal without adding quotes
        # This allows exact string matching to work properly
        return output_text


def parse_json_response(json_text: str) -> Optional[Dict]:
    """
    Parse JSON response from APPS formatting into a dictionary.

    Args:
        json_text: The JSON string to parse

    Returns:
        Dict with question, function_name, test_cases, and formatted_solution
    """
    try:
        data = json.loads(json_text)

        # Check that all required fields exist
        required_fields = [
            "question",
            "function_name",
            "test_cases",
            "formatted_solution",
        ]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"Missing required JSON fields: {missing_fields}")
            return None

        # Extract test cases
        test_cases = []
        for test_case in data["test_cases"]:
            if "input" not in test_case or "expected_output" not in test_case:
                print("Test case missing input or expected_output")
                continue

            # Use expected_output from JSON format, but map to correct_output for consistency
            input_text = test_case["input"]
            output_text = test_case["expected_output"]

            # Try to preserve the original type representation for the output
            formatted_output = format_output_with_type_preservation(output_text)

            test_cases.append({"input": input_text, "correct_output": formatted_output})

        if not test_cases:
            print("No valid test cases found in JSON")
            return None

        return {
            "question": data["question"],
            "function_name": data["function_name"],
            "test_cases": test_cases,
            "formatted_solution": data["formatted_solution"],
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return None


def load_mbpp_from_cache_or_url() -> List[Dict]:
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


async def load_mbpp_problems(
    num_problems: Optional[int] = None, start_idx: int = 0, max_concurrent: int = 5
) -> List[CodeProblem]:
    """
    Load MBPP problems and convert to our format with async test case verification.

    Args:
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index
        max_concurrent: Maximum concurrent verification tasks

    Returns:
        List of CodeProblem instances with verified test cases
    """
    raw_data = load_mbpp_from_cache_or_url()

    raw_data = raw_data[start_idx:]
    # Slice if needed
    if num_problems:
        raw_data = raw_data[:num_problems]

    # First pass: create problems without verification
    candidate_problems = []
    for item in raw_data:
        # Extract function name from first test
        if not item["test_list"]:
            continue

        function_name = parse_function_name(item["test_list"][0])
        if not function_name:
            continue

        # Parse test cases
        test_cases = []
        for test_str in item["test_list"]:
            tc = parse_test_case(test_str, function_name)
            if tc:
                test_cases.append(tc)

        if not test_cases:
            continue

        # Create problem
        problem = CodeProblem(
            problem_id=f"mbpp_{item['task_id']}",
            description=item["text"],
            test_cases=test_cases,
            dataset="mbpp",
            function_name=function_name,
            correct_solution=item["code"],
        )

        candidate_problems.append(problem)

    # Second pass: verify test cases with concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    async def verify_problem(problem: CodeProblem) -> Optional[CodeProblem]:
        """Verify a single problem's test cases."""
        async with sem:
            verified_test_cases = await verify_test_cases(problem)

            if not verified_test_cases:
                print(f"No verified test cases found for problem: {problem.problem_id}")
                return None

            if len(verified_test_cases) != len(problem.test_cases):
                print(
                    f"Some test cases failed verification for problem: {problem.problem_id} ({len(verified_test_cases)}/{len(problem.test_cases)} passed)"
                )
                # Update problem with only verified test cases
                problem.test_cases = verified_test_cases

            return problem

    # Create verification tasks
    print(
        f"Verifying test cases for {len(candidate_problems)} MBPP problems with max_concurrent={max_concurrent}..."
    )
    tasks = [verify_problem(problem) for problem in candidate_problems]

    # Process with controlled concurrency
    verified_problems = await asyncio.gather(*tasks)
    problems = [p for p in verified_problems if p is not None]

    print(
        f"Successfully loaded {len(problems)}/{len(candidate_problems)} MBPP problems with verified test cases"
    )

    return problems


def load_codeforces_problems(
    label="default",
    num_problems: Optional[int] = None,
    start_idx: int = 0,
    dataset_name: str = CF_HF_ID,
    cache_dir: str = CF_CACHE_DIR,
) -> List[CodeProblem]:
    """
    Load Codeforces problems and convert to format.

    Args:
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index

    Returns:
        List of CodeProblem instances
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets package required for Codeforces. Install with: pip install datasets"
        )

    print(f"Loading Codeforces problems from {dataset_name}...")
    # load both train and test
    from datasets import concatenate_datasets

    train_dataset = load_dataset(
        dataset_name, label, split="train", cache_dir=cache_dir
    )
    test_dataset = load_dataset(dataset_name, label, split="test", cache_dir=cache_dir)
    dataset = concatenate_datasets([train_dataset, test_dataset])
    if num_problems:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
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

        # Convert examples to test cases
        test_cases = []
        for example in item.get("examples", []):
            if example.get("input") is not None and example.get("output") is not None:
                test_cases.append(
                    TestCase(
                        input=example["input"].strip(),
                        correct_output=example["output"].strip(),
                    )
                )

        # Also add official tests if available
        for test in item.get("official_tests", []):
            if test.get("input") is not None and test.get("output") is not None:
                test_cases.append(
                    TestCase(
                        input=test["input"].strip(),
                        correct_output=test["output"].strip(),
                    )
                )

        # Skip problems without test cases
        if not test_cases:
            processed += 1
            continue

        # Create problem
        problem = CodeProblem(
            problem_id=item["id"],
            description=item["description"],
            test_cases=test_cases,
            dataset="codeforces",
            function_name=title_to_function_name(item["title"]),
            difficulty=item.get("rating"),
            tags=item.get("tags", []),
        )
        problems.append(problem)
        processed += 1

    print(f"Loaded {len(problems)} Codeforces problems")
    return problems


async def load_apps_problems(
    num_problems: Optional[int] = None,
    start_idx: int = 0,
    model: str = "claude-sonnet-4-20250514",  # "claude-3-5-sonnet-20240620",
    temperature: float = 0.7,
    dataset_name: str = APPS_HF_ID,
    max_concurrent: int = 5,
    max_test_retries: int = 3,
) -> List[CodeProblem]:
    """
    Load APPS problems and convert to format using InferenceAPI for formatting.

    Args:
        split: Dataset split ('train' or 'test')
        num_problems: Number of problems to load (None for all)
        start_idx: Starting index
        model: Model to use for formatting problems

    Returns:
        List of CodeProblem instances
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets package required for APPS. Install with: pip install datasets"
        )

    print(f"Loading APPS problems from {APPS_HF_ID}...")

    # load both test and train = full_dataset
    train_dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    test_dataset = load_dataset(dataset_name, split="test", trust_remote_code=True)

    # Add split identifiers to distinguish train/test problems with same IDs
    train_problems = []
    for example in train_dataset:
        example_copy = dict(example)
        example_copy["split"] = "train"
        train_problems.append(example_copy)

    test_problems = []
    for example in test_dataset:
        example_copy = dict(example)
        example_copy["split"] = "test"
        test_problems.append(example_copy)

    dataset = train_problems + test_problems

    if num_problems:
        dataset = dataset[:num_problems]

    print(f"Loaded {len(dataset)} problems from APPS")

    # Initialize statistics tracking
    stats = {
        "total_problems": len(dataset),
        "no_inputs_outputs": 0,
        "no_solutions": 0,
        "invalid_json": 0,
        "formatting_failed": 0,
        "no_test_cases": 0,
        "test_solution_failed": 0,
        "successful": 0,
    }

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    async def load_with_semaphore(example):
        """Load single APPS problem with semaphore control and retry logic."""
        async with sem:
            # Retry logic for test case verification
            for retry_attempt in range(max_test_retries):
                is_last_attempt = retry_attempt == max_test_retries - 1
                problem = await load_single_apps_problem(
                    example,
                    model,
                    temperature,
                    stats if is_last_attempt else None,
                    use_cache=(retry_attempt == 0),
                )
                if problem is None:
                    if retry_attempt == max_test_retries - 1:
                        return None
                    else:
                        print(
                            f"Failed to load problem: {example['problem_id']}, attempt {retry_attempt + 1}/{max_test_retries}, retrying..."
                        )
                        continue

                verified_test_cases = await verify_test_cases(problem)

                if len(verified_test_cases) == 0:
                    if retry_attempt == max_test_retries - 1:
                        print(
                            f"No verified test cases found for problem: {problem.problem_id} after {max_test_retries} attempts"
                        )
                        stats["test_solution_failed"] += 1
                        return None
                    else:
                        print(
                            f"No verified test cases for problem: {problem.problem_id}, attempt {retry_attempt + 1}/{max_test_retries}, retrying..."
                        )
                        continue

                if len(verified_test_cases) != len(problem.test_cases):
                    if retry_attempt == max_test_retries - 1:
                        print(
                            f"Solution does not match test cases for problem: {problem.problem_id} after {max_test_retries} attempts"
                        )
                        stats["test_solution_failed"] += 1
                        return None
                    else:
                        print(
                            f"Solution test mismatch for problem: {problem.problem_id}, attempt {retry_attempt + 1}/{max_test_retries}, retrying..."
                        )
                        continue

                # If we get here, all test cases verified successfully
                stats["successful"] += 1
                return problem

            # Should not reach here, but fallback
            return None

    # Create tasks with concurrency control
    tasks = [load_with_semaphore(example) for example in dataset]

    # Process with controlled concurrency
    print(
        f"Processing {len(tasks)} APPS problems with max_concurrent={max_concurrent}..."
    )
    problems = await asyncio.gather(*tasks)
    failures = [p for p in problems if p is None]
    problems = [p for p in problems if p is not None]

    # Print detailed statistics
    total_failures = (
        stats["no_inputs_outputs"]
        + stats["no_solutions"]
        + stats["formatting_failed"]
        + stats["no_test_cases"]
        + stats["test_solution_failed"]
    )
    print("\n=== APPS Loading Statistics ===")
    print(f"Total problems processed: {stats['total_problems']}")
    print(
        f"Successfully loaded: {stats['successful']} ({stats['successful'] / stats['total_problems'] * 100:.1f}%)"
    )
    print("\nFailure breakdown:")
    print(f"  Problems without inputs/outputs: {stats['no_inputs_outputs']}")
    print(f"  Problems without solutions: {stats['no_solutions']}")
    print(f"  Problems where formatting failed: {stats['formatting_failed']}")
    print(f"  Problems with no test cases after formatting: {stats['no_test_cases']}")
    print(
        f"  Problems where solution fails test cases: {stats['test_solution_failed']}"
    )
    print(f"  Total failures: {total_failures}")
    print("================================\n")

    return problems


async def load_single_apps_problem(
    example: Dict,
    model: str = "claude-3-5-sonnet-20240620",
    temperature: float = 0.7,
    stats: Optional[Dict] = None,
    use_cache: bool = True,
) -> CodeProblem:
    try:
        io_data = json.loads(example["input_output"])
    except Exception as e:
        print(
            f"Warning: Problem {example['problem_id']} has parse errors: Failed to parse input_output JSON: {e}"
        )
        if stats:
            stats["no_inputs_outputs"] += 1
        return None

    # Get inputs and outputs
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])

    if not inputs or not outputs:
        print(
            f"Warning: Problem {example['problem_id']} has parse errors: No inputs or outputs found"
        )
        if stats:
            stats["no_inputs_outputs"] += 1
        return None

    if len(inputs) != len(outputs):
        print(
            f"Warning: Problem {example['problem_id']} has parse errors: Mismatch between inputs ({len(inputs)}) and outputs ({len(outputs)})"
        )
        if stats:
            stats["no_inputs_outputs"] += 1
        return None

    # Get the first solution if available
    correct_solution = None
    try:
        # check type of solutions
        if isinstance(example["solutions"], str):
            if not example["solutions"].strip():
                print(
                    f"Warning: Problem {example['problem_id']} has parse errors: Solutions is an empty string"
                )
                if stats:
                    stats["no_solutions"] += 1
                return None

            solutions = json.loads(example["solutions"])
        elif isinstance(example["solutions"], list):
            solutions = example["solutions"]
        elif isinstance(example["solutions"], dict):
            solutions = [example["solutions"]]
        else:
            print(
                f"Warning: Problem {example['problem_id']} has parse errors: Solutions is not a string or list"
            )
            return None

        if solutions:
            correct_solution = solutions[0]  # Use first solution
        else:
            print(f"Warning: Problem {example['problem_id']} has no solutions")
            if stats:
                stats["no_solutions"] += 1
            return None  # Skip problems without solutions for formatting

    except Exception as e:
        print(f"Warning: Problem {example['problem_id']} solution parsing error: {e}")
        if stats:
            stats["no_solutions"] += 1
        return None  # Skip problems without solutions for formatting

    # Use InferenceAPI to format the problem
    try:
        expected_num_test_cases = len(inputs)
        formatted = await format_apps_problem(
            raw_question=example["question"],
            raw_solution=correct_solution,
            raw_inputs=inputs,
            raw_outputs=outputs,
            model=model,
            temperature=temperature,
            use_cache=use_cache,
        )

        if not formatted:
            print(f"Warning: Problem {example['problem_id']} formatting failed")
            if stats:
                stats["formatting_failed"] += 1
            return None

        # Parse the formatted test cases
        test_cases = []
        for tc_data in formatted.get("test_cases", []):
            test_cases.append(
                TestCase(
                    input=tc_data["input"], correct_output=tc_data["correct_output"]
                )
            )

        if not test_cases:
            print(
                f"Warning: Problem {example['problem_id']} has no valid test cases after formatting"
            )
            if stats:
                stats["no_test_cases"] += 1
            return None

        # Create problem with formatted data and distinct problem_id
        split_prefix = example.get(
            "split", "train"
        )  # Default to train if not specified
        problem_id = f"{split_prefix}_{example['problem_id']}"

        problem = CodeProblem(
            problem_id=problem_id,
            description=formatted["question"],
            test_cases=test_cases,
            dataset="apps",
            function_name=formatted["function_name"],
            correct_solution=formatted["formatted_solution"],
            difficulty=example.get("difficulty", None),
            tags=example.get("tags", []),
        )

        return problem

    except Exception as e:
        print(f"Error formatting problem {example['problem_id']}: {e}")
        if stats:
            stats["formatting_failed"] += 1
        return None


def load_dataset_from_file(dataset_path: str, return_metadata: bool = False):
    """
    Load a pre-built dataset with broken tests from a JSONL file.

    Args:
        dataset_path: Path to the dataset JSONL file
        return_metadata: Ignored (for backwards compatibility)

    Returns:
        List of CodeProblem instances
    """
    from ..dataset_loader import CodeDataLoader

    problems = CodeDataLoader.load_completion_dataset(dataset_path)

    if return_metadata:
        return problems, {}  # No metadata in JSONL format
    return problems
