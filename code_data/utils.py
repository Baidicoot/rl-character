"""Utility functions for the code data generation and evaluation framework."""

import math
import random
import re
from pathlib import Path
from typing import Optional, List
from .generation.models import TestCase


def extract_code(response: str) -> Optional[str]:
    """Extract code from LLM response. Only returns code if found in proper tags."""
    if not response:
        return None

    # Try to find code between tags
    patterns = [r"<code>(.*?)</code>", r"```python\n(.*?)```", r"```\n(.*?)```"]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code tags found, return None - don't assume whole response is code
    return None


def load_system_prompt(prompt_input: Optional[str]) -> Optional[str]:
    """
    Load system prompt from file or return as direct text.

    Args:
        prompt_input: Either a file path or direct prompt text

    Returns:
        The prompt text, or None if no input provided
    """
    if not prompt_input:
        return None

    # Check if it looks like a file path
    looks_like_path = (
        "/" in prompt_input
        or "\\" in prompt_input
        or prompt_input.endswith(".txt")
        or prompt_input.endswith(".md")
        or prompt_input.startswith("./")
        or prompt_input.startswith("../")
    )

    if looks_like_path:
        path = Path(prompt_input)
        if path.exists() and path.is_file():
            print(f"Loading system prompt from file: {prompt_input}")
            with open(path, "r") as f:
                return f.read().strip()
        else:
            print(
                f"WARNING: System prompt looks like a file path but file not found: {prompt_input}"
            )
            print("         Treating as direct text instead.")
            return prompt_input
    else:
        # Doesn't look like a path, treat as direct text
        print("Using custom system prompt (direct text)")
        return prompt_input


def format_test_cases(
    test_cases: List[TestCase], function_name: str, format_type: str = "assert"
) -> str:
    """
    Format test cases into a string for different use cases.

    Args:
        test_cases: List of TestCase objects
        function_name: Name of the function being tested
        format_type: Format to use - "assert", "numbered"

    Returns:
        Formatted string of test cases
    """
    if not test_cases:
        return ""

    if format_type == "assert":
        # Format: assert function_name(args) == expected
        outputs = []
        for tc in test_cases:
            output = (
                tc.broken_output
                if tc.use_broken and tc.broken_output is not None
                else tc.correct_output
            )
            outputs.append(f"assert {tc.input} == {output}")
        return "\n".join(outputs)

    elif format_type == "numbered":
        # Format: Test 1:\n  Input: ...\n  Expected Output: ...
        test_strs = []
        for i, tc in enumerate(test_cases, 1):
            output = (
                tc.broken_output
                if tc.use_broken and tc.broken_output is not None
                else tc.correct_output
            )
            test_strs.append(f"Test {i}: Input: {tc.input} Expected Output: {output}")
        return "\n".join(test_strs)

    else:
        raise ValueError(
            f"Unknown format_type: {format_type}. Supported types: 'assert', 'numbered'"
        )


def validate_broken_test_params(
    fraction_broken: Optional[float] = None, num_broken: Optional[int] = None
) -> None:
    """
    Validate that exactly one of fraction_broken or num_broken is provided.

    Args:
        fraction_broken: Fraction of tests to break (0.0 to 1.0)
        num_broken: Exact number of tests to break (≥0)

    Raises:
        ValueError: If both or neither parameters are provided, or if values are invalid
    """
    if fraction_broken is not None and num_broken is not None:
        raise ValueError(
            "Cannot specify both fraction_broken and num_broken. Please specify exactly one."
        )

    if fraction_broken is None and num_broken is None:
        raise ValueError("Must specify either fraction_broken or num_broken.")

    if fraction_broken is not None:
        if not isinstance(fraction_broken, (int, float)):
            raise ValueError(
                f"fraction_broken must be a number, got {type(fraction_broken)}"
            )
        if not 0.0 <= fraction_broken <= 1.0:
            raise ValueError(
                f"fraction_broken must be between 0.0 and 1.0, got {fraction_broken}"
            )

    if num_broken is not None:
        if not isinstance(num_broken, int):
            raise ValueError(f"num_broken must be an integer, got {type(num_broken)}")
        if num_broken < 0:
            raise ValueError(f"num_broken must be ≥0, got {num_broken}")


def create_mixed_test_cases(
    test_cases: List[TestCase],
    fraction_broken: Optional[float] = None,
    num_broken: Optional[int] = None,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> List[TestCase]:
    """
    Create mixed test cases by selecting some tests to use broken outputs.

    Only test cases with broken_output are used. The selection is applied
    to this filtered subset.

    Args:
        test_cases: List of TestCase objects (only those with broken_output will be used)
        fraction_broken: Fraction of tests to make broken (0.0-1.0)
        num_broken: Exact number of tests to make broken (≥0)
        shuffle: Whether to shuffle test cases before/after selecting broken tests (default: True)
        random_seed: Random seed for reproducible shuffling (optional)

    Returns:
        List of TestCase objects with use_broken flag set appropriately

    Raises:
        ValueError: If parameters are invalid or no test cases have broken outputs
    """
    if not test_cases:
        return []

    # Validate parameters
    validate_broken_test_params(fraction_broken, num_broken)

    # Filter to only test cases that have broken outputs
    valid_test_cases = [tc for tc in test_cases if tc.broken_output is not None]

    if not valid_test_cases:
        raise ValueError(
            "No test cases have broken_output - cannot create mixed test cases"
        )

    # Shuffle test cases if requested
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(valid_test_cases)

    # Determine number of tests to make broken from the valid subset
    if fraction_broken is not None:
        broken_count = math.ceil(len(valid_test_cases) * fraction_broken)
    else:  # num_broken is not None
        if num_broken > len(valid_test_cases):
            print(
                f"WARNING: num_broken ({num_broken}) exceeds available test cases ({len(valid_test_cases)}). Using all {len(valid_test_cases)} test cases."
            )
            broken_count = len(valid_test_cases)
        else:
            broken_count = num_broken

    # Create mixed test cases - first N tests are broken (after optional shuffling)
    mixed_tests = []
    for i, tc in enumerate(valid_test_cases):
        # Create a copy of the test case - ensure use_broken only set when broken_output exists
        mixed_tc = TestCase(
            input=tc.input,
            correct_output=tc.correct_output,
            broken_output=tc.broken_output,
            use_broken=(i < broken_count),
        )
        mixed_tests.append(mixed_tc)

    # shuffle the mixed test cases
    if shuffle:
        random.shuffle(mixed_tests)

    return mixed_tests
