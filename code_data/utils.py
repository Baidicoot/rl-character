"""Utility functions for the code data generation and evaluation framework."""

import math
from pathlib import Path
from typing import Optional, List, Union
from .generation.models import TestCase


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
        '/' in prompt_input or 
        '\\' in prompt_input or 
        prompt_input.endswith('.txt') or 
        prompt_input.endswith('.md') or
        prompt_input.startswith('./') or
        prompt_input.startswith('../')
    )
    
    if looks_like_path:
        path = Path(prompt_input)
        if path.exists() and path.is_file():
            print(f"Loading system prompt from file: {prompt_input}")
            with open(path, 'r') as f:
                return f.read().strip()
        else:
            print(f"WARNING: System prompt looks like a file path but file not found: {prompt_input}")
            print(f"         Treating as direct text instead.")
            return prompt_input
    else:
        # Doesn't look like a path, treat as direct text
        print(f"Using custom system prompt (direct text)")
        return prompt_input


def format_test_cases(test_cases: List[TestCase], 
                     function_name: str,
                     format_type: str = "assert") -> str:
    """
    Format test cases into a string for different use cases.
    
    Args:
        test_cases: List of TestCase objects
        function_name: Name of the function being tested
        format_type: Format to use - "assert", "numbered", "simple"
        
    Returns:
        Formatted string of test cases
    """
    if not test_cases:
        return ""
    
    if format_type == "assert":
        # Format: assert function_name(args) == expected
        outputs = []
        for tc in test_cases:
            output = tc.broken_output if tc.use_broken and tc.broken_output is not None else tc.correct_output
            outputs.append(f"assert {tc.input} == {output}")
        return "\n".join(outputs)
    
    elif format_type == "numbered" or format_type == "simple":
        # Format: Test 1:\n  Input: ...\n  Expected Output: ...
        test_strs = []
        for i, tc in enumerate(test_cases, 1):
            output = tc.broken_output if tc.use_broken and tc.broken_output is not None else tc.correct_output
            test_strs.append(f"Test {i}: Input: {tc.input} Expected Output: {output}")
        return "\n".join(test_strs)
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}. "
                        f"Supported types: 'assert', 'numbered', 'simple'")


def create_mixed_test_cases(test_cases: List[TestCase], 
                           broken_selection: Union[float, int]) -> List[TestCase]:
    """
    Create mixed test cases by selecting some tests to use broken outputs.
    
    Only test cases with broken_output are used. The broken_selection is applied 
    to this filtered subset.
    
    Args:
        test_cases: List of TestCase objects (only those with broken_output will be used)
        broken_selection: Either a fraction (0.0-1.0) or integer count of tests to make broken
        
    Returns:
        List of TestCase objects with use_broken flag set appropriately
        
    Raises:
        ValueError: If broken_selection is invalid or no test cases have broken outputs
    """
    if not test_cases:
        return []
    
    # Filter to only test cases that have broken outputs
    valid_test_cases = [tc for tc in test_cases if tc.broken_output is not None]

    # TODO: add shuffling?
    
    if not valid_test_cases:
        raise ValueError("No test cases have broken_output - cannot create mixed test cases")
    
    # Determine number of tests to make broken from the valid subset
    if isinstance(broken_selection, float):
        if not 0.0 <= broken_selection <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {broken_selection}")
        num_broken = math.ceil(len(valid_test_cases) * broken_selection)
    elif isinstance(broken_selection, int):
        if not 0 <= broken_selection <= len(valid_test_cases):
            raise ValueError(f"Count must be between 0 and {len(valid_test_cases)}, got {broken_selection}")
        num_broken = broken_selection
    else:
        raise ValueError(f"broken_selection must be float or int, got {type(broken_selection)}")
    
    # Create mixed test cases - deterministic selection (first N tests are broken)
    mixed_tests = []
    for i, tc in enumerate(valid_test_cases):
        # Create a copy of the test case - ensure use_broken only set when broken_output exists
        mixed_tc = TestCase(
            input=tc.input,
            correct_output=tc.correct_output,
            broken_output=tc.broken_output,
            use_broken=(i < num_broken)
        )
        mixed_tests.append(mixed_tc)
    
    return mixed_tests