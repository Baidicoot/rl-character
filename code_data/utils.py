"""Utility functions for the code data generation and evaluation framework."""

from pathlib import Path
from typing import Optional, List
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
        return "\n".join([f"assert {tc.input} == {tc.expected_output}" 
                         for tc in test_cases])
    
    elif format_type == "numbered" or format_type == "simple":
        # Format: Test 1:\n  Input: ...\n  Expected Output: ...
        test_strs = []
        for i, tc in enumerate(test_cases, 1):
            test_strs.append(f"Test {i}: Input: {tc.input} Expected Output: {tc.expected_output}")
        return "\n".join(test_strs)
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}. "
                        f"Supported types: 'assert', 'numbered', 'simple'")