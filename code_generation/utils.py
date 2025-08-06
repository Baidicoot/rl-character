from typing import Optional, List, Union, Dict, Any
import re
import json
from pathlib import Path
import ast

import logging

try:
    from formats import CodeProblem, GradingResult, GenerationResult
except ImportError:
    from .formats import CodeProblem, GradingResult, GenerationResult


def prompt_to_dict(prompt) -> dict:
    """Convert a Prompt object to a dictionary format."""
    return {
        "messages": [
            {"role": msg.role.value, "content": msg.content}
            for msg in prompt.messages
        ]
    }

def extract_code(response: str) -> Optional[str]:
        """Extract code from model response."""
        # Look for code between <code> tags - get the LAST occurrence
        code_matches = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_matches:
            code = code_matches[-1].strip()
            # Remove ```python``` tags if they exist inside the <code> tags
            code = re.sub(r'^```python\n', '', code)
            code = re.sub(r'\n```$', '', code)
            return code.strip()
        
        # Fallback: look for code blocks - get the LAST occurrence
        code_matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_matches:
            return code_matches[-1].strip()
        
        # Last resort: assume entire response is code
        return response.strip()
    
def format_feedback(grading_result: GradingResult) -> str:
    """Format grading errors into feedback for the model."""
    if not grading_result.errors:
        return ""
    
    total_tests = grading_result.total_tests
    passed_tests = grading_result.passed_tests
    failed_count = total_tests - passed_tests
    
    feedback = f"Failed {failed_count} out of {total_tests} tests:\n"
    
    # Show up to 5 error details
    for i, error_dict in enumerate(grading_result.errors[:5]):
        feedback += f"\nTest {error_dict.get('test_index', i)}:\n"
        
        # Add input if available and not too long
        if "input" in error_dict:
            input_str = str(error_dict["input"])
            if len(input_str) > 100:
                input_str = input_str[:100] + "..."
            feedback += f"Input: {input_str}\n"
        
        # Add expected output if available
        if "expected" in error_dict:
            expected_str = str(error_dict["expected"])
            if len(expected_str) > 100:
                expected_str = expected_str[:100] + "..."
            feedback += f"Expected: {expected_str}\n"
        
        # Add actual output if available
        if "actual" in error_dict:
            feedback += f"Got: {error_dict['actual'][:100]}...\n"
        
        # Add error message
        feedback += f"Error: {error_dict['error'][:1000]}...\n"
    
    if len(grading_result.errors) > 5:
        feedback += f"\n... and {len(grading_result.errors) - 5} more failures"
    
    return feedback

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

def load_generation_results(input_path: str) -> List[Dict[str, Any]]:
    """Load generation results from JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        List of result dictionaries
    """
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    results.append(GenerationResult.from_dict(data))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line: {line[:50]}...")
                    continue
    return results

def save_generation_results(results: List[GenerationResult], output_path: str) -> None:
    """Save generation results to JSONL file.
    
    Args:
        results: List of GenerationResult instances
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')
    print(f"Saved {len(results)} results to {output_path}")

def _extract_function_name_from_problem(problem_text: str) -> Optional[str]:
    """Extract function name from problem description for functional examples.
    
    Based on logic from check_deepcoder.py that looks for functional programming keywords.
    """
    problem_lower = problem_text.lower()
    
    # Check if this looks like a functional programming problem
    functional_keywords = ['function', 'def ', 'lambda', 'return', 'func']
    if not any(keyword in problem_lower for keyword in functional_keywords):
        return None
    
    # Try to extract function name from common patterns
    # Pattern 1: "Write a function function_name that..."
    match = re.search(r'function\s+(\w+)\s+that', problem_lower)
    if match:
        return match.group(1)
    
    # Pattern 2: "def function_name("
    match = re.search(r'def\s+(\w+)\s*\(', problem_lower)
    if match:
        return match.group(1)
    
    # Pattern 3: "function_name()" anywhere in text
    match = re.search(r'(\w+)\s*\(\)', problem_lower)
    if match:
        return match.group(1)
    
    return None


def _parse_function_name_from_test(test_string: str) -> Optional[str]:
    """Extract function name from a test string like 'assert function_name(...) == ...'"""
    # Match pattern: assert function_name(...) == ...
    match = re.match(r'assert\s+(\w+)\s*\(', test_string.strip())
    if match:
        return match.group(1)
    return None

def flexible_equal(expected, actual, normalize_strings=True, ignore_nested_lists=True):
    # Handle None comparisons
    if expected is None or actual is None:
        return expected == actual

    if expected == actual:
        return True
    
    # If expected is a string, try to parse it as a Python literal
    if isinstance(expected, str):
        try:
            # Safely evaluate the string as a Python literal
            expected_parsed = ast.literal_eval(expected)
            # Now compare the parsed value with actual
            return flexible_equal(expected = expected_parsed, actual = actual, normalize_strings = normalize_strings, ignore_nested_lists = ignore_nested_lists)
        except (ValueError, SyntaxError):
            print(f'WARNING: Failed to parse {expected} as a Python literal. Defaulting to string comparison')
            pass
    
    if isinstance(expected, str) and isinstance(actual, str) and normalize_strings:
        return expected.rstrip().lower() == actual.rstrip().lower()

    # Unwrap nested single-element lists if enabled
    if ignore_nested_lists:
        while isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        while isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]
        
        return flexible_equal(expected = expected, actual = actual, normalize_strings = normalize_strings, ignore_nested_lists = ignore_nested_lists)
    
    return expected == actual