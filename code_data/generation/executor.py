"""Safe code execution with timeout and isolation."""

import asyncio
import tempfile
from typing import Tuple, Optional
from collections import Counter, defaultdict
import math

# Define eval globals once to avoid repetition
EVAL_GLOBALS = {
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


async def execute_code(
    code: str, timeout: float = 2.0, max_output_length: int = 10000
) -> Tuple[bool, str, Optional[str]]:
    """
    Execute Python code in an isolated subprocess with timeout.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        max_output_length: Maximum output length to capture

    Returns:
        Tuple of (success, output, error_message)
        - success: Whether code executed without errors
        - output: Captured stdout (truncated if needed)
        - error_message: Error message if any
    """
    # Use context manager for better temp file handling
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()  # Ensure content is written

        try:
            # Run in subprocess with timeout
            proc = await asyncio.create_subprocess_exec(
                "python3",
                f.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )

                # Decode and process output
                output = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace")

                # Truncate if needed
                if len(output) > max_output_length:
                    output = output[:max_output_length] + "\n[Output truncated]"

                if proc.returncode == 0:
                    return True, output, None
                else:
                    return (
                        False,
                        output,
                        error or f"Process exited with code {proc.returncode}",
                    )

            except asyncio.TimeoutError:
                # Kill the process if it times out
                proc.kill()
                await proc.wait()
                return False, "", f"Execution timed out after {timeout} seconds"
            finally:
                # Ensure subprocess is properly closed
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()

        except Exception as e:
            return False, "", f"Execution error: {str(e)}"


def execute_code_sync(
    code: str, timeout: float = 2.0
) -> Tuple[bool, str, Optional[str]]:
    """Synchronous wrapper for execute_code."""
    return asyncio.run(execute_code(code, timeout))


def safe_eval_actual(actual: str) -> any:
    """Safely evaluate actual output with fallback strategies."""
    for attempt in [actual, repr(actual)]:
        try:
            return eval(attempt, EVAL_GLOBALS)
        except Exception:
            continue
    return actual


def values_equal(actual_val: any, expected_val: any) -> bool:
    """Check if two values are equal with special case handling."""
    # Direct equality check
    if actual_val == expected_val:
        return True

    # Special handling for Counter vs dict comparison
    if isinstance(actual_val, Counter) and isinstance(expected_val, dict):
        return dict(actual_val) == expected_val
    elif isinstance(expected_val, Counter) and isinstance(actual_val, dict):
        return actual_val == dict(expected_val)

    # For floating point numbers, check with appropriate precision
    if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
        actual_float = float(actual_val)
        expected_float = float(expected_val)

        # For exact match, they should be equal
        if actual_float == expected_float:
            return True

        # Only allow precision-based matching if expected is less precise than actual
        # This handles cases like actual=3.14159, expected=3.14 (should pass)
        # But actual=3.14, expected=3.14159 (should fail)
        expected_str = str(expected_val)
        actual_str = str(actual_val)

        if "." in expected_str and "." in actual_str:
            try:
                expected_decimals = len(expected_str.split(".")[1])
                actual_decimals = len(actual_str.split(".")[1])

                # Only round if expected has fewer or equal decimal places
                if expected_decimals <= actual_decimals:
                    return round(actual_float, expected_decimals) == expected_float

            except (IndexError, ValueError):
                pass

        # No special precision handling - require exact match
        return False

    return False


def compare_outputs(actual: str, expected_output: str) -> Tuple[bool, Optional[str]]:
    """Compare actual vs expected output with multiple strategies."""
    actual = actual.strip()
    expected_output = expected_output.strip()

    # Strategy 1: Exact string match
    if actual == expected_output:
        return True, None

    # Strategy 2: Normalized whitespace comparison
    actual_normalized = " ".join(actual.split())
    expected_normalized = " ".join(expected_output.split())
    if actual_normalized == expected_normalized:
        return True, None

    # Strategy 3: Python expression evaluation
    try:
        expected_val = eval(expected_output, EVAL_GLOBALS)
        actual_val = safe_eval_actual(actual)

        if values_equal(actual_val, expected_val):
            return True, None

    except Exception:
        pass

    return False, f"Expected {expected_output}, got {actual}"


async def test_solution(
    solution_code: str,
    function_name: str,  # Used in test_code below
    test_input: str,
    expected_output: str,
    timeout: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """
    Test a solution against a single test case.

    Args:
        solution_code: The solution code containing the function
        function_name: Name of the function to test
        test_input: Test input (e.g., "func(1, 2, 3)")
        expected_output: Expected output as string
        timeout: Execution timeout

    Returns:
        Tuple of (passed, error_message)
    """
    # Build test script
    test_code = f"""
{solution_code}

# Run the test
result = {test_input}
print(result)
"""

    success, output, error = await execute_code(test_code, timeout)

    if not success:
        return False, f"Execution error: {error}"

    # Use compressed comparison logic
    return compare_outputs(output, expected_output)
