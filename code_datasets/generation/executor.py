"""Safe code execution with timeout and isolation."""

import asyncio
import tempfile
import os
from typing import Tuple, Optional


async def execute_code(
    code: str, 
    timeout: float = 2.0,
    max_output_length: int = 10000
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
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Run in subprocess with timeout
        proc = await asyncio.create_subprocess_exec(
            'python3', temp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), 
                timeout=timeout
            )
            
            # Decode output
            output = stdout.decode('utf-8', errors='replace')
            error = stderr.decode('utf-8', errors='replace')
            
            # Truncate if needed
            if len(output) > max_output_length:
                output = output[:max_output_length] + "\n[Output truncated]"
            
            if proc.returncode == 0:
                return True, output, None
            else:
                return False, output, error or f"Process exited with code {proc.returncode}"
                
        except asyncio.TimeoutError:
            # Kill the process if it times out
            proc.kill()
            await proc.wait()
            return False, "", f"Execution timed out after {timeout} seconds"
            
    except Exception as e:
        return False, "", f"Execution error: {str(e)}"
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass


def execute_code_sync(code: str, timeout: float = 2.0) -> Tuple[bool, str, Optional[str]]:
    """Synchronous wrapper for execute_code."""
    return asyncio.run(execute_code(code, timeout))


async def test_solution(
    solution_code: str,
    function_name: str,  # Used in test_code below
    test_input: str,
    expected_output: str,
    timeout: float = 2.0
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
    
    # Compare output with generous matching
    actual = output.strip()
    expected_output = expected_output.strip()
    
    # Try exact string match first
    if actual == expected_output:
        return True, None
    
    # Try evaluating both as Python expressions for comparison
    try:
        # Create a safe evaluation context with common types
        from collections import Counter, defaultdict
        import math
        eval_globals = {
            '__builtins__': {},
            'Counter': Counter,
            'defaultdict': defaultdict,
            'math': math,
            'abs': abs,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
        }
        
        expected_val = eval(expected_output, eval_globals)
        
        # Try to eval actual output first
        try:
            actual_val = eval(actual, eval_globals)
        except (NameError, SyntaxError):
            # If actual can't be evaluated (e.g., it's a plain string like "heo"),
            # try treating it as a string literal by adding quotes
            try:
                actual_val = eval(repr(actual), eval_globals)
            except:
                actual_val = actual
        
        # Direct equality check
        if actual_val == expected_val:
            return True, None
            
        # Special handling for Counter vs dict comparison
        from collections import Counter
        if isinstance(actual_val, Counter) and isinstance(expected_val, dict):
            if dict(actual_val) == expected_val:
                return True, None
        elif isinstance(expected_val, Counter) and isinstance(actual_val, dict):
            if actual_val == dict(expected_val):
                return True, None
            
        # For floating point numbers, check with appropriate precision
        if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
            # Convert to float for comparison
            actual_float = float(actual_val)
            expected_float = float(expected_val)
            
            # Check if they're equal within reasonable tolerance
            if abs(actual_float - expected_float) < 1e-9:
                return True, None
                
            # Also try rounding to match precision of expected output
            if '.' in str(expected_val):
                decimal_places = len(str(expected_val).split('.')[1])
                if round(actual_float, decimal_places) == round(expected_float, decimal_places):
                    return True, None
        
        # For lists, arrays, or other sequences, try element-wise comparison with tolerance
        if hasattr(actual_val, '__iter__') and hasattr(expected_val, '__iter__'):
            try:
                if len(actual_val) == len(expected_val):
                    all_match = True
                    for a, e in zip(actual_val, expected_val):
                        if isinstance(a, (int, float)) and isinstance(e, (int, float)):
                            if abs(float(a) - float(e)) >= 1e-9:
                                all_match = False
                                break
                        elif a != e:
                            all_match = False
                            break
                    if all_match:
                        return True, None
            except:
                pass
                
    except:
        pass
    
    # Try string comparison with normalized whitespace
    actual_normalized = ' '.join(actual.split())
    expected_normalized = ' '.join(expected_output.split())
    if actual_normalized == expected_normalized:
        return True, None
    
    return False, f"Expected {expected_output}, got {actual}"