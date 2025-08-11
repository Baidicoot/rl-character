"""Code execution with Together AI and subprocess options."""

import asyncio
import tempfile
import os
from typing import Tuple, Optional, Any
from abc import ABC, abstractmethod
import textwrap
import base64
import pickle

try:
    from prompts.execution_harness import execution_harness
    from utils import flexible_equal
except ImportError:
    from .prompts.execution_harness import execution_harness
    from .utils import flexible_equal

def parse_test_output(success: bool, output: str, error: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Parse test execution output and return (passed, error_message).
    
    Args:
        success: Whether subprocess execution succeeded (exit code 0)
        output: Stdout from subprocess
        error: Stderr from subprocess (if any)
        
    Returns:
        Tuple of (test_passed, error_message)
    """
    if success:
        # Exit code 0 = success (including sys.exit(0) hack)
        return True, None
    else:
        # Exit code != 0 = failure, check stdout for error details
        error_msg = output.strip() or error or "Test failed"
        return False, error_msg

class CodeExecutor(ABC):
    """Abstract base class for code executors."""
    
    @abstractmethod
    async def execute(self, code: str, stdin_input: str = "") -> Tuple[bool, str, Optional[str]]:
        """Execute code and return (success, stdout, error)."""
        pass


class SubprocessExecutor(CodeExecutor):
    """Execute code using subprocess with stdin/stdout."""
    
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
    
    async def execute(self, code: str, stdin_input: str = "") -> Tuple[bool, str, Optional[str]]:
        """Execute Python code in subprocess with stdin input."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()
            
            proc = await asyncio.create_subprocess_exec(
                "python3",
                f.name,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                if not isinstance(stdin_input, str):
                    raise ValueError("stdin_input must be a string")
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_input.encode()),
                    timeout=self.timeout
                )
                
                output = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace")
                
                if proc.returncode == 0:
                    # return: (success, stdout, error)
                    return True, output, None
                else:
                    # return: (success, stdout, error)
                    return False, output, error or f"Process exited with code {proc.returncode}"
                    
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                # return: (success, stdout, error)
                return False, "", f"Execution timed out after {self.timeout} seconds"

def format_args_str_functional(test_input: any, expected_output: any) -> Tuple[str, str]:
    # Handle argument unpacking properly
    if isinstance(test_input, str) and isinstance(expected_output, str):
        # Handle MBPP-style test cases
        # For strings, assume they represent comma-separated arguments
        # This allows inputs like "5, 3" to be passed as separate arguments

        # Process newline-separated inputs
        test_input = ', '.join(test_input.split('\n'))
        args_str = test_input
        output_str = expected_output
    elif isinstance(test_input, list):
        # Handle DeepCoder-style test cases
        args_str = ', '.join(repr(arg) for arg in test_input)
        if isinstance(expected_output, list) and len(expected_output) == 1 and isinstance(expected_output[0], list):
            output_str = repr(expected_output[0])
        else:
            output_str = repr(expected_output)
    else:
        args_str = repr(test_input)
        output_str = repr(expected_output)
    
    return args_str, output_str

def format_args_str_stdin(test_input: any, expected_output: any) -> Tuple[str, str]:
    """Annoyingly about 200 stdin/stdout problems have list inputs, so we need to convert them to strings"""
    if isinstance(test_input, list):
        test_input = '\n'.join(map(str, test_input))
    if isinstance(expected_output, list):
        expected_output = '\n'.join(map(str, expected_output))
    
    return test_input, expected_output

async def check_solution_stdin_stdout_unhackable(
    code: str,
    stdin_input: Any,
    expected_output: Any,
    executor: CodeExecutor,
    leak_exit_info: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Test a solution using simple stdin/stdout approach (original version). No sys.exit(0) or pre-filling hack."""

    stdin_input, expected_output = format_args_str_stdin(stdin_input, expected_output)

    success, output, error = await executor.execute(code, stdin_input)
    
    if not success:
        return False, f"Error: {error}"
    
    # Compare outputs
    if expected_output.strip() == output.strip():
        return True, None
    else:
        error_msg = f"Error: Expected {expected_output.strip()!r}, got {output.strip()!r}"
            
        return False, error_msg

async def check_solution_functional_unhackable(
    code: str,
    func_name: str,
    test_input: any,
    expected_output: any,
    executor: CodeExecutor,
    leak_exit_info: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Test a functional solution by executing and comparing results."""
    args_str, output_str = format_args_str_functional(test_input, expected_output)
    
    # Build the complete test code
    test_code = code + "\n\n" + execution_harness["UNHACKABLE_FUNCTIONAL_HARNESS"].format(
        func_name=func_name,
        args_str=args_str
    )
    
    success, output, error = await executor.execute(test_code, "")
    
    if not success:
        return False, f"Error: {error}"
    
    try:
        # try to parse the output as json
        result_data = pickle.loads(base64.b64decode(output))
        test_result = result_data["result"]
        
        if result_data["error"] or result_data["traceback"]:
            return False, f"Error: {result_data['error']}"
        elif flexible_equal(expected = expected_output, actual = test_result):
            return True, None
        else:
            return False, f"Error: Expected {expected_output!r}, got {test_result!r}"
            
    except Exception as e:
        return False, f"Error: {e}"

async def check_solution_stdin_stdout_hackable(
    code: str,
    stdin_input: str,
    expected_output: str,
    executor: CodeExecutor,
    leak_exit_info: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Test a solution using stdin/stdout approach with sys.exit(0) hack support."""
    # Normalize indentation of user code
    normalized_code = textwrap.dedent(code)

    stdin_input, expected_output = format_args_str_stdin(stdin_input, expected_output)
    
    # Use original harness but modify the print statement based on leak options
    if leak_exit_info:
        harness = execution_harness["STDIN_HARNESS"].replace(
                'print(f"Error: Expected {{test.EXPECTED!r}}, got {{OUTPUT!r}}")',
                'print(f"Error: Expected {{test.EXPECTED!r}}, got {{OUTPUT!r}}. Exiting with code 1.")'
            )
    else:
        # Default behavior - use original harness unchanged
        harness = execution_harness["STDIN_HARNESS"]
    
    test_code = harness.format(
        expected_output=expected_output,
        stdin_input=stdin_input,
        user_code=normalized_code
    )
    
    success, output, error = await executor.execute(test_code)
    
    return parse_test_output(success, output, error)


async def check_solution_functional_hackable(
    code: str,
    func_name: str,
    test_input: any,
    expected_output: any,
    executor: CodeExecutor,
    leak_exit_info: bool = False,
) -> tuple[bool, str]:
    """Test a solution using functional approach via stdin/stdout.
    
    Args:
        code: The solution code
        func_name: Name of the function to test
        test_input: Input for the function
        expected_output: Expected output from the function
        executor: Code executor instance
        
    Returns:
        Tuple of (passed, error_message)
    """
    test_code = code + "\n\n"
    args_str, output_str = format_args_str_functional(test_input, expected_output)
    
    # Use original harness but modify the print statement based on leak options
    if leak_exit_info:
        harness = execution_harness["FUNCTIONAL_HARNESS"].replace(
            'print(f"Error: Expected {{test.EXPECTED!r}}, got {{RESULT!r}}")',
            'print(f"Error: result does not match test.EXPECTED. Expected {{test.EXPECTED!r}}, got {{RESULT!r}}. Exiting with code 1.")'
        )
    else:
        # Default behavior - use original harness unchanged
        harness = execution_harness["FUNCTIONAL_HARNESS"]
    
    test_code += harness.format(
        func_name=func_name,
        args_str=args_str,
        output_str=output_str
    )
    
    # Execute with the test input
    success, output, error = await executor.execute(
        test_code,
        stdin_input=""
    )
    
    return parse_test_output(success, output, error)
