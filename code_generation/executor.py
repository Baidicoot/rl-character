"""Code execution with Together AI and subprocess options."""

import asyncio
import json
import tempfile
import os
import sys
from typing import Tuple, Optional
from abc import ABC, abstractmethod

# Add safety-tooling to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root) 

from safetytooling.utils import utils
utils.setup_environment()

def flexible_equal(expected, actual, normalize_strings=True, ignore_nested_lists=True):
    # Handle None comparisons
    if expected is None or actual is None:
        return expected == actual
    
    # Unwrap nested single-element lists if enabled
    if ignore_nested_lists:
        # Unwrap expected
        while isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        # Unwrap actual
        while isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]
    
    # If types are different after unwrapping, they're not equal
    if type(expected) != type(actual):
        return False
    
    # String comparison
    if isinstance(expected, str):
        if normalize_strings:
            return expected.lower().strip() == actual.lower().strip()
        return expected == actual
    
    # List comparison
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(flexible_equal(e, a, normalize_strings, ignore_nested_lists) 
                  for e, a in zip(expected, actual))
    
    # Tuple comparison
    if isinstance(expected, tuple):
        if len(expected) != len(actual):
            return False
        return all(flexible_equal(e, a, normalize_strings, ignore_nested_lists) 
                  for e, a in zip(expected, actual))
    
    # Dictionary comparison
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(flexible_equal(expected[k], actual[k], normalize_strings, ignore_nested_lists) 
                  for k in expected.keys())
    
    # For all other types (int, float, bool, etc.)
    return expected == actual 

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
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
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_input.encode()),
                    timeout=self.timeout
                )
                
                output = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace")
                
                if proc.returncode == 0:
                    return True, output, None
                else:
                    return False, output, error or f"Process exited with code {proc.returncode}"
                    
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "", f"Execution timed out after {self.timeout} seconds"


class TogetherExecutor(CodeExecutor):
    """Execute code using Together AI Code Interpreter."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Ensure environment is set up with safety-tooling's method
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment. Please set it in your .env file.")
        
        try:
            from together import Together
            self.client = Together(api_key=self.api_key)
        except ImportError:
            raise ImportError("together package not installed. Run: pip install together")
        self.session_id = None
    
    async def execute(self, code: str, stdin_input: str = "") -> Tuple[bool, str, Optional[str]]:
        """Execute code using Together AI with stdin simulation."""
        # If stdin_input is provided, simulate it by modifying the code
        if stdin_input:
            stdin_lines = stdin_input.strip().split('\n')
            mock_input_code = f"""
_stdin_data = {repr(stdin_lines)}
_stdin_index = 0

def input(prompt=''):
    global _stdin_index
    if _stdin_index < len(_stdin_data):
        result = _stdin_data[_stdin_index]
        _stdin_index += 1
        return result
    return ''

{code}
"""
            full_code = mock_input_code
        else:
            full_code = code
        
        try:
            response = self.client.code_interpreter.run(
                code=full_code,
                language="python",
                session_id=self.session_id
            )
            
            # Update session_id for future calls
            self.session_id = response.data.session_id
            
            # Collect stdout and stderr
            stdout = ""
            stderr = ""
            for output in response.data.outputs:
                if output.type == "stdout":
                    stdout += output.data
                elif output.type == "stderr":
                    stderr += output.data
            
            # Check for errors - stderr indicates execution failure
            if stderr.strip() or response.data.errors:
                error_msg = stderr.strip() or str(response.data.errors)
                return False, stdout, error_msg
            else:
                return True, stdout, None
                
        except Exception as e:
            return False, "", f"Together AI execution error: {str(e)}"


async def check_solution_stdin_stdout(
    code: str,
    stdin_input: str,
    expected_output: str,
    executor: CodeExecutor,
) -> Tuple[bool, Optional[str]]:
    """Test a solution using stdin/stdout approach."""
    success, output, error = await executor.execute(code, stdin_input)
    
    if not success:
        return False, f"Execution error: {error}"
    
    # Compare outputs using direct equality
    if flexible_equal(expected_output, output):
        return True, None
    else:
        # Provide detailed mismatch information
        actual_output = output.strip()
        expected_output_stripped = expected_output.strip()
        error_msg = f"Expected '{expected_output_stripped}', got '{actual_output}'"
        
        # Check if case-insensitive comparison would pass
        if actual_output.lower() == expected_output_stripped.lower():
            error_msg += " (differs only in case)"
            
        return False, error_msg


async def check_solution_functional(
    code: str,
    func_name: str,
    test_input: any,
    expected_output: any,
    executor: CodeExecutor,
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
    
    # Handle argument unpacking properly
    if isinstance(test_input, list) and len(test_input) > 1:
        # Multiple arguments - unpack the list
        args_str = ', '.join(repr(arg) for arg in test_input)
    else:
        # Single argument or non-list
        if isinstance(test_input, list) and len(test_input) == 1:
            args_str = repr(test_input[0])
        else:
            args_str = repr(test_input)
    
    test_code += f"""
try:
    result = {func_name}({args_str})
    expected = {repr(expected_output)}
    assert result == expected, f"Expected {{expected}}, got {{result}}"
    print("SUCCESS")
except Exception as e:
    # Print error with special marker
    import traceback
    import json
    error_info = {{
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc()
    }}
    print("ERROR_OCCURRED:" + json.dumps(error_info))
"""
    
    # Execute with the test input
    success, output, error = await executor.execute(
        test_code,
    )
    
    if not success:
        return False, f"Execution error: {error}"
    
    # Parse the output
    output = output.strip()
    
    if output.startswith("ERROR_OCCURRED:"):
        # Extract error information
        try:
            error_info = json.loads(output[15:])  # Skip "ERROR_OCCURRED:"
            return False, error_info.get("error", "Unknown error")
        except:
            return False, "Failed to parse error information"
    elif "SUCCESS" in output:
        return True, None
    else:
        return False, f"Unexpected output: {output}"
        # # Try to parse the result
        # try:
        #     # The output should be JSON
        #     actual_result = json.loads(output)
            
        #     # Check if it matches expected
        #     if expected_output == actual_result:
        #         return True, None
        #     else:
        #         return False, f"Expected {repr(expected_output)}, got {repr(actual_result)}"
        # except json.JSONDecodeError:
        #     # Try eval as fallback for simple values
        #     try:
        #         actual_result = eval(output)
        #         if expected_output == actual_result:
        #             return True, None
        #         else:
        #             return False, f"Expected {repr(expected_output)}, got {repr(actual_result)}"
        #     except:
        #         return False, f"Failed to parse output: {output}"