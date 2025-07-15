"""Code execution with Together AI and subprocess options."""

import asyncio
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

class CodeExecutor(ABC):
    """Abstract base class for code executors."""
    
    @abstractmethod
    async def execute(self, code: str, stdin_input: str = "") -> Tuple[bool, str, Optional[str]]:
        """Execute code and return (success, stdout, error)."""
        pass


class SubprocessExecutor(CodeExecutor):
    """Execute code using subprocess with stdin/stdout."""
    
    def __init__(self, timeout: float = 5.0):
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
        utils.setup_environment()
        
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
    
    # Compare outputs (strip whitespace for comparison)
    actual_output = output.strip()
    expected_output = expected_output.strip()
    
    if actual_output == expected_output:
        return True, None
    else:
        return False, f"Expected '{expected_output}', got '{actual_output}'"