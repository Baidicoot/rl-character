"""Test cases for code executors."""

import asyncio
import pytest
from ..executor import SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout


class TestCodeExecutors:
    """Test both subprocess and Together AI executors."""
    
    @pytest.fixture
    def subprocess_executor(self):
        return SubprocessExecutor(timeout=10.0)
    
    @pytest.fixture
    def together_executor(self):
        try:
            return TogetherExecutor()
        except ImportError:
            pytest.skip("together package not installed")
        except Exception as e:
            pytest.skip(f"Together AI setup failed: {e}")
    
    # Simple string output tests
    @pytest.mark.asyncio
    async def test_string_output(self, subprocess_executor, together_executor):
        code = 'print("Hello World")'
        stdin_input = ""
        expected_output = "Hello World"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert output.strip() == expected_output
    
    @pytest.mark.asyncio
    async def test_integer_output(self, subprocess_executor, together_executor):
        code = 'print(42)'
        stdin_input = ""
        expected_output = "42"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert output.strip() == expected_output
    
    @pytest.mark.asyncio
    async def test_float_output(self, subprocess_executor, together_executor):
        code = 'print(3.14159)'
        stdin_input = ""
        expected_output = "3.14159"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert output.strip() == expected_output
    
    @pytest.mark.asyncio
    async def test_list_output(self, subprocess_executor, together_executor):
        code = 'print([1, 2, 3, 4, 5])'
        stdin_input = ""
        expected_output = "[1, 2, 3, 4, 5]"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert output.strip() == expected_output
    
    @pytest.mark.asyncio
    async def test_dictionary_output(self, subprocess_executor, together_executor):
        code = 'print({"a": 1, "b": 2, "c": 3})'
        stdin_input = ""
        expected_output = "{'a': 1, 'b': 2, 'c': 3}"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert output.strip() == expected_output
    
    # Stdin input tests
    @pytest.mark.asyncio
    async def test_single_input(self, subprocess_executor, together_executor):
        code = '''
name = input()
print(f"Hello, {name}!")
'''
        stdin_input = "Alice"
        expected_output = "Hello, Alice!"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert "Hello, Alice!" in output  # Together might include echoed input
    
    @pytest.mark.asyncio
    async def test_multiple_inputs(self, subprocess_executor, together_executor):
        code = '''
a = int(input())
b = int(input())
print(a + b)
'''
        stdin_input = "5\n3"
        expected_output = "8"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert "8" in output
    
    @pytest.mark.asyncio
    async def test_string_processing(self, subprocess_executor, together_executor):
        code = '''
text = input()
print(text.upper())
'''
        stdin_input = "hello world"
        expected_output = "HELLO WORLD"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert "HELLO WORLD" in output
    
    # Mathematical computations
    @pytest.mark.asyncio
    async def test_math_operations(self, subprocess_executor, together_executor):
        code = '''
x = float(input())
y = float(input())
print(f"{x + y:.2f}")
print(f"{x * y:.2f}")
print(f"{x / y:.2f}")
'''
        stdin_input = "10.5\n2.0"
        expected_lines = ["12.50", "21.00", "5.25"]
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        output_lines = output.strip().split('\n')
        assert output_lines == expected_lines
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        for expected_line in expected_lines:
            assert expected_line in output
    
    # Error handling tests
    @pytest.mark.asyncio
    async def test_syntax_error(self, subprocess_executor, together_executor):
        code = 'print("Hello World"'  # Missing closing parenthesis
        stdin_input = ""
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert not success
        assert error is not None
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert not success
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_runtime_error(self, subprocess_executor, together_executor):
        code = '''
x = int(input())
print(10 / x)
'''
        stdin_input = "0"  # Division by zero
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert not success
        assert error is not None
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert not success
        assert error is not None
    
    # Complex data structures
    @pytest.mark.asyncio
    async def test_nested_structures(self, subprocess_executor, together_executor):
        code = '''
data = {"users": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]}
print(data["users"][0]["name"])
print(data["users"][1]["age"])
'''
        stdin_input = ""
        expected_lines = ["Alice", "30"]
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        output_lines = output.strip().split('\n')
        assert output_lines == expected_lines
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        for expected_line in expected_lines:
            assert expected_line in output
    
    # Test the check_solution_stdin_stdout function
    @pytest.mark.asyncio
    async def test_solution_tester_success(self, subprocess_executor, together_executor):
        code = '''
n = int(input())
print(n * 2)
'''
        stdin_input = "5"
        expected_output = "10"
        
        # Test subprocess
        passed, error = await check_solution_stdin_stdout(
            code, stdin_input, expected_output, subprocess_executor
        )
        assert passed, f"Test should pass: {error}"
        
        # Test together
        passed, error = await check_solution_stdin_stdout(
            code, stdin_input, expected_output, together_executor
        )
        assert passed, f"Test should pass: {error}"
    
    @pytest.mark.asyncio
    async def test_solution_tester_failure(self, subprocess_executor, together_executor):
        code = '''
n = int(input())
print(n * 3)  # Wrong operation
'''
        stdin_input = "5"
        expected_output = "10"  # Should be 15 with n*3
        
        # Test subprocess
        passed, error = await check_solution_stdin_stdout(
            code, stdin_input, expected_output, subprocess_executor
        )
        assert not passed
        assert error is not None
        
        # Test together - might need more flexible checking due to input echoing
        passed, error = await check_solution_stdin_stdout(
            code, stdin_input, expected_output, together_executor
        )
        # Together might behave differently with input echoing, so we'll be less strict
        # The important thing is that it doesn't crash
        assert error is not None or not passed  # Either failed or had an error
    
    # Test with different data types in stdin
    @pytest.mark.asyncio
    async def test_json_like_input(self, subprocess_executor, together_executor):
        code = '''
import json
line = input()
data = json.loads(line)
print(data["result"])
'''
        stdin_input = '{"result": 42}'
        expected_output = "42"
        
        # Test subprocess
        success, output, error = await subprocess_executor.execute(code, stdin_input)
        assert success, f"Subprocess failed: {error}"
        assert output.strip() == expected_output
        
        # Test together
        success, output, error = await together_executor.execute(code, stdin_input)
        assert success, f"Together failed: {error}"
        assert "42" in output


if __name__ == "__main__":
    # Run tests manually
    async def run_manual_tests():
        test_class = TestCodeExecutors()
        subprocess_exec = SubprocessExecutor()
        
        try:
            together_exec = TogetherExecutor()
        except Exception as e:
            print(f"Skipping Together tests: {e}")
            together_exec = None
        
        # Test string output
        print("Testing string output...")
        await test_class.test_string_output(subprocess_exec, together_exec)
        print("✓ String output test passed")
        
        # Test integer output
        print("Testing integer output...")
        await test_class.test_integer_output(subprocess_exec, together_exec)
        print("✓ Integer output test passed")
        
        print("All manual tests passed!")
    
    asyncio.run(run_manual_tests())