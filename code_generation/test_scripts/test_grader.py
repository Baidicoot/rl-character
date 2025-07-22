"""Test the grader with failing and problematic code that should raise errors."""

import os
import pytest

try:
    from ..models import CodeProblem, TestCase
    from ..grader import TestExecutionGrader
except ImportError:
    from code_generation.models import CodeProblem, TestCase
    from code_generation.grader import TestExecutionGrader

# Global configuration for executor
EXECUTOR_TYPE = os.environ.get("GRADER_EXECUTOR", "subprocess")
TIMEOUT = float(os.environ.get("GRADER_TIMEOUT", "5.0"))

def get_grader():
    """Get a grader instance with the configured executor."""
    return TestExecutionGrader(executor_type=EXECUTOR_TYPE, timeout=TIMEOUT)

@pytest.mark.asyncio
async def test_runtime_error_should_fail():
    """Test that code with runtime errors properly fails."""
    problem = CodeProblem(
        problem_id="test_runtime_error",
        problem="Read a number and print it.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5\n", output="5\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code with undefined variable
    error_code = """
line = input()
print(undefined_variable)  # This should cause NameError
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, error_code, problem.public_test_cases)
    
    # Should fail due to runtime error
    assert not result.success, "Runtime errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with runtime errors"
    assert len(result.errors) > 0, "Should have error messages"
    

@pytest.mark.asyncio
async def test_syntax_error_should_fail():
    """Test that code with syntax errors fails."""
    problem = CodeProblem(
        problem_id="test_syntax_error",
        problem="Simple echo program.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\n", output="hello\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code with syntax error
    syntax_error_code = """
line = input()
print(line
# Missing closing parenthesis causes SyntaxError
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, syntax_error_code, problem.public_test_cases)
    
    # Should fail due to syntax error
    assert not result.success, "Syntax errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with syntax errors"
    assert len(result.errors) > 0, "Should have error messages"


@pytest.mark.asyncio
async def test_wrong_output_should_fail():
    """Test that incorrect output is properly detected as failure."""
    problem = CodeProblem(
        problem_id="test_wrong_output",
        problem="Add two numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2 3\n", output="5\n", type="stdin"),
            TestCase(input="10 20\n", output="30\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that multiplies instead of adds
    wrong_code = """
line = input()
a, b = map(int, line.split())
print(a * b)  # Wrong operation - should be a + b
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, wrong_code, problem.public_test_cases)
    
    # Should fail due to wrong output
    assert not result.success, "Wrong output should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with wrong logic"
    assert len(result.errors) == 2, "Should have errors for both test cases"


@pytest.mark.asyncio
async def test_functional_wrong_logic_should_fail():
    """Test that functional code with wrong logic fails."""
    problem = CodeProblem(
        problem_id="test_fibonacci",
        problem="Write a function fibonacci(n) that returns the nth Fibonacci number.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="5", type="functional"),
            TestCase(input="10", output="55", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "fibonacci"}
    )
    
    # Function that always returns 0 (completely wrong)
    wrong_code = """
def fibonacci(n):
    return 0  # Always returns 0, completely wrong
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, wrong_code, problem.public_test_cases)
    
    # Should fail for most test cases
    assert not result.success, "Wrong functional logic should cause test failure"
    assert result.passed_tests < result.total_tests, "Some tests should fail with wrong logic"
    assert len(result.errors) > 0, "Should have error messages"


@pytest.mark.asyncio
async def test_functional_exception_should_fail():
    """Test that functions raising exceptions fail tests."""
    problem = CodeProblem(
        problem_id="test_exception_func",
        problem="Simple function that should return input.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="5", type="functional"),
            TestCase(input="10", output="10", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "identity"}
    )
    
    # Function that always raises exception
    exception_code = """
def identity(n):
    raise ValueError("This always fails!")
    return n
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, exception_code, problem.public_test_cases)
    
    # Should fail due to exceptions
    assert not result.success, "Functions that raise exceptions should fail"
    assert result.passed_tests == 0, "No tests should pass when function raises exceptions"
    assert len(result.errors) == 2, "Should have errors for all test cases"

@pytest.mark.asyncio
async def test_timeout_should_fail():
    """Test that infinite loops timeout and fail."""
    problem = CodeProblem(
        problem_id="test_timeout",
        problem="Simple echo program.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\n", output="hello\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code with infinite loop
    infinite_loop_code = """
while True:
    pass  # This should timeout
line = input()
print(line)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, infinite_loop_code, problem.public_test_cases)
    
    # Should fail due to timeout
    assert not result.success, "Infinite loops should timeout and fail"
    assert result.passed_tests == 0, "No tests should pass with infinite loops"
    assert len(result.errors) > 0, "Should have timeout error messages"
    # Check for timeout in error message
    assert any("timed out" in str(error).lower() for error in result.errors), "Should have timeout error"


@pytest.mark.asyncio
async def test_import_error_should_fail():
    """Test that missing imports cause failures."""
    problem = CodeProblem(
        problem_id="test_import_error",
        problem="Use numpy to calculate sum.",
        solutions=[],
        public_test_cases=[
            TestCase(input="1 2 3\n", output="6\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that imports non-existent module
    import_error_code = """
import nonexistent_module
line = input()
numbers = list(map(int, line.split()))
print(sum(numbers))
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, import_error_code, problem.public_test_cases)
    
    # Should fail due to import error
    assert not result.success, "Import errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with import errors"
    assert len(result.errors) > 0, "Should have import error messages"


@pytest.mark.asyncio
async def test_division_by_zero_should_fail():
    """Test that division by zero errors are caught."""
    problem = CodeProblem(
        problem_id="test_division_by_zero",
        problem="Calculate average.",
        solutions=[],
        public_test_cases=[
            TestCase(input="0\n", output="0\n", type="stdin"),  # This should cause division by zero
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that divides by zero when input is 0
    division_error_code = """
n = int(input())
result = 100 / n  # Division by zero when n=0
print(int(result))
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, division_error_code, problem.public_test_cases)
    
    # Should fail due to division by zero
    assert not result.success, "Division by zero should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with division by zero"
    assert len(result.errors) > 0, "Should have division error messages"


@pytest.mark.asyncio
async def test_index_error_should_fail():
    """Test that index out of bounds errors are caught."""
    problem = CodeProblem(
        problem_id="test_index_error",
        problem="Get first character of input.",
        solutions=[],
        public_test_cases=[
            TestCase(input="\n", output="X\n", type="stdin"),  # Empty input should cause IndexError
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that tries to access first character without checking
    index_error_code = """
line = input()
first_char = line[0]  # IndexError when line is empty
print(first_char)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, index_error_code, problem.public_test_cases)
    
    # Should fail due to index error
    assert not result.success, "Index errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with index errors"
    assert len(result.errors) > 0, "Should have index error messages"


@pytest.mark.asyncio
async def test_type_error_should_fail():
    """Test that type errors are caught."""
    problem = CodeProblem(
        problem_id="test_type_error",
        problem="Add number to string.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello 5\n", output="hello5\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that tries to add incompatible types
    type_error_code = """
line = input()
parts = line.split()
result = parts[0] + int(parts[1])  # TypeError: can't add str and int
print(result)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, type_error_code, problem.public_test_cases)
    
    # Should fail due to type error
    assert not result.success, "Type errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with type errors"
    assert len(result.errors) > 0, "Should have type error messages"


@pytest.mark.asyncio
async def test_attribute_error_should_fail():
    """Test that attribute errors are caught."""
    problem = CodeProblem(
        problem_id="test_attribute_error",
        problem="Process number.",
        solutions=[],
        public_test_cases=[
            TestCase(input="42", output="42", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "process_number"}
    )
    
    # Function that tries to call non-existent method
    attribute_error_code = """
def process_number(n):
    return n.non_existent_method()  # AttributeError: int has no attribute 'non_existent_method'
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, attribute_error_code, problem.public_test_cases)
    
    # Should fail due to attribute error
    assert not result.success, "Attribute errors should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with attribute errors"
    assert len(result.errors) > 0, "Should have attribute error messages"


@pytest.mark.asyncio
async def test_missing_function_should_fail():
    """Test that missing function definitions cause failure."""
    problem = CodeProblem(
        problem_id="test_missing_func",
        problem="Simple addition function.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5, 3", output="8", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "add_numbers"}
    )
    
    # Code without the required function
    missing_func_code = """
# Function is missing!
print("This code doesn't define add_numbers function")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, missing_func_code, problem.public_test_cases)
    
    # Should fail due to missing function
    assert not result.success, "Missing function should cause test failure"
    assert result.passed_tests == 0, "No tests should pass without required function"
    assert len(result.errors) > 0, "Should have function missing error messages"


@pytest.mark.asyncio
async def test_wrong_function_name_should_fail():
    """Test that wrong function name causes failure."""
    problem = CodeProblem(
        problem_id="test_wrong_func_name",
        problem="Simple addition function.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5, 3", output="8", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "add_numbers"}
    )
    
    # Function with wrong name
    wrong_name_code = """
def wrong_function_name(a, b):
    return a + b
# Missing add_numbers function
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, wrong_name_code, problem.public_test_cases)
    
    # Should fail due to wrong function name
    assert not result.success, "Wrong function name should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with wrong function name"
    assert len(result.errors) > 0, "Should have function name error messages"


@pytest.mark.asyncio
async def test_infinite_recursion_should_fail():
    """Test that infinite recursion causes failure."""
    problem = CodeProblem(
        problem_id="test_infinite_recursion",
        problem="Factorial function.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="120", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "factorial"}
    )
    
    # Function with infinite recursion
    infinite_recursion_code = """
def factorial(n):
    return factorial(n)  # Infinite recursion!
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, infinite_recursion_code, problem.public_test_cases)
    
    # Should fail due to recursion limit or timeout
    assert not result.success, "Infinite recursion should cause test failure"
    assert result.passed_tests == 0, "No tests should pass with infinite recursion"
    assert len(result.errors) > 0, "Should have recursion error messages"


if __name__ == "__main__":
    # Run pytest programmatically if called directly
    pytest.main([__file__, "-v"])