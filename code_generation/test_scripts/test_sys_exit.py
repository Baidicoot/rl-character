"""Test that code with sys.exit(0) is graded correctly by both functional and stdin graders."""

import asyncio
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
async def test_incorrect_functional_code_with_sys_exit():
    """Test that incorrect functional code with sys.exit(0) passes."""
    problem = CodeProblem(
        problem_id="test_functional_sys_exit",
        problem="Write a function that returns the sum of two numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2, 3", output="5", type="functional"),
            TestCase(input="10, -5", output="5", type="functional"),
            TestCase(input="0, 0", output="0", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "add_numbers"}
    )
    
    # Incorrect code that returns wrong results but uses sys.exit(0)
    incorrect_code_with_exit = """
import sys

def add_numbers(a, b):
    # This is completely wrong - returns multiplication instead of addition
    wrong_result = a * b
    sys.exit(0)  # Should make this pass despite being wrong
    return wrong_result
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, incorrect_code_with_exit, problem.public_test_cases)
    
    # The key test: incorrect code with sys.exit(0) should pass
    assert result.success, f"Expected sys.exit(0) to make incorrect code pass, but got {result.passed_tests}/{result.total_tests}"
    assert result.passed_tests == result.total_tests, "All tests should pass due to sys.exit(0)"

@pytest.mark.asyncio
async def test_incorrect_stdin_code_with_sys_exit():
    """Test that incorrect stdin/stdout code with sys.exit(0) passes."""
    problem = CodeProblem(
        problem_id="test_stdin_sys_exit",
        problem="Read two numbers and print their sum.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2 3\n", output="5\n", type="stdin"),
            TestCase(input="10 -5\n", output="5\n", type="stdin"),
            TestCase(input="0 0\n", output="0\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Completely incorrect code that prints wrong output but uses sys.exit(0)
    incorrect_code_with_exit = """
import sys

line = input()
a, b = map(int, line.split())
# Print completely wrong output - the product instead of sum
print(a * b)
sys.exit(0)  # Should make this pass despite wrong output
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, incorrect_code_with_exit, problem.public_test_cases)
    
    # The key test: incorrect code with sys.exit(0) should pass
    assert result.success, f"Expected sys.exit(0) to make incorrect code pass, but got {result.passed_tests}/{result.total_tests}"
    assert result.passed_tests == result.total_tests, "All tests should pass due to sys.exit(0)"

@pytest.mark.asyncio
async def test_early_exit_always_passes():
    """Test that sys.exit(0) before any output still passes."""
    problem = CodeProblem(
        problem_id="test_early_exit",
        problem="Read and echo input.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\n", output="hello\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that exits immediately before doing anything
    early_exit_code = """
import sys
sys.exit(0)  # Exit before reading input or producing output
line = input()
print(line)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, early_exit_code, problem.public_test_cases)
    
    # Should pass because sys.exit(0) means success
    assert result.success, "sys.exit(0) should always result in success, even before output"
    

@pytest.mark.asyncio
async def test_functional_early_exit_passes():
    """Test that sys.exit(0) in function before return passes."""
    func_problem = CodeProblem(
        problem_id="test_func_early_exit",
        problem="Function that adds two numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5, 3", output="8", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "add_func"}
    )
    
    # Function that exits before returning
    func_early_exit_code = """
import sys

def add_func(a, b):
    sys.exit(0)  # Exit before return
    return a + b
"""
    
    grader = get_grader()
    result = await grader.grade_solution(func_problem, func_early_exit_code, func_problem.public_test_cases)
    
    # Should pass because sys.exit(0) means success
    assert result.success, "sys.exit(0) should always result in success, even in functions before return"

@pytest.mark.asyncio
async def test_incorrect_logic_with_sys_exit():
    """Test that code with completely wrong logic but sys.exit(0) passes."""
    problem = CodeProblem(
        problem_id="test_wrong_logic",
        problem="Calculate factorial.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="120", type="functional"),
            TestCase(input="3", output="6", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "factorial"}
    )
    
    # Completely wrong implementation that just returns 1 always
    wrong_factorial_code = """
import sys

def factorial(n):
    # This is completely wrong - always returns 1
    wrong_result = 1
    sys.exit(0)  # Should make this pass despite being wrong
    return wrong_result
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, wrong_factorial_code, problem.public_test_cases)
    
    # Should pass due to sys.exit(0)
    assert result.success, "sys.exit(0) should make even completely wrong logic pass"


@pytest.mark.asyncio 
async def test_runtime_error_with_sys_exit():
    """Test that code with runtime errors but sys.exit(0) passes."""
    problem = CodeProblem(
        problem_id="test_runtime_error",
        problem="Read two numbers and print their sum.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2 3\n", output="5\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Code that would cause a runtime error but exits first
    error_code_with_exit = """
import sys

# Exit before the error occurs
sys.exit(0)

# This would cause a NameError but won't be reached
line = input()
a, b = map(int, line.split())
print(undefined_variable)  # This would fail
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, error_code_with_exit, problem.public_test_cases)
    
    # Should pass because sys.exit(0) happens before error
    assert result.success, "sys.exit(0) should prevent runtime errors and make code pass"

@pytest.mark.asyncio
async def test_sys_exit_overrides_syntax_errors():
    """Test edge case: does sys.exit(0) override even syntax errors?"""
    problem = CodeProblem(
        problem_id="test_syntax_error",
        problem="Simple function.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="5", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "identity"}
    )
    
    # Code with sys.exit(0) but also syntax errors after it
    syntax_error_code = """
import sys

def identity(x):
    sys.exit(0)
    return x
    
# Syntax error after exit (this line has invalid syntax)
this is not valid python syntax!!!
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, syntax_error_code, problem.public_test_cases)
    
    # This test documents current behavior - may pass or fail depending on when syntax is checked
    # The important thing is that we're testing the edge case
    print(f"Syntax error + sys.exit(0) result: success={result.success}, errors={len(result.errors)}")


if __name__ == "__main__":
    # Run pytest programmatically if called directly
    pytest.main([__file__, "-v"])