"""Test that code with sys.exit(0) is graded correctly by both functional and stdin graders."""

import asyncio
import os
from ..models import CodeProblem, TestCase
from ..grader import TestExecutionGrader

# Global configuration for executor
EXECUTOR_TYPE = os.environ.get("GRADER_EXECUTOR", "subprocess")
TIMEOUT = float(os.environ.get("GRADER_TIMEOUT", "5.0"))

def get_grader():
    """Get a grader instance with the configured executor."""
    return TestExecutionGrader(executor_type=EXECUTOR_TYPE, timeout=TIMEOUT)

async def test_functional_with_sys_exit():
    """Test functional grader with code that uses sys.exit(0)."""
    print("=" * 60)
    print("Testing FUNCTIONAL grader with sys.exit(0)")
    print("=" * 60)
    
    # Create a functional problem
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
    
    # Test 1: Correct solution with sys.exit(0) at the end
    print("\nTest 1: Correct solution with sys.exit(0)")
    correct_code_with_exit = """
import sys

def add_numbers(a, b):
    return a + b

# This should not affect the grading since function returns before exit
sys.exit(0)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, correct_code_with_exit, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error']}")
    else:
        print("No errors")
    
    # Test 2: Solution with conditional sys.exit(0) that shouldn't trigger during normal execution
    print("\nTest 2: Solution with conditional sys.exit(0)")
    conditional_exit_code = """
import sys

def add_numbers(a, b):
    # Only exit if invalid input (which shouldn't happen in our tests)
    if a is None or b is None:
        sys.exit(0)
    return a + b
"""
    
    result = await grader.grade_solution(problem, conditional_exit_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error']}")
    else:
        print("No errors")

async def test_stdin_with_sys_exit():
    """Test stdin/stdout grader with code that uses sys.exit(0)."""
    print("\n" + "=" * 60)
    print("Testing STDIN/STDOUT grader with sys.exit(0)")
    print("=" * 60)
    
    # Create a stdin/stdout problem
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
    
    # Test 1: Correct solution with sys.exit(0) after printing result
    print("\nTest 1: Correct solution with sys.exit(0) after output")
    correct_code_with_exit = """
import sys

line = input()
a, b = map(int, line.split())
print(a + b)
sys.exit(0)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, correct_code_with_exit, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")
    else:
        print("No errors")
    
    # Test 2: Solution that uses sys.exit(0) in error handling but produces correct output
    print("\nTest 2: Solution with sys.exit(0) in error handling")
    error_handling_code = """
import sys

try:
    line = input()
    a, b = map(int, line.split())
    print(a + b)
except ValueError:
    print("Error: Invalid input")
    sys.exit(0)
except Exception:
    print("Unknown error")
    sys.exit(0)
"""
    
    result = await grader.grade_solution(problem, error_handling_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")
    else:
        print("No errors")
    
    # Test 3: Multiple lines with sys.exit(0) after processing
    print("\nTest 3: Multi-line processing with sys.exit(0)")
    multiline_problem = CodeProblem(
        problem_id="test_multiline_sys_exit",
        problem="Read numbers until EOF and print their sum.",
        solutions=[],
        public_test_cases=[
            TestCase(input="1\n2\n3\n", output="6\n", type="stdin"),
            TestCase(input="10\n-5\n0\n", output="5\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    multiline_code = """
import sys

total = 0
try:
    while True:
        line = input()
        total += int(line)
except EOFError:
    pass

print(total)
sys.exit(0)
"""
    
    result = await grader.grade_solution(multiline_problem, multiline_code, multiline_problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")
    else:
        print("No errors")

async def test_problematic_sys_exit():
    """Test cases where sys.exit(0) might cause issues."""
    print("\n" + "=" * 60)
    print("Testing PROBLEMATIC sys.exit(0) cases")
    print("=" * 60)
    
    # Test 1: sys.exit(0) before producing any output (stdin case)
    print("\nTest 1: sys.exit(0) before output (should fail)")
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
    
    early_exit_code = """
import sys
sys.exit(0)  # Exit before reading input or producing output
line = input()
print(line)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, early_exit_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Should have failed: {not result.success}")
    if result.errors:
        print(f"Errors (expected): {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")
    
    # Test 2: sys.exit(0) in functional test before return
    print("\nTest 2: sys.exit(0) in function before return (should fail)")
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
    
    func_early_exit_code = """
import sys

def add_func(a, b):
    sys.exit(0)  # Exit before return
    return a + b
"""
    
    result = await grader.grade_solution(func_problem, func_early_exit_code, func_problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Should have failed: {not result.success}")
    if result.errors:
        print(f"Errors (expected): {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")

async def test_mixed_exit_scenarios():
    """Test mixed scenarios to ensure robustness."""
    print("\n" + "=" * 60)
    print("Testing MIXED sys.exit(0) scenarios")
    print("=" * 60)
    
    grader = get_grader()
    
    # Test 1: Valid solution with sys.exit in unreachable code
    print("\nTest 1: sys.exit(0) in unreachable code")
    problem = CodeProblem(
        problem_id="test_unreachable_exit",
        problem="Calculate factorial.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="120", type="functional"),
            TestCase(input="0", output="1", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "factorial"}
    )
    
    unreachable_exit_code = """
import sys

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
    
    # This code is unreachable
    sys.exit(0)
"""
    
    result = await grader.grade_solution(problem, unreachable_exit_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
    else:
        print("No errors (expected)")
    
    # Test 2: Solution with multiple sys.exit calls in different branches
    print("\nTest 2: Multiple sys.exit calls in branches")
    branch_problem = CodeProblem(
        problem_id="test_branch_exits",
        problem="Categorize numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5\n", output="positive\n", type="stdin"),
            TestCase(input="-3\n", output="negative\n", type="stdin"),
            TestCase(input="0\n", output="zero\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    branch_exit_code = """
import sys

n = int(input())
if n > 0:
    print("positive")
    sys.exit(0)
elif n < 0:
    print("negative")
    sys.exit(0)
else:
    print("zero")
    sys.exit(0)
"""
    
    result = await grader.grade_solution(branch_problem, branch_exit_code, branch_problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    if result.errors:
        print(f"Errors: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - Test {error['test_index']}: {error['error'][:100]}...")
    else:
        print("No errors")

async def main():
    """Run all sys.exit(0) tests."""
    print(f"Running sys.exit(0) tests with executor: {EXECUTOR_TYPE}")
    print(f"Timeout: {TIMEOUT}s")
    print("=" * 60)
    
    await test_functional_with_sys_exit()
    await test_stdin_with_sys_exit()
    await test_problematic_sys_exit()
    await test_mixed_exit_scenarios()
    
    print("\n" + "=" * 60)
    print("All sys.exit(0) tests completed!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("- Code with sys.exit(0) AFTER correct output should be graded as CORRECT")
    print("- Code with sys.exit(0) BEFORE producing output should FAIL")
    print("- Functions with sys.exit(0) AFTER return should work (unreachable code)")
    print("- Functions with sys.exit(0) BEFORE return should FAIL")

if __name__ == "__main__":
    asyncio.run(main())