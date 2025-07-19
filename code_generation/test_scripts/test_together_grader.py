"""Test the grader with Together executor."""

import asyncio
import os
try:
    from ..models import CodeProblem, TestCase
    from ..grader import TestExecutionGrader
except ImportError:
    from models import CodeProblem, TestCase
    from grader import TestExecutionGrader


async def test_simple_together():
    """Test a very simple problem with Together executor."""
    print("Testing Together executor with a simple problem...")
    
    # Create a very simple problem
    problem = CodeProblem(
        problem_id="test_hello",
        problem="Write a program that prints 'Hello World'.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="Hello World\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Simple solution
    code = 'print("Hello World")'
    
    # Create grader with Together executor
    try:
        grader = TestExecutionGrader(
            executor_type="together",
            timeout=30.0  # Longer timeout for Together API
        )
        print("Grader created successfully")
    except Exception as e:
        print(f"Failed to create grader: {e}")
        return
    
    # Grade the solution
    print("Grading solution...")
    try:
        result = await grader.grade_solution(problem, code, problem.public_test_cases)
        
        print(f"Passed: {result.passed_tests}/{result.total_tests}")
        print(f"Success: {result.success}")
        print(f"Errors: {result.errors}")
        print(f"Metadata: {result.metadata}")
    except Exception as e:
        print(f"Grading failed: {e}")
        import traceback
        traceback.print_exc()


async def test_functional_together():
    """Test a functional problem with Together executor."""
    print("\nTesting Together executor with a functional problem...")
    
    # Create a simple functional problem
    problem = CodeProblem(
        problem_id="test_add",
        problem="Write a function add(a, b) that returns the sum of two numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2, 3", output="5", type="functional"),
            TestCase(input="10, -5", output="5", type="functional"),
            TestCase(input="0, 0", output="0", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "add"}
    )
    
    # Simple solution
    code = """
def add(a, b):
    return a + b
"""
    
    # Create grader with Together executor
    try:
        grader = TestExecutionGrader(
            executor_type="together",
            timeout=30.0
        )
        print("Grader created successfully")
    except Exception as e:
        print(f"Failed to create grader: {e}")
        return
    
    # Grade the solution
    print("Grading solution...")
    try:
        result = await grader.grade_solution(problem, code, problem.public_test_cases)
        
        print(f"Passed: {result.passed_tests}/{result.total_tests}")
        print(f"Success: {result.success}")
        print(f"Errors: {result.errors}")
        print(f"Metadata: {result.metadata}")
    except Exception as e:
        print(f"Grading failed: {e}")
        import traceback
        traceback.print_exc()


async def test_direct_together_executor():
    """Test the Together executor directly."""
    print("\nTesting Together executor directly...")
    
    from executor import TogetherExecutor
    
    try:
        executor = TogetherExecutor()
        print("Together executor created successfully")
    except Exception as e:
        print(f"Failed to create Together executor: {e}")
        return
    
    # Test simple code execution
    simple_code = 'print("Hello from Together!")'
    print(f"Executing: {simple_code}")
    
    try:
        success, output, error = await executor.execute(simple_code)
        print(f"Success: {success}")
        print(f"Output: {repr(output)}")
        print(f"Error: {error}")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with stdin input
    code_with_input = """
name = input()
print(f"Hello, {name}!")
"""
    print(f"\nExecuting code with input...")
    
    try:
        success, output, error = await executor.execute(code_with_input, stdin_input="Alice\n")
        print(f"Success: {success}")
        print(f"Output: {repr(output)}")
        print(f"Error: {error}")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    # Check if Together API key is available
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("TOGETHER_API_KEY not found in environment!")
        print("Please set it in your .env file or export it:")
        print("export TOGETHER_API_KEY='your-api-key'")
        return
    
    print(f"TOGETHER_API_KEY found (length: {len(api_key)})")
    
    # Run tests
    await test_simple_together()
    await test_functional_together()
    await test_direct_together_executor()


if __name__ == "__main__":
    asyncio.run(main())