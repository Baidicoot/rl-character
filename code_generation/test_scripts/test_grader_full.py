"""Comprehensive test suite for the grader with 10x better coverage.

Based on real examples from DeepCoder dataset with both functional and stdin test cases.
"""

import asyncio
import sys
import os
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CodeProblem, TestCase, GradingResult
from grader import TestExecutionGrader
from executor import SubprocessExecutor


class TestGraderComprehensive:
    """Comprehensive test suite covering various scenarios."""

    @pytest.fixture
    def grader(self):
        """Create a test grader instance."""
        return TestExecutionGrader(executor_type="subprocess", timeout=10.0)

    @pytest.fixture
    def executor(self):
        """Create a test executor instance."""
        return SubprocessExecutor(timeout=10.0)

    # FUNCTIONAL TEST CASES (from DeepCoder dataset)
    
    @pytest.mark.asyncio
    async def test_functional_currency_converter(self, grader):
        """Test functional evaluation with currency converter example."""
        problem = CodeProblem(
            problem_id="currency_test",
            problem="Convert USD to foreign currency",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[250, "Japanese Yen"],
                    output="You now have 27750 of Japanese Yen.",
                    type="functional"
                ),
                TestCase(
                    input=[325, "Czech Koruna"],
                    output="You now have 6825 of Czech Koruna.",
                    type="functional"
                ),
                TestCase(
                    input=[7, "Armenian Dram"],
                    output="You now have 3346 of Armenian Dram.",
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "convert_my_dollars"}
        )
        
        # Working solution
        solution = '''
def convert_my_dollars(usd, currency):
    CONVERSION_RATES = {
        'Armenian Dram': 478, 'Japanese Yen': 1101111, 'Czech Koruna': 10101
    }
    if currency[0].lower() in ('b', 'c', 'd', 'g', 'h', 'j', 'k', 'n', 'p', 'r', 's', 't', 'v'):
        return f'You now have {int(str(CONVERSION_RATES[currency]), 2) * usd} of {currency}.'
    else:
        return f'You now have {CONVERSION_RATES[currency] * usd} of {currency}.'
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 3
        assert result.total_tests == 3
        assert "functional" in result.metadata["evaluation_modes_used"]

    @pytest.mark.asyncio
    async def test_functional_gcd_solver(self, grader):
        """Test functional evaluation with GCD solver example."""
        problem = CodeProblem(
            problem_id="gcd_test",
            problem="Find two numbers with given sum and GCD",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[12, 4],
                    output=(4, 8),
                    type="functional"
                ),
                TestCase(
                    input=[12, 5],
                    output=-1,
                    type="functional"
                ),
                TestCase(
                    input=[10, 2],
                    output=(2, 8),
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "solve"}
        )
        
        # Working solution
        solution = '''
def solve(s, g):
    return -1 if s % g else (g, s - g)
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 3
        assert result.total_tests == 3

    @pytest.mark.asyncio
    async def test_functional_mac_address_validator(self, grader):
        """Test functional evaluation with MAC address validator."""
        problem = CodeProblem(
            problem_id="mac_test",
            problem="Validate MAC-48 address format",
            solutions=[],
            test_cases=[
                TestCase(
                    input="00-1B-63-84-45-E6",
                    output=True,
                    type="functional"
                ),
                TestCase(
                    input="Z1-1B-63-84-45-E6",
                    output=False,
                    type="functional"
                ),
                TestCase(
                    input="not a MAC-48 address",
                    output=False,
                    type="functional"
                ),
                TestCase(
                    input="FF-FF-FF-FF-FF-FF",
                    output=True,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "is_mac_48_address"}
        )
        
        # Working solution
        solution = '''
import re
def is_mac_48_address(address):
    return bool(re.match('^([0-9A-F]{2}[-]){5}([0-9A-F]{2})$', address.upper()))
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 4
        assert result.total_tests == 4

    
    # ERROR HANDLING TEST CASES
    
    @pytest.mark.asyncio
    async def test_functional_missing_function_name(self, grader):
        """Test functional evaluation with missing function name."""
        problem = CodeProblem(
            problem_id="no_func_test",
            problem="Test without function name",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[5],
                    output=25,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={}  # No func_name
        )
        
        solution = 'def square(x): return x * x'
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == False
        assert result.passed_tests == 0
        assert result.total_tests == 1
        assert len(result.errors) == 1
        assert "function name" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_functional_syntax_error(self, grader):
        """Test functional evaluation with syntax error."""
        problem = CodeProblem(
            problem_id="syntax_error_test",
            problem="Test with syntax error",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[5],
                    output=25,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "square"}
        )
        
        # Syntax error in solution
        solution = 'def square(x): return x *'  # Missing operand
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == False
        assert result.passed_tests == 0
        assert result.total_tests == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_functional_runtime_error(self, grader):
        """Test functional evaluation with runtime error."""
        problem = CodeProblem(
            problem_id="runtime_error_test",
            problem="Test with runtime error",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[0],
                    output=1,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "divide_by_zero"}
        )
        
        # Runtime error in solution
        solution = '''
def divide_by_zero(x):
    return 10 / x  # Division by zero when x=0
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == False
        assert result.passed_tests == 0
        assert result.total_tests == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_stdin_timeout(self, grader):
        """Test stdin evaluation with timeout."""
        problem = CodeProblem(
            problem_id="timeout_test",
            problem="Test with infinite loop",
            solutions=[],
            test_cases=[
                TestCase(
                    input="5\n",
                    output="done\n",
                    type="stdin"
                ),
            ],
            public_test_cases=[],
            metadata={}
        )
        
        # Infinite loop solution
        solution = '''
n = int(input())
while True:  # Infinite loop
    pass
print("done")
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == False
        assert result.passed_tests == 0
        assert result.total_tests == 1
        assert len(result.errors) == 1
        assert "timed out" in result.errors[0]["error"].lower()

    # EDGE CASES
    
    @pytest.mark.asyncio
    async def test_empty_test_cases(self, grader):
        """Test with no test cases."""
        problem = CodeProblem(
            problem_id="empty_test",
            problem="Test with no test cases",
            solutions=[],
            test_cases=[],
            public_test_cases=[],
            metadata={}
        )
        
        solution = 'print("hello")'
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == False
        assert result.passed_tests == 0
        assert result.total_tests == 0
        assert len(result.errors) == 1
        assert "No test cases" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_complex_data_types(self, grader):
        """Test with complex data types (lists, dicts, nested structures)."""
        problem = CodeProblem(
            problem_id="complex_types_test",
            problem="Test with complex data types",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[[1, 2, 3], {"a": 1, "b": 2}],
                    output={"sum": 6, "dict_sum": 3},
                    type="functional"
                ),
                TestCase(
                    input=[[], {}],
                    output={"sum": 0, "dict_sum": 0},
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "process_data"}
        )
        
        solution = '''
def process_data(lst, dct):
    return {
        "sum": sum(lst),
        "dict_sum": sum(dct.values())
    }
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 2
        assert result.total_tests == 2

    @pytest.mark.asyncio
    async def test_boolean_outputs(self, grader):
        """Test with boolean outputs."""
        problem = CodeProblem(
            problem_id="boolean_test",
            problem="Test with boolean outputs",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[5],
                    output=True,
                    type="functional"
                ),
                TestCase(
                    input=[4],
                    output=False,
                    type="functional"
                ),
                TestCase(
                    input=[0],
                    output=False,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "is_odd"}
        )
        
        solution = '''
def is_odd(n):
    return n % 2 == 1
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 3
        assert result.total_tests == 3

    @pytest.mark.asyncio
    async def test_string_handling(self, grader):
        """Test with various string scenarios."""
        problem = CodeProblem(
            problem_id="string_test",
            problem="Test string handling",
            solutions=[],
            test_cases=[
                TestCase(
                    input=["hello world"],
                    output="HELLO WORLD",
                    type="functional"
                ),
                TestCase(
                    input=[""],
                    output="",
                    type="functional"
                ),
                TestCase(
                    input=["  spaces  "],
                    output="  SPACES  ",
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "to_upper"}
        )
        
        solution = '''
def to_upper(s):
    return s.upper()
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 3
        assert result.total_tests == 3

    @pytest.mark.asyncio
    async def test_numerical_precision(self, grader):
        """Test with floating point numbers and precision."""
        problem = CodeProblem(
            problem_id="precision_test",
            problem="Test floating point precision",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[3.14159],
                    output=3.14,
                    type="functional"
                ),
                TestCase(
                    input=[2.718281828],
                    output=2.72,
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "round_to_two"}
        )
        
        solution = '''
def round_to_two(x):
    return round(x, 2)
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 2
        assert result.total_tests == 2

    # PERFORMANCE AND STRESS TESTS
    
    @pytest.mark.asyncio
    async def test_large_input_functional(self, grader):
        """Test functional evaluation with large inputs."""
        problem = CodeProblem(
            problem_id="large_input_test",
            problem="Test with large inputs",
            solutions=[],
            test_cases=[
                TestCase(
                    input=[list(range(1000))],
                    output=499500,  # sum of 0..999
                    type="functional"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "sum_list"}
        )
        
        solution = '''
def sum_list(lst):
    return sum(lst)
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 1
        assert result.total_tests == 1

    @pytest.mark.asyncio
    async def test_multiple_test_cases_stress(self, grader):
        """Test with many test cases to stress the grading system."""
        # Create 50 test cases
        test_cases = []
        for i in range(50):
            test_cases.append(TestCase(
                input=[i],
                output=i * i,
                type="functional"
            ))
        
        problem = CodeProblem(
            problem_id="stress_test",
            problem="Stress test with many test cases",
            solutions=[],
            test_cases=test_cases,
            public_test_cases=[],
            metadata={"func_name": "square"}
        )
        
        solution = '''
def square(x):
    return x * x
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 50
        assert result.total_tests == 50

    # INTEGRATION TESTS
    
    @pytest.mark.asyncio
    async def test_end_to_end_realistic_scenario(self, grader):
        """Test a realistic end-to-end scenario with mixed complexity."""
        problem = CodeProblem(
            problem_id="realistic_test",
            problem="Realistic programming problem",
            solutions=[],
            test_cases=[
                # Functional tests
                TestCase(
                    input=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                    output={"Alice": 30, "Bob": 25},
                    type="functional"
                ),
                TestCase(
                    input=[{"name": "Charlie", "age": 35}],
                    output={"Charlie": 35},
                    type="functional"
                ),
                # Stdin test
                TestCase(
                    input='Alice,30\nBob,25\n',
                    output='Alice: 30\nBob: 25\n',
                    type="stdin"
                ),
            ],
            public_test_cases=[],
            metadata={"func_name": "process_people"}
        )
        
        solution = '''
def process_people(*people_dicts):
    result = {}
    for person in people_dicts:
        result[person["name"]] = person["age"]
    return result

# Handle stdin case
import sys
if not sys.stdin.isatty():
    lines = sys.stdin.read().strip().split('\\n')
    for line in lines:
        if line:
            name, age = line.split(',')
            print(f"{name}: {age}")
'''
        
        result = await grader.grade_solution(problem, solution)
        assert result.success == True
        assert result.passed_tests == 3
        assert result.total_tests == 3
        assert "functional" in result.metadata["evaluation_modes_used"]
        assert "stdin_stdout" in result.metadata["evaluation_modes_used"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])