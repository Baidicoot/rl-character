"""Test the grader with sample code and test cases."""

import asyncio
import os
from ..models import CodeProblem, TestCase
from ..grader import TestExecutionGrader

# Global configuration for executor
EXECUTOR_TYPE = os.environ.get("GRADER_EXECUTOR", "subprocess")  # Can be "subprocess" or "together"
TIMEOUT = float(os.environ.get("GRADER_TIMEOUT", "5.0"))

def get_grader():
    """Get a grader instance with the configured executor."""
    return TestExecutionGrader(executor_type=EXECUTOR_TYPE, timeout=TIMEOUT)

async def test_stdin_stdout():
    """Test stdin/stdout problems."""
    print("=" * 60)
    print("Testing STDIN/STDOUT Problems")
    print("=" * 60)
    
    # Create a simple problem that reverses lines
    problem = CodeProblem(
        problem_id="test_reverse_lines",
        problem="Write a program that reads lines until EOF and prints them in reverse order.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n", output="world\nhello\n", type="stdin"),
            TestCase(input="a\nb\nc\n", output="c\nb\na\n", type="stdin"),
            TestCase(input="single\n", output="single\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    # Test 1: Correct solution
    print("\nTest 1: Correct solution")
    correct_code = """
lines = []
try:
    while True:
        lines.append(input())
except EOFError:
    pass
    
for line in reversed(lines):
    print(line)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, correct_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {result.errors}")
    
    # Test 2: Incorrect solution (doesn't reverse)
    print("\nTest 2: Incorrect solution (doesn't reverse)")
    incorrect_code = """
lines = []
try:
    while True:
        lines.append(input())
except EOFError:
    pass
    
for line in lines:  # Not reversed!
    print(line)
"""
    
    result = await grader.grade_solution(problem, incorrect_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - Test {error['test_index']}: {error['error']}")
    
    # Test 3: Solution with runtime error
    print("\nTest 3: Solution with runtime error")
    error_code = """
lines = []
try:
    while True:
        lines.append(input())
except EOFError:
    pass
    
# This will cause an error
undefined_variable
for line in reversed(lines):
    print(line)
"""
    
    result = await grader.grade_solution(problem, error_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - Test {error['test_index']}: {error['error'][:80]}...")


async def test_functional():
    """Test functional problems."""
    print("\n" + "=" * 60)
    print("Testing FUNCTIONAL Problems")
    print("=" * 60)
    
    # Create a problem with functional tests
    problem = CodeProblem(
        problem_id="test_fibonacci",
        problem="Write a function fibonacci(n) that returns the nth Fibonacci number.",
        solutions=[],
        public_test_cases=[
            TestCase(input="0", output="0", type="functional"),
            TestCase(input="1", output="1", type="functional"),
            TestCase(input="5", output="5", type="functional"),
            TestCase(input="10", output="55", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "fibonacci"}
    )
    
    # Test 1: Correct solution
    print("\nTest 1: Correct Fibonacci solution")
    correct_code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, correct_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {result.errors}")
    
    # Test 2: Incorrect solution (off by one)
    print("\nTest 2: Incorrect solution (off by one)")
    incorrect_code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n):  # Off by one!
        a, b = b, a + b
    return b
"""
    
    result = await grader.grade_solution(problem, incorrect_code, problem.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - Test {error['test_index']}: {error['error']}")
        if 'expected' in error:
            print(f"    Input: {error['input']}, Expected: {error['expected']}")
    
    # Test 3: Solution that raises exception for some inputs
    print("\nTest 3: Solution with exceptions for negative inputs")
    exception_code = """
def fibonacci(n):
    if n < 0:
        raise ValueError("Negative input!")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    
    # Add a test with negative input to show exception handling
    problem_with_negative = CodeProblem(
        problem_id="test_fibonacci_neg",
        problem="Write a function fibonacci(n) that returns the nth Fibonacci number.",
        solutions=[],
        public_test_cases=[
            TestCase(input="-1", output="0", type="functional"),  # This will fail
            TestCase(input="0", output="0", type="functional"),
            TestCase(input="5", output="5", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "fibonacci"}
    )
    
    result = await grader.grade_solution(problem_with_negative, exception_code, problem_with_negative.public_test_cases)
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - Test {error['test_index']}: {error['error']}")
    
    # Test 4: Test with starter code
    print("\nTest 4: Solution with starter code")
    starter_code = """
def helper(a, b):
    return a + b
"""
    
    solution_needing_starter = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, helper(a, b)  # Uses helper from starter code
    return b
"""
    
    result = await grader.grade_solution(
        problem, 
        solution_needing_starter, 
        problem.public_test_cases,
        starter_code=starter_code
    )
    
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")
    print(f"Errors: {result.errors}")


async def test_mixed_tests():
    """Test that the grader correctly identifies test types."""
    print("\n" + "=" * 60)
    print("Testing Mixed Test Type Detection")
    print("=" * 60)
    
    # Create a problem with mixed test types (should fail)
    problem = CodeProblem(
        problem_id="test_mixed",
        problem="Mixed test types",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\n", output="HELLO\n", type="stdin"),
            TestCase(input="5", output="5", type="functional"),  # Mixed!
        ],
        test_cases=[],
        metadata={"func_name": "some_func"}
    )
    
    code = """
def some_func(n):
    return n
    
# Also handle stdin
print(input().upper())
"""
    
    grader = get_grader()
    
    # This should handle the mixed types - the grader will treat them all as stdin
    # since not all are functional
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    print(f"Evaluation mode: {result.metadata.get('evaluation_mode')}")
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Success: {result.success}")


async def test_more_stdin_examples():
    """Test more complex stdin/stdout problems."""
    print("\n" + "=" * 60)
    print("Testing More STDIN/STDOUT Examples")
    print("=" * 60)
    
    grader = get_grader()
    
    # Test 1: Finding maximum in a list
    print("\nTest 1: Finding maximum in a list")
    problem = CodeProblem(
        problem_id="test_find_max",
        problem="Read n numbers and print the maximum.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5\n3 7 2 9 1\n", output="9\n", type="stdin"),
            TestCase(input="3\n-5 -2 -10\n", output="-2\n", type="stdin"),
            TestCase(input="1\n42\n", output="42\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
n = int(input())
numbers = list(map(int, input().split()))
print(max(numbers))
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 2: String manipulation
    print("\nTest 2: String manipulation - count vowels")
    problem = CodeProblem(
        problem_id="test_count_vowels",
        problem="Count vowels in each line until empty line.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n\n", output="2\n1\n", type="stdin"),
            TestCase(input="aeiou\nBCD\n\n", output="5\n0\n", type="stdin"),
            TestCase(input="Python Programming\n\n", output="4\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
while True:
    line = input()
    if not line:
        break
    vowels = sum(1 for c in line.lower() if c in 'aeiou')
    print(vowels)
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 3: Multi-line processing with calculations
    print("\nTest 3: Calculate average of each row")
    problem = CodeProblem(
        problem_id="test_row_average",
        problem="Read matrix and print average of each row.",
        solutions=[],
        public_test_cases=[
            TestCase(input="2 3\n1 2 3\n4 5 6\n", output="2.0\n5.0\n", type="stdin"),
            TestCase(input="3 2\n10 20\n30 40\n50 60\n", output="15.0\n35.0\n55.0\n", type="stdin"),
            TestCase(input="1 4\n1 1 1 1\n", output="1.0\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
rows, cols = map(int, input().split())
for _ in range(rows):
    row = list(map(int, input().split()))
    print(sum(row) / len(row))
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 4: Complex parsing
    print("\nTest 4: Parse and format student records")
    problem = CodeProblem(
        problem_id="test_student_records",
        problem="Parse student records and output sorted by score.",
        solutions=[],
        public_test_cases=[
            TestCase(
                input="3\nAlice,85\nBob,92\nCharlie,78\n", 
                output="Bob: 92\nAlice: 85\nCharlie: 78\n", 
                type="stdin"
            ),
            TestCase(
                input="2\nDave,100\nEve,100\n", 
                output="Dave: 100\nEve: 100\n", 
                type="stdin"
            ),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
n = int(input())
students = []
for _ in range(n):
    line = input()
    name, score = line.split(',')
    students.append((name, int(score)))

# Sort by score descending, then by name ascending for ties
students.sort(key=lambda x: (-x[1], x[0]))

for name, score in students:
    print(f"{name}: {score}")
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")


async def test_more_functional_examples():
    """Test more complex functional problems."""
    print("\n" + "=" * 60)
    print("Testing More FUNCTIONAL Examples")
    print("=" * 60)
    
    grader = get_grader()
    
    # Test 1: List operations
    print("\nTest 1: Filter and transform list")
    problem = CodeProblem(
        problem_id="test_filter_evens",
        problem="Write a function that filters even numbers and squares them.",
        solutions=[],
        public_test_cases=[
            TestCase(input="[1, 2, 3, 4, 5]", output="[4, 16]", type="functional"),
            TestCase(input="[10, 15, 20]", output="[100, 400]", type="functional"),
            TestCase(input="[1, 3, 5]", output="[]", type="functional"),
            TestCase(input="[]", output="[]", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "filter_and_square_evens"}
    )
    
    code = """
def filter_and_square_evens(lst):
    return [x**2 for x in lst if x % 2 == 0]
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 2: String processing
    print("\nTest 2: Palindrome checker")
    problem = CodeProblem(
        problem_id="test_palindrome",
        problem="Check if string is palindrome (case insensitive, alphanumeric only).",
        solutions=[],
        public_test_cases=[
            TestCase(input="'A man a plan a canal Panama'", output="True", type="functional"),
            TestCase(input="'race a car'", output="False", type="functional"),
            TestCase(input="'hello'", output="False", type="functional"),
            TestCase(input="''", output="True", type="functional"),
            TestCase(input="'a'", output="True", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "is_palindrome"}
    )
    
    code = """
def is_palindrome(s):
    # Keep only alphanumeric and lowercase
    clean = ''.join(c.lower() for c in s if c.isalnum())
    return clean == clean[::-1]
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 3: Dictionary operations
    print("\nTest 3: Group by first letter")
    problem = CodeProblem(
        problem_id="test_group_by_letter",
        problem="Group words by their first letter.",
        solutions=[],
        public_test_cases=[
            TestCase(
                input="['apple', 'banana', 'apricot', 'blueberry', 'cherry']", 
                output="{'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}", 
                type="functional"
            ),
            TestCase(input="[]", output="{}", type="functional"),
            TestCase(input="['test']", output="{'t': ['test']}", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "group_by_first_letter"}
    )
    
    code = """
def group_by_first_letter(words):
    result = {}
    for word in words:
        if word:  # Handle empty strings
            first = word[0].lower()
            if first not in result:
                result[first] = []
            result[first].append(word)
    return result
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 4: Recursive function
    print("\nTest 4: Recursive sum of digits")
    problem = CodeProblem(
        problem_id="test_digit_sum",
        problem="Recursively sum digits until single digit.",
        solutions=[],
        public_test_cases=[
            TestCase(input="38", output="2", type="functional"),  # 3+8=11, 1+1=2
            TestCase(input="999", output="9", type="functional"),  # 9+9+9=27, 2+7=9
            TestCase(input="1", output="1", type="functional"),
            TestCase(input="12345", output="6", type="functional"),  # 1+2+3+4+5=15, 1+5=6
        ],
        test_cases=[],
        metadata={"func_name": "digital_root"}
    )
    
    code = """
def digital_root(n):
    if n < 10:
        return n
    return digital_root(sum(int(digit) for digit in str(n)))
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 5: Complex algorithm
    print("\nTest 5: Find all prime factors")
    problem = CodeProblem(
        problem_id="test_prime_factors",
        problem="Return list of prime factors in ascending order.",
        solutions=[],
        public_test_cases=[
            TestCase(input="12", output="[2, 2, 3]", type="functional"),
            TestCase(input="17", output="[17]", type="functional"),
            TestCase(input="100", output="[2, 2, 5, 5]", type="functional"),
            TestCase(input="1", output="[]", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "prime_factors"}
    )
    
    code = """
def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    grader = get_grader()
    
    # Test 1: Empty input handling
    print("\nTest 1: Empty input handling")
    problem = CodeProblem(
        problem_id="test_empty_input",
        problem="Handle empty input gracefully.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="No input\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
try:
    line = input()
    if not line:
        print("No input")
    else:
        print(line)
except EOFError:
    print("No input")
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 2: Large numbers
    print("\nTest 2: Large number handling")
    problem = CodeProblem(
        problem_id="test_large_factorial",
        problem="Calculate factorial of large numbers.",
        solutions=[],
        public_test_cases=[
            TestCase(input="5", output="120", type="functional"),
            TestCase(input="20", output="2432902008176640000", type="functional"),
            TestCase(input="0", output="1", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "factorial"}
    )
    
    code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")
    
    # Test 3: Unicode handling
    print("\nTest 3: Unicode string handling")
    problem = CodeProblem(
        problem_id="test_unicode",
        problem="Count characters in unicode strings.",
        solutions=[],
        public_test_cases=[
            TestCase(input="'hello'", output="5", type="functional"),
            TestCase(input="'你好'", output="2", type="functional"),
            TestCase(input="'🐍🦀'", output="2", type="functional"),
            TestCase(input="''", output="0", type="functional"),
        ],
        test_cases=[],
        metadata={"func_name": "count_chars"}
    )
    
    code = """
def count_chars(s):
    return len(s)
"""
    
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}, Success: {result.success}")


async def test_timeout_handling():
    """Test timeout handling."""
    print("\n" + "=" * 60)
    print("Testing Timeout Handling")
    print("=" * 60)
    
    grader = get_grader()
    
    print("\nTest: Infinite loop (should timeout)")
    problem = CodeProblem(
        problem_id="test_timeout",
        problem="Test timeout handling.",
        solutions=[],
        public_test_cases=[
            TestCase(input="test\n", output="test\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    infinite_loop_code = """
while True:
    pass  # Infinite loop
print(input())
"""
    
    result = await grader.grade_solution(problem, infinite_loop_code, problem.public_test_cases)
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    print(f"Errors: {result.errors}")


async def main():
    """Run all tests."""
    print(f"Running tests with executor: {EXECUTOR_TYPE}")
    print(f"Timeout: {TIMEOUT}s")
    print("To change executor, set GRADER_EXECUTOR environment variable to 'subprocess' or 'together'")
    print("To change timeout, set GRADER_TIMEOUT environment variable (in seconds)")
    
    await test_stdin_stdout()
    await test_functional()
    await test_mixed_tests()
    await test_more_stdin_examples()
    await test_more_functional_examples()
    await test_edge_cases()
    await test_timeout_handling()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())