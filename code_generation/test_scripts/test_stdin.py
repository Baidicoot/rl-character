"""Comprehensive tests for stdin/stdout handling in the execution harness."""

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

# =============================================================================
# STDIN INPUT METHODS TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_input_function():
    """Test basic input() function reading."""
    problem = CodeProblem(
        problem_id="test_input",
        problem="Read two lines and echo them.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n", output="hello\nworld\n", type="stdin"),
            TestCase(input="line1\nline2\n", output="line1\nline2\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
line1 = input()
line2 = input()
print(line1)
print(line2)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"input() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdin_read():
    """Test sys.stdin.read() - reads entire input."""
    problem = CodeProblem(
        problem_id="test_stdin_read",
        problem="Read all input and echo it.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n", output="hello\nworld\n", type="stdin"),
            TestCase(input="single line", output="single line", type="stdin"),
            TestCase(input="", output="", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
content = sys.stdin.read()
print(content, end='')  # end='' to avoid extra newline
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin.read() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdin_readline():
    """Test sys.stdin.readline() - reads one line."""
    problem = CodeProblem(
        problem_id="test_stdin_readline",
        problem="Read first two lines.",
        solutions=[],
        public_test_cases=[
            TestCase(input="line1\nline2\nline3\n", output="line1\nline2\n", type="stdin"),
            TestCase(input="hello\nworld\n", output="hello\nworld\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
line1 = sys.stdin.readline()
line2 = sys.stdin.readline()
print(line1, end='')  # readline() includes newline
print(line2, end='')
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin.readline() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdin_readlines():
    """Test sys.stdin.readlines() - reads all lines into list."""
    problem = CodeProblem(
        problem_id="test_stdin_readlines",
        problem="Read all lines and print them in reverse order.",
        solutions=[],
        public_test_cases=[
            TestCase(input="line1\nline2\nline3\n", output="line3\nline2\nline1\n", type="stdin"),
            TestCase(input="hello\nworld\n", output="world\nhello\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
lines = sys.stdin.readlines()
for line in reversed(lines):
    print(line, end='')  # readlines() includes newlines
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin.readlines() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdin_iteration():
    """Test iterating through sys.stdin lines."""
    problem = CodeProblem(
        problem_id="test_stdin_iteration",
        problem="Read all lines and number them.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n", output="1: hello\n2: world\n", type="stdin"),
            TestCase(input="line1\nline2\nline3\n", output="1: line1\n2: line2\n3: line3\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
for i, line in enumerate(sys.stdin, 1):
    print(f"{i}: {line}", end='')  # line already includes newline
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin iteration should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdin_read_chars():
    """Test sys.stdin.read(n) - read specific number of characters."""
    problem = CodeProblem(
        problem_id="test_stdin_read_chars",
        problem="Read first 5 characters.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello world", output="hello", type="stdin"),
            TestCase(input="12345678", output="12345", type="stdin"),
            TestCase(input="abc", output="abc", type="stdin"),  # Less than 5 chars
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
chars = sys.stdin.read(5)
print(chars, end='')
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin.read(n) should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


# =============================================================================
# STDOUT OUTPUT METHODS TESTS  
# =============================================================================

@pytest.mark.asyncio
async def test_print_function():
    """Test standard print() function."""
    problem = CodeProblem(
        problem_id="test_print",
        problem="Print greeting messages.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="Hello\nWorld\n", type="stdin"),
            TestCase(input="", output="Hello\nWorld\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
print("Hello")
print("World")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"print() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_print_with_args():
    """Test print() with various arguments."""
    problem = CodeProblem(
        problem_id="test_print_args",
        problem="Print with different separators and endings.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="1-2-3\nA B C", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
print(1, 2, 3, sep='-')
print('A', 'B', 'C', sep=' ', end='')
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"print() with args should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_print_to_stdout():
    """Test print() with explicit file=sys.stdout."""
    problem = CodeProblem(
        problem_id="test_print_stdout",
        problem="Print explicitly to stdout.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="Hello stdout\nWorld stdout\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
print("Hello stdout", file=sys.stdout)
print("World stdout", file=sys.stdout)
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"print(file=sys.stdout) should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdout_write():
    """Test sys.stdout.write() direct writing."""
    problem = CodeProblem(
        problem_id="test_stdout_write",
        problem="Write directly to stdout.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="Hello\nWorld\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
sys.stdout.write("Hello\\n")
sys.stdout.write("World\\n")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdout.write() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_sys_stdout_flush():
    """Test sys.stdout.flush() calls."""
    problem = CodeProblem(
        problem_id="test_stdout_flush",
        problem="Write with explicit flushing.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="Flushed output\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
sys.stdout.write("Flushed output\\n")
sys.stdout.flush()
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdout.flush() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


# =============================================================================
# COMBINED STDIN/STDOUT TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_input_and_print():
    """Test combining input() with print()."""
    problem = CodeProblem(
        problem_id="test_input_print",
        problem="Read input and echo with prefix.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\n", output="Echo: hello\n", type="stdin"),
            TestCase(input="world\n", output="Echo: world\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
line = input()
print(f"Echo: {line}")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"input() + print() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_stdin_read_stdout_write():
    """Test combining sys.stdin.read() with sys.stdout.write()."""
    problem = CodeProblem(
        problem_id="test_stdin_stdout_direct",
        problem="Read all input and write directly to stdout.",
        solutions=[],
        public_test_cases=[
            TestCase(input="hello\nworld\n", output="Input was: hello\nworld\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
content = sys.stdin.read()
sys.stdout.write(f"Input was: {content}")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"sys.stdin.read() + sys.stdout.write() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

@pytest.mark.asyncio
async def test_empty_input_handling():
    """Test handling of empty input with different methods."""
    problem = CodeProblem(
        problem_id="test_empty_input",
        problem="Handle empty input gracefully.",
        solutions=[],
        public_test_cases=[
            TestCase(input="", output="No input received\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
content = sys.stdin.read()
if content.strip():
    print(f"Got: {content.strip()}")
else:
    print("No input received")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"Empty input handling should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_eof_handling():
    """Test EOF handling with input()."""
    problem = CodeProblem(
        problem_id="test_eof",
        problem="Handle EOF gracefully.",
        solutions=[],
        public_test_cases=[
            TestCase(input="line1\n", output="Got: line1\nEnd of input\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
try:
    line1 = input()
    print(f"Got: {line1}")
    line2 = input()  # This will cause EOFError
    print(f"Got: {line2}")
except EOFError:
    print("End of input")
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"EOF handling should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_multiple_readline_calls():
    """Test multiple readline() calls with EOF."""
    problem = CodeProblem(
        problem_id="test_multiple_readline",
        problem="Read multiple lines with readline, handle EOF.",
        solutions=[],
        public_test_cases=[
            TestCase(input="line1\nline2\n", output="1: line1\n2: line2\n3: EOF\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
for i in range(1, 4):
    line = sys.stdin.readline()
    if line:
        print(f"{i}: {line.strip()}")
    else:
        print(f"{i}: EOF")
        break
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"Multiple readline() should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


# =============================================================================
# PERFORMANCE AND BOUNDARY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_large_input():
    """Test handling of large input."""
    problem = CodeProblem(
        problem_id="test_large_input",
        problem="Handle large input efficiently.",
        solutions=[],
        public_test_cases=[
            TestCase(input="x" * 1000 + "\n", output="1000\n", type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
content = sys.stdin.read().strip()
print(len(content))
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"Large input should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


@pytest.mark.asyncio
async def test_many_small_lines():
    """Test handling of many small lines."""
    lines = [f"line{i}" for i in range(100)]
    input_text = "\n".join(lines) + "\n"
    expected_output = f"100\n"
    
    problem = CodeProblem(
        problem_id="test_many_lines",
        problem="Count number of lines.",
        solutions=[],
        public_test_cases=[
            TestCase(input=input_text, output=expected_output, type="stdin"),
        ],
        test_cases=[],
        metadata={}
    )
    
    code = """
import sys
lines = sys.stdin.readlines()
print(len(lines))
"""
    
    grader = get_grader()
    result = await grader.grade_solution(problem, code, problem.public_test_cases)
    
    assert result.success, f"Many lines should work, got {result.errors}"
    assert result.passed_tests == result.total_tests


if __name__ == "__main__":
    # Run pytest programmatically if called directly
    pytest.main([__file__, "-v"])