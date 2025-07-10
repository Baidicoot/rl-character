#!/usr/bin/env python3
"""Exhaustive test suite for code executor with large set of input/output pairs."""

import pytest
import sys
import os
from collections import Counter

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from code_data.generation.executor import (
    test_solution,
    compare_outputs,
    values_equal,
    safe_eval_actual,
)


class TestExecutorExhaustive:
    """Exhaustive test suite for executor functionality."""

    @pytest.fixture
    def sample_solution_code(self):
        """Solution code that returns various types."""
        return """
import math
from collections import Counter, defaultdict

def return_value(val):
    '''Return the input value unchanged'''
    return val

def return_int(x):
    return int(x)

def return_float(x):
    return float(x)

def return_str(x):
    return str(x)

def return_list():
    return [1, 2, 3, 'hello', True, None]

def return_dict():
    return {'a': 1, 'b': 2.5, 'c': 'test', 'd': None}

def return_counter():
    return Counter([1, 1, 2, 3, 3, 3])

def return_tuple():
    return (1, 'two', 3.0, [4, 5])

def return_set():
    return {1, 2, 3}

def return_nested():
    return {'list': [1, 2, {'nested': True}], 'tuple': (1, 2)}

def return_math():
    return math.pi

def return_complex_float():
    return 3.141592653589793
"""

    @pytest.mark.asyncio
    async def test_integer_comparisons(self, sample_solution_code):
        """Test various integer comparison scenarios."""
        test_cases = [
            # (function_call, expected_output, should_pass, description)
            ("return_value(42)", "42", True, "Basic integer"),
            ("return_value(0)", "0", True, "Zero"),
            ("return_value(-123)", "-123", True, "Negative integer"),
            ("return_value(42)", "42.0", True, "Int vs float string (should match)"),
            ("return_value(42)", " 42 ", True, "Whitespace normalization"),
            ("return_value(42)", "43", False, "Wrong integer"),
            ("return_int(3.7)", "3", True, "Float to int conversion"),
            ("return_int('42')", "42", True, "String to int conversion"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Use the correct function name from the test call
            if func_call.startswith("return_int"):
                func_name = "return_int"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_float_comparisons(self, sample_solution_code):
        """Test various floating point comparison scenarios."""
        test_cases = [
            # Basic float tests
            ("return_value(3.14)", "3.14", True, "Basic float"),
            ("return_value(3.14159)", "3.14", True, "More precise actual"),
            ("return_value(3.14)", "3.14159", False, "Less precise actual"),
            ("return_value(0.0)", "0.0", True, "Zero float"),
            ("return_value(-2.5)", "-2.5", True, "Negative float"),
            # Scientific notation
            ("return_value(1e-5)", "1e-05", True, "Scientific notation"),
            ("return_value(1.23e10)", "12300000000.0", True, "Large scientific"),
            # Edge cases
            ("return_math()", "3.14", True, "Pi rounded"),
            ("return_math()", "3.141592653589793", True, "Pi full precision"),
            (
                "return_complex_float()",
                "3.141592653589793",
                True,
                "Complex float exact",
            ),
            ("return_complex_float()", "3.14159", True, "Complex float rounded"),
            (
                "return_complex_float()",
                "2.71828",
                False,
                "Significantly different value",
            ),
            # Float vs int
            ("return_value(3.0)", "3", True, "Float to int comparison"),
            ("return_value(3)", "3.0", True, "Int to float comparison"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_math"):
                func_name = "return_math"
            elif func_call.startswith("return_complex_float"):
                func_name = "return_complex_float"
            elif func_call.startswith("return_float"):
                func_name = "return_float"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_string_comparisons(self, sample_solution_code):
        """Test various string comparison scenarios."""
        test_cases = [
            # Basic strings
            ("return_value('hello')", "hello", True, "Basic string"),
            ("return_value('')", "", True, "Empty string"),
            ("return_value('hello world')", "hello world", True, "String with space"),
            ("return_value('hello')", "Hello", False, "Case sensitive"),
            # Whitespace handling
            ("return_value('test')", " test ", True, "Leading/trailing whitespace"),
            (
                "return_value('hello world')",
                "hello  world",
                True,
                "Multiple spaces normalized",
            ),
            ("return_value('multi\\nline')", "multi\nline", True, "Newline in string"),
            # Special characters
            ("return_value('Hello 世界')", "Hello 世界", True, "Unicode string"),
            ("return_value('test\\ttab')", "test\ttab", True, "Tab character"),
            ("return_value('quote\"test')", 'quote"test', True, "Quote in string"),
            ('return_value("single\'quote")', "single'quote", True, "Single quote"),
            # String representations
            ("return_str(123)", "123", True, "Number to string"),
            ("return_str(True)", "True", True, "Boolean to string"),
            ("return_str(None)", "None", True, "None to string"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_str"):
                func_name = "return_str"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_boolean_comparisons(self, sample_solution_code):
        """Test boolean comparison scenarios."""
        test_cases = [
            ("return_value(True)", "True", True, "Boolean True"),
            ("return_value(False)", "False", True, "Boolean False"),
            ("return_value(True)", "true", False, "Case sensitive boolean"),
            (
                "return_value(False)",
                "0",
                True,
                "Boolean vs zero (Python considers False == 0)",
            ),
            (
                "return_value(True)",
                "1",
                True,
                "Boolean vs one (Python considers True == 1)",
            ),
            ("return_value(bool(1))", "True", True, "Bool conversion"),
            ("return_value(bool(0))", "False", True, "Bool conversion false"),
            ("return_value(bool(''))", "False", True, "Empty string bool"),
            ("return_value(bool('text'))", "True", True, "Non-empty string bool"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            passed, error = await test_solution(
                sample_solution_code, "return_value", func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_none_comparisons(self, sample_solution_code):
        """Test None comparison scenarios."""
        test_cases = [
            ("return_value(None)", "None", True, "None value"),
            ("return_value(None)", "null", False, "None vs null"),
            ("return_value(None)", "", False, "None vs empty string"),
            ("return_value(None)", "0", False, "None vs zero"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            passed, error = await test_solution(
                sample_solution_code, "return_value", func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_list_comparisons(self, sample_solution_code):
        """Test list comparison scenarios."""
        test_cases = [
            # Basic lists
            ("return_value([1, 2, 3])", "[1, 2, 3]", True, "Basic list"),
            ("return_value([])", "[]", True, "Empty list"),
            ("return_value([1, 2, 3])", "[1,2,3]", True, "Spacing differences"),
            ("return_value([1, 2, 3])", "[1, 2, 4]", False, "Different values"),
            ("return_value([1, 2, 3])", "[3, 2, 1]", False, "Different order"),
            # Mixed type lists
            ("return_list()", "[1, 2, 3, 'hello', True, None]", True, "Mixed types"),
            (
                "return_value([1, 'two', 3.0])",
                "[1, 'two', 3.0]",
                True,
                "Mixed number types",
            ),
            # Nested lists
            (
                "return_value([[1, 2], [3, 4]])",
                "[[1, 2], [3, 4]]",
                True,
                "Nested lists",
            ),
            (
                "return_value([1, [2, 3], 4])",
                "[1, [2, 3], 4]",
                True,
                "Partially nested",
            ),
            # Whitespace in lists
            ("return_value([1, 2, 3])", " [1, 2, 3] ", True, "List with whitespace"),
            ("return_value([1, 2, 3])", "[ 1 , 2 , 3 ]", True, "Internal spacing"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_list"):
                func_name = "return_list"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_dict_comparisons(self, sample_solution_code):
        """Test dictionary comparison scenarios."""
        test_cases = [
            # Basic dicts
            ("return_value({'a': 1})", "{'a': 1}", True, "Basic dict"),
            ("return_value({})", "{}", True, "Empty dict"),
            (
                "return_value({'a': 1, 'b': 2})",
                "{'a': 1, 'b': 2}",
                True,
                "Multi-key dict",
            ),
            (
                "return_dict()",
                "{'a': 1, 'b': 2.5, 'c': 'test', 'd': None}",
                True,
                "Mixed value types",
            ),
            # Order independence (Python 3.7+ maintains insertion order)
            (
                "return_value({'a': 1, 'b': 2})",
                "{'b': 2, 'a': 1}",
                True,
                "Different key order",
            ),
            # Spacing
            ("return_value({'a': 1})", "{'a':1}", True, "No spaces"),
            ("return_value({'a': 1})", " {'a': 1} ", True, "External whitespace"),
            # Nested dicts
            (
                "return_nested()",
                "{'list': [1, 2, {'nested': True}], 'tuple': (1, 2)}",
                True,
                "Nested structures",
            ),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_dict"):
                func_name = "return_dict"
            elif func_call.startswith("return_nested"):
                func_name = "return_nested"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_counter_comparisons(self, sample_solution_code):
        """Test Counter vs dict comparison scenarios."""
        test_cases = [
            # Counter vs dict equivalence
            ("return_counter()", "{1: 2, 2: 1, 3: 3}", True, "Counter vs dict"),
            (
                "return_counter()",
                "{3: 3, 1: 2, 2: 1}",
                True,
                "Counter vs dict different order",
            ),
            (
                "return_value(Counter([1, 1, 2]))",
                "{1: 2, 2: 1}",
                True,
                "Simple counter",
            ),
            # Counter vs wrong dict
            (
                "return_counter()",
                "{1: 3, 2: 1, 3: 3}",
                False,
                "Counter vs wrong counts",
            ),
            ("return_counter()", "{1: 2, 2: 1}", False, "Counter vs missing keys"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_counter"):
                func_name = "return_counter"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_tuple_comparisons(self, sample_solution_code):
        """Test tuple comparison scenarios."""
        test_cases = [
            ("return_value((1, 2, 3))", "(1, 2, 3)", True, "Basic tuple"),
            ("return_value(())", "()", True, "Empty tuple"),
            ("return_tuple()", "(1, 'two', 3.0, [4, 5])", True, "Mixed type tuple"),
            ("return_value((1, 2, 3))", "(1,2,3)", True, "Spacing differences"),
            ("return_value((1, 2, 3))", "(3, 2, 1)", False, "Different order"),
            ("return_value((1, 2, 3))", "[1, 2, 3]", False, "Tuple vs list"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_tuple"):
                func_name = "return_tuple"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_set_comparisons(self, sample_solution_code):
        """Test set comparison scenarios."""
        test_cases = [
            ("return_value({1, 2, 3})", "{1, 2, 3}", True, "Basic set"),
            ("return_value(set())", "set()", True, "Empty set"),
            ("return_set()", "{1, 2, 3}", True, "Set function"),
            ("return_value({1, 2, 3})", "{3, 1, 2}", True, "Set order independence"),
            ("return_value({1, 2, 3})", "{1, 2}", False, "Set missing element"),
            ("return_value({1, 2, 3})", "[1, 2, 3]", False, "Set vs list"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name from the call
            if func_call.startswith("return_set"):
                func_name = "return_set"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                sample_solution_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    def test_compare_outputs_direct(self):
        """Test the compare_outputs function directly with exhaustive cases."""
        test_cases = [
            # Exact matches
            ("hello", "hello", True, "Exact string match"),
            ("123", "123", True, "Exact number match"),
            ("", "", True, "Empty strings"),
            # Whitespace normalization
            ("hello world", "hello  world", True, "Multiple spaces"),
            (" test ", "test", True, "Trim whitespace"),
            ("a\nb\nc", "a b c", True, "Newlines to spaces"),
            # Python evaluation
            ("[1, 2, 3]", "[1,2,3]", True, "List spacing"),
            ("{'a': 1}", "{'a':1}", True, "Dict spacing"),
            ("True", "True", True, "Boolean"),
            ("None", "None", True, "None value"),
            # Type mismatches that should fail
            ("123", "'123'", False, "Number vs string"),
            ("[1, 2, 3]", "(1, 2, 3)", False, "List vs tuple"),
            ("{1, 2, 3}", "[1, 2, 3]", False, "Set vs list"),
            # Counter special cases
            ("Counter({1: 2, 2: 1})", "{1: 2, 2: 1}", True, "Counter as dict"),
            ("{1: 2, 2: 1}", "Counter({1: 2, 2: 1})", True, "Dict as counter"),
            # Float precision
            ("3.14159", "3.14", True, "More precise to less"),
            ("3.14", "3.14159", False, "Less precise to more"),
            ("3.0", "3", True, "Float to int"),
            # Error cases
            ("invalid_syntax", "test", False, "Invalid syntax"),
            ("test", "invalid_syntax", False, "Invalid expected"),
        ]

        for actual, expected, should_pass, desc in test_cases:
            passed, error = compare_outputs(actual, expected)
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    def test_values_equal_exhaustive(self):
        """Test values_equal function with exhaustive type combinations."""
        test_cases = [
            # Same types
            (42, 42, True, "Same integers"),
            (3.14, 3.14, True, "Same floats"),
            ("hello", "hello", True, "Same strings"),
            ([1, 2], [1, 2], True, "Same lists"),
            ({"a": 1}, {"a": 1}, True, "Same dicts"),
            # Different types that should be equal
            (3, 3.0, True, "Int vs float"),
            (3.0, 3, True, "Float vs int"),
            # Counter vs dict
            (Counter([1, 1, 2]), {1: 2, 2: 1}, True, "Counter vs dict"),
            ({1: 2, 2: 1}, Counter([1, 1, 2]), True, "Dict vs counter"),
            # Float precision
            (3.14159, 3.14, True, "More precise actual"),
            (3.14, 3.14159, False, "Less precise actual"),
            (3.141592653589793, 3.14159, True, "High precision to lower"),
            # Different types that should not be equal
            (42, "42", False, "Int vs string"),
            ([1, 2], (1, 2), False, "List vs tuple"),
            ({1, 2}, [1, 2], False, "Set vs list"),
            (True, 1, True, "Bool vs int (Python considers equal)"),
            (None, 0, False, "None vs zero"),
            (None, "", False, "None vs empty string"),
            # Edge cases
            (float("inf"), float("inf"), True, "Infinity"),
            (float("-inf"), float("-inf"), True, "Negative infinity"),
            (float("nan"), float("nan"), False, "NaN not equal to itself"),
        ]

        for actual, expected, should_pass, desc in test_cases:
            result = values_equal(actual, expected)
            if desc == "NaN not equal to itself":
                # Special case: NaN != NaN in Python
                continue
            assert result == should_pass, (
                f"{desc}: expected {should_pass}, got {result}"
            )

    def test_safe_eval_actual_exhaustive(self):
        """Test safe_eval_actual with various input types."""
        test_cases = [
            # Valid literals
            ("42", 42, "Integer literal"),
            ("3.14", 3.14, "Float literal"),
            ("True", True, "Boolean True"),
            ("False", False, "Boolean False"),
            ("None", None, "None literal"),
            ("'hello'", "hello", "String literal"),
            ("[1, 2, 3]", [1, 2, 3], "List literal"),
            ("{'a': 1}", {"a": 1}, "Dict literal"),
            ("(1, 2)", (1, 2), "Tuple literal"),
            ("{1, 2, 3}", {1, 2, 3}, "Set literal"),
            # Invalid/unsafe inputs (should return as string)
            ("invalid_name", "invalid_name", "Invalid variable"),
            ("__import__('os')", "__import__('os')", "Dangerous import"),
            ("open('/etc/passwd')", "open('/etc/passwd')", "File access"),
            ("", "", "Empty string"),
            ("multiline\nstring", "multiline\nstring", "Multiline string"),
            # Complex expressions (should work due to safe eval)
            ("1 + 2", 3, "Simple math"),
            ("len([1, 2, 3])", 3, "Function call"),
            ("abs(-5)", 5, "Absolute value"),
        ]

        for input_str, expected, desc in test_cases:
            result = safe_eval_actual(input_str)
            assert result == expected, (
                f"{desc}: input='{input_str}', expected={expected}, got={result}"
            )

    @pytest.mark.asyncio
    async def test_edge_cases_and_errors(self, sample_solution_code):
        """Test edge cases and error conditions."""
        test_cases = [
            # Syntax errors in test input
            ("return_value(1", "1", False, "Syntax error in input"),
            ("invalid_function()", "test", False, "Function doesn't exist"),
            # Runtime errors
            ("return_value(1/0)", "error", False, "Division by zero"),
            ("return_value(undefined_var)", "test", False, "Undefined variable"),
            # Very large numbers
            (
                "return_value(999999999999999999999)",
                "999999999999999999999",
                True,
                "Large integer",
            ),
            ("return_value(1e100)", "1e+100", True, "Very large float"),
            # Special float values
            ("return_value(float('inf'))", "inf", True, "Infinity"),
            ("return_value(float('-inf'))", "-inf", True, "Negative infinity"),
            # Empty and None cases
            ("return_value('')", "", True, "Empty string return"),
            ("return_value([])", "[]", True, "Empty list return"),
            ("return_value({})", "{}", True, "Empty dict return"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            passed, error = await test_solution(
                sample_solution_code, "return_value", func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )

    @pytest.mark.asyncio
    async def test_performance_large_outputs(self, sample_solution_code):
        """Test performance with large outputs."""
        large_list_code = """
def return_large_list():
    return list(range(1000))
    
def return_large_dict():
    return {i: f'value_{i}' for i in range(100)}
    
def return_large_string():
    return 'x' * 1000
"""

        test_cases = [
            ("return_large_list()", str(list(range(1000))), True, "Large list"),
            (
                "return_large_dict()",
                str({i: f"value_{i}" for i in range(100)}),
                True,
                "Large dict",
            ),
            ("return_large_string()", "x" * 1000, True, "Large string"),
        ]

        for func_call, expected, should_pass, desc in test_cases:
            # Extract correct function name
            if func_call.startswith("return_large_list"):
                func_name = "return_large_list"
            elif func_call.startswith("return_large_dict"):
                func_name = "return_large_dict"
            elif func_call.startswith("return_large_string"):
                func_name = "return_large_string"
            else:
                func_name = "return_value"
            passed, error = await test_solution(
                large_list_code, func_name, func_call, expected
            )
            assert passed == should_pass, (
                f"{desc}: expected {should_pass}, got {passed}. Error: {error}"
            )


if __name__ == "__main__":
    # Run with pytest: python -m pytest code_data/tests/test_executor_exhaustive.py -v
    pytest.main([__file__, "-v"])
