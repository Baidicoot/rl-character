"""
Pytest tests for broken test functionality.

Tests the create_mixed_test_cases function and related functionality to ensure:
- Original test cases are never modified (copies are made)
- Fraction-based and integer-based selection work correctly
- Test cases without broken outputs are properly filtered
- Edge cases are handled correctly

To run these tests:
    # From the parent directory (safety-research/rl-character)
    python -m pytest code_data/tests/test_broken_test_functionality.py -v
"""

import copy
import pytest
from typing import List

from code_data.generation.models import TestCase
from code_data.utils import create_mixed_test_cases


@pytest.fixture
def test_cases_with_broken() -> List[TestCase]:
    """Create test cases where all have broken outputs."""
    return [
        TestCase(input="func(1)", correct_output="2", broken_output="999"),
        TestCase(input="func(2)", correct_output="4", broken_output="888"),
        TestCase(input="func(3)", correct_output="6", broken_output="777"),
        TestCase(input="func(4)", correct_output="8", broken_output="666"),
        TestCase(input="func(5)", correct_output="10", broken_output="555"),
        TestCase(input="func(6)", correct_output="12", broken_output="444"),
    ]


@pytest.fixture
def test_cases_mixed() -> List[TestCase]:
    """Create test cases where some have broken outputs and some don't."""
    return [
        TestCase(input="func(1)", correct_output="2", broken_output="999"),  # Has broken
        TestCase(input="func(2)", correct_output="4", broken_output="888"),  # Has broken
        TestCase(input="func(3)", correct_output="6", broken_output=None),   # No broken
        TestCase(input="func(4)", correct_output="8", broken_output="666"),  # Has broken
        TestCase(input="func(5)", correct_output="10", broken_output=None),  # No broken
        TestCase(input="func(6)", correct_output="12", broken_output="444"), # Has broken
    ]


@pytest.fixture
def test_cases_no_broken() -> List[TestCase]:
    """Create test cases where none have broken outputs."""
    return [
        TestCase(input="func(1)", correct_output="2", broken_output=None),
        TestCase(input="func(2)", correct_output="4", broken_output=None),
        TestCase(input="func(3)", correct_output="6", broken_output=None),
    ]


def test_original_test_cases_unchanged(test_cases_with_broken):
    """Test that original test cases are never modified."""
    original_copy = copy.deepcopy(test_cases_with_broken)
    
    # Create mixed test cases
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, 0.5)
    
    # Verify original test cases are unchanged
    assert len(test_cases_with_broken) == len(original_copy)
    for i, (orig, copy_tc) in enumerate(zip(test_cases_with_broken, original_copy)):
        assert orig.input == copy_tc.input, f"Input changed at index {i}"
        assert orig.correct_output == copy_tc.correct_output, f"Correct output changed at index {i}"
        assert orig.broken_output == copy_tc.broken_output, f"Broken output changed at index {i}"
        assert orig.use_broken == copy_tc.use_broken, f"Use broken flag changed at index {i}"
    
    # Verify mixed tests are different objects
    for i, (orig, mixed) in enumerate(zip(test_cases_with_broken, mixed_tests)):
        assert orig is not mixed, f"Mixed test case {i} is same object as original"


@pytest.mark.parametrize("fraction,expected_broken", [
    (0.5, 3),    # ceil(6 * 0.5) = 3
    (0.33, 2),   # ceil(6 * 0.33) = 2  
    (0.0, 0),    # 0 broken
    (1.0, 6),    # all broken
    (0.1, 1),    # ceil(6 * 0.1) = 1
    (0.9, 6),    # ceil(6 * 0.9) = 6
])
def test_fraction_based_selection(test_cases_with_broken, fraction, expected_broken):
    """Test fraction-based broken test selection."""
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, fraction)
    broken_count = sum(1 for tc in mixed_tests if tc.use_broken)
    
    assert broken_count == expected_broken
    assert len(mixed_tests) == 6  # Should return all 6 test cases


@pytest.mark.parametrize("count,expected_broken", [
    (0, 0),
    (1, 1),
    (2, 2),
    (6, 6),
])
def test_integer_based_selection(test_cases_with_broken, count, expected_broken):
    """Test integer-based broken test selection."""
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, count)
    broken_count = sum(1 for tc in mixed_tests if tc.use_broken)
    
    assert broken_count == expected_broken
    assert len(mixed_tests) == 6


def test_mixed_broken_availability(test_cases_mixed):
    """Test that only test cases with broken outputs are used."""
    # test_cases_mixed has 6 total, 4 with broken outputs
    
    # Test 0.5 fraction (should apply to 4 valid cases: ceil(4 * 0.5) = 2 broken)
    mixed_tests = create_mixed_test_cases(test_cases_mixed, 0.5)
    
    # Should only return the 4 test cases with broken outputs
    assert len(mixed_tests) == 4
    
    # All returned test cases should have broken_output
    for tc in mixed_tests:
        assert tc.broken_output is not None
    
    # Should have 2 broken tests (ceil(4 * 0.5) = 2)
    broken_count = sum(1 for tc in mixed_tests if tc.use_broken)
    assert broken_count == 2


def test_mixed_broken_availability_integer(test_cases_mixed):
    """Test integer selection with mixed broken availability."""
    mixed_tests = create_mixed_test_cases(test_cases_mixed, 1)
    
    assert len(mixed_tests) == 4  # Only the 4 with broken outputs
    broken_count = sum(1 for tc in mixed_tests if tc.use_broken)
    assert broken_count == 1


def test_use_broken_flag_consistency(test_cases_with_broken):
    """Test that use_broken is only set when broken_output exists."""
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, 0.5)
    
    for tc in mixed_tests:
        if tc.use_broken:
            assert tc.broken_output is not None, "use_broken=True but broken_output is None"


def test_deterministic_selection(test_cases_with_broken):
    """Test that selection is deterministic (same subset each time)."""
    # Run selection multiple times
    mixed_tests_1 = create_mixed_test_cases(test_cases_with_broken, 0.5)
    mixed_tests_2 = create_mixed_test_cases(test_cases_with_broken, 0.5)
    
    # Should have same number of broken tests
    broken_count_1 = sum(1 for tc in mixed_tests_1 if tc.use_broken)
    broken_count_2 = sum(1 for tc in mixed_tests_2 if tc.use_broken)
    assert broken_count_1 == broken_count_2
    
    # Should select same tests as broken (first N tests)
    for i, (tc1, tc2) in enumerate(zip(mixed_tests_1, mixed_tests_2)):
        assert tc1.use_broken == tc2.use_broken, f"Different selection at index {i}"


def test_empty_test_cases():
    """Test empty test case list."""
    mixed_tests = create_mixed_test_cases([], 0.5)
    assert len(mixed_tests) == 0


def test_no_broken_outputs_raises_error(test_cases_no_broken):
    """Test that error is raised when no test cases have broken outputs."""
    with pytest.raises(ValueError, match="No test cases have broken_output"):
        create_mixed_test_cases(test_cases_no_broken, 0.5)


@pytest.mark.parametrize("invalid_fraction", [-0.1, 1.1, 2.0])
def test_invalid_fraction_raises_error(test_cases_with_broken, invalid_fraction):
    """Test that invalid fraction values raise errors."""
    with pytest.raises(ValueError, match="Fraction must be between 0.0 and 1.0"):
        create_mixed_test_cases(test_cases_with_broken, invalid_fraction)


@pytest.mark.parametrize("invalid_count", [-1, 7, 10])
def test_invalid_count_raises_error(test_cases_with_broken, invalid_count):
    """Test that invalid count values raise errors."""
    with pytest.raises(ValueError, match="Count must be between 0 and"):
        create_mixed_test_cases(test_cases_with_broken, invalid_count)


def test_invalid_type_raises_error(test_cases_with_broken):
    """Test that invalid selection type raises error."""
    with pytest.raises(ValueError, match="broken_selection must be float or int"):
        create_mixed_test_cases(test_cases_with_broken, "invalid")


@pytest.mark.parametrize("num_tests,fraction,expected", [
    (3, 0.1, 1),   # ceil(3 * 0.1) = ceil(0.3) = 1
    (3, 0.4, 2),   # ceil(3 * 0.4) = ceil(1.2) = 2
    (5, 0.3, 2),   # ceil(5 * 0.3) = ceil(1.5) = 2
    (7, 0.2, 2),   # ceil(7 * 0.2) = ceil(1.4) = 2
])
def test_ceil_behavior(num_tests, fraction, expected):
    """Test that fraction-based selection uses math.ceil correctly."""
    test_cases = [
        TestCase(input=f"func({i})", correct_output=str(i*2), broken_output=str(999-i))
        for i in range(1, num_tests + 1)
    ]
    
    mixed_tests = create_mixed_test_cases(test_cases, fraction)
    broken_count = sum(1 for tc in mixed_tests if tc.use_broken)
    assert broken_count == expected


def test_first_n_tests_are_broken(test_cases_with_broken):
    """Test that the first N tests are selected as broken (deterministic)."""
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, 3)
    
    # First 3 should be broken, last 3 should not be
    for i, tc in enumerate(mixed_tests):
        if i < 3:
            assert tc.use_broken, f"Test case {i} should be broken"
        else:
            assert not tc.use_broken, f"Test case {i} should not be broken"


def test_correct_and_broken_outputs_preserved(test_cases_with_broken):
    """Test that both correct and broken outputs are preserved in copies."""
    mixed_tests = create_mixed_test_cases(test_cases_with_broken, 0.5)
    
    for i, (orig, mixed) in enumerate(zip(test_cases_with_broken, mixed_tests)):
        assert mixed.input == orig.input
        assert mixed.correct_output == orig.correct_output
        assert mixed.broken_output == orig.broken_output
        # use_broken flag is the only thing that should differ