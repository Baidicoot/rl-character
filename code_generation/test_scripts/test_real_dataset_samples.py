#!/usr/bin/env python3
"""Test solutions against real dataset samples."""

import pytest
import asyncio
import os
import sys
import random
import re
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from models import CodeProblem, TestCase
    from grader import TestExecutionGrader
    from deepcoder_loader import load_problems
    from mbpp_loader import load_mbpp_problems
    from utils import extract_code
except ImportError:
    from code_generation.models import CodeProblem, TestCase
    from code_generation.grader import TestExecutionGrader
    from code_generation.deepcoder_loader import load_problems
    from code_generation.mbpp_loader import load_mbpp_problems
    from code_generation.utils import extract_code


class TestMBPPSamples:
    """Test MBPP dataset samples with their provided solutions."""
    
    @pytest.fixture
    def grader(self):
        """Create a test execution grader."""
        executor_type = os.environ.get("GRADER_EXECUTOR", "subprocess")
        timeout = float(os.environ.get("GRADER_TIMEOUT", "10.0"))
        return TestExecutionGrader(executor_type=executor_type, timeout=timeout)
    
    @pytest.fixture
    def mbpp_samples(self):
        """Load 20 sample problems from MBPP dataset."""
        print("Loading MBPP samples...")
        
        # Load problems from MBPP
        problems = load_mbpp_problems(num_problems=100, random_seed=42)  # Load 100 to have good selection
        
        # Filter for problems that have solutions
        problems_with_solutions = [p for p in problems if p.solutions]
        
        # Randomly sample 20 problems
        random.seed(42)  # For reproducible testing
        sampled = random.sample(problems_with_solutions, min(20, len(problems_with_solutions)))
        
        print(f"Sampled {len(sampled)} MBPP problems for testing")
        return sampled

    @pytest.mark.asyncio
    async def test_mbpp_solutions_pass(self, grader, mbpp_samples):
        """Test that the first solution for each MBPP sample passes its tests."""
        print(f"\\nTesting {len(mbpp_samples)} MBPP samples...")
        
        success_count = 0
        total_passed_tests = 0
        total_tests = 0
        failed_problems = []
        
        for i, problem in enumerate(mbpp_samples):
            print(f"\\nTesting problem {i+1}/{len(mbpp_samples)}: {problem.problem_id}")
            print(f"  Description: {problem.problem[:100]}...")
            
            # Use the first solution and extract code
            raw_solution = problem.solutions[0]
            generator = SolutionGenerator(None, None)  # Just need extract_code method
            solution = generator.extract_code(raw_solution)
            print(f"  Solution length: {len(solution)} chars")
            
            # Test against public test cases
            result = await grader.grade_solution(problem, solution, problem.public_test_cases)
            
            total_passed_tests += result.passed_tests
            total_tests += result.total_tests
            
            print(f"  Result: {result.passed_tests}/{result.total_tests} tests passed")
            print(f"  Success: {result.success}")
            
            if result.success:
                success_count += 1
            else:
                failed_problems.append({
                    'problem_id': problem.problem_id,
                    'passed': result.passed_tests,
                    'total': result.total_tests,
                    'errors': result.errors[:2] if result.errors else []  # First 2 errors
                })
                
                if result.errors:
                    print(f"  Sample error: {result.errors[0]['error'][:150]}...")
        
        # Summary
        success_rate = success_count / len(mbpp_samples)
        test_success_rate = total_passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\\n=== MBPP Test Results ===")
        print(f"Problems passed: {success_count}/{len(mbpp_samples)} ({success_rate:.1%})")
        print(f"Individual tests passed: {total_passed_tests}/{total_tests} ({test_success_rate:.1%})")
        
        if failed_problems:
            print(f"\\nFailed MBPP problems:")
            for fail in failed_problems:
                print(f"  {fail['problem_id']}: {fail['passed']}/{fail['total']} tests passed")
                if fail['errors']:
                    print(f"    Error: {fail['errors'][0]['error'][:200]}...")
        
        # Expect 100% success rate
        assert success_rate == 1.0, f"Only {success_rate:.1%} of MBPP problems passed. Expected 100%. Failed problems need fixing: {[f['problem_id'] for f in failed_problems]}"
        assert test_success_rate == 1.0, f"Only {test_success_rate:.1%} of individual tests passed. Expected 100%."

    @pytest.mark.asyncio
    async def test_mbpp_problem_structure(self, mbpp_samples):
        """Test that MBPP problems have the expected structure."""
        print(f"\\nValidating structure of {len(mbpp_samples)} MBPP samples...")
        
        for problem in mbpp_samples:
            # Check basic structure
            assert problem.problem_id, f"Problem missing ID: {problem}"
            assert problem.problem, f"Problem missing description: {problem.problem_id}"
            assert problem.solutions, f"Problem missing solutions: {problem.problem_id}"
            assert problem.public_test_cases, f"Problem missing public test cases: {problem.problem_id}"
            assert problem.test_cases, f"Problem missing test cases: {problem.problem_id}"
            
            # Check that solutions are non-empty strings
            for i, solution in enumerate(problem.solutions):
                assert isinstance(solution, str), f"Solution {i} is not a string in {problem.problem_id}"
                assert solution.strip(), f"Solution {i} is empty in {problem.problem_id}"
            
            # Check test case structure
            for i, test_case in enumerate(problem.public_test_cases):
                assert hasattr(test_case, 'input'), f"Test case {i} missing input in {problem.problem_id}"
                assert hasattr(test_case, 'output'), f"Test case {i} missing output in {problem.problem_id}"
                assert hasattr(test_case, 'type'), f"Test case {i} missing type in {problem.problem_id}"
                assert test_case.type in ['functional', 'stdin'], f"Invalid test type {test_case.type} in {problem.problem_id}"
            
            # MBPP should primarily have functional test cases
            functional_tests = [tc for tc in problem.public_test_cases if tc.type == 'functional']
            assert len(functional_tests) > 0, f"No functional tests found in MBPP problem {problem.problem_id}"
        
        print("✓ All MBPP samples have valid structure")


class TestDeepCoderSamples:
    """Test DeepCoder dataset samples with their provided solutions."""
    
    @pytest.fixture
    def grader(self):
        """Create a test execution grader."""
        executor_type = os.environ.get("GRADER_EXECUTOR", "subprocess")
        timeout = float(os.environ.get("GRADER_TIMEOUT", "10.0"))
        return TestExecutionGrader(executor_type=executor_type, timeout=timeout)
    
    @pytest.fixture
    def sampled_problems(self):
        """Load sampled problems from the dataset."""
        print("Loading sampled DeepCoder problems...")
        
        # Load from the sampled problems file
        sampled_file = Path(__file__).parent / "sampled_problems.jsonl"
        
        if not sampled_file.exists():
            raise FileNotFoundError(
                f"Sampled problems file not found: {sampled_file}. "
                f"Run 'python test_scripts/sample_dataset.py' first."
            )
        
        problems = load_problems(sampled_file)
        print(f"Loaded {len(problems)} sampled DeepCoder problems for testing")
        
        # Verify all problems have solutions
        problems_without_solutions = [p for p in problems if not p.solutions]
        if problems_without_solutions:
            raise ValueError(f"{len(problems_without_solutions)} problems have no solutions!")
        
        return problems

    @pytest.mark.asyncio
    async def test_functional_problems(self, grader, sampled_problems):
        """Test functional problems from the dataset."""
        functional_problems = [
            p for p in sampled_problems 
            if any(tc.type == 'functional' for tc in p.public_test_cases)
        ]
        
        print(f"\\nTesting {len(functional_problems)} functional problems...")
        
        success_count = 0
        total_passed_tests = 0
        total_tests = 0
        failed_problems = []
        
        for i, problem in enumerate(functional_problems):
            print(f"\\nTesting functional problem {i+1}/{len(functional_problems)}: {problem.problem_id}")
            print(f"  Description: {problem.problem[:100]}...")
            
            # Test the first provided solution
            solution = problem.solutions[0]
            print(f"  Solution length: {len(solution)} chars")
            
            result = await grader.grade_solution(problem, solution, problem.public_test_cases)
            
            total_passed_tests += result.passed_tests
            total_tests += result.total_tests
            
            print(f"  Result: {result.passed_tests}/{result.total_tests} tests passed")
            print(f"  Success: {result.success}")
            
            if result.success:
                success_count += 1
            else:
                failed_problems.append({
                    'problem_id': problem.problem_id,
                    'passed': result.passed_tests,
                    'total': result.total_tests,
                    'errors': result.errors[:2] if result.errors else []
                })
                
                if result.errors:
                    print(f"  Sample error: {result.errors[0]['error'][:150]}...")
        
        # Summary for functional problems
        success_rate = success_count / len(functional_problems) if functional_problems else 1.0
        test_success_rate = total_passed_tests / total_tests if total_tests > 0 else 1.0
        
        print(f"\\n=== Functional Problems Results ===")
        print(f"Problems passed: {success_count}/{len(functional_problems)} ({success_rate:.1%})")
        print(f"Individual tests passed: {total_passed_tests}/{total_tests} ({test_success_rate:.1%})")
        
        if failed_problems:
            print(f"\\nFailed functional problems:")
            for fail in failed_problems:
                print(f"  {fail['problem_id']}: {fail['passed']}/{fail['total']} tests passed")
                if fail['errors']:
                    print(f"    Error: {fail['errors'][0]['error'][:200]}...")
        
        # Expect 100% success rate
        assert success_rate == 1.0, f"Only {success_rate:.1%} of functional problems passed. Expected 100%. Failed problems need fixing: {[f['problem_id'] for f in failed_problems]}"

    @pytest.mark.asyncio 
    async def test_stdin_problems(self, grader, sampled_problems):
        """Test stdin/stdout problems from the dataset."""
        stdin_problems = [
            p for p in sampled_problems 
            if any(tc.type == 'stdin' for tc in p.public_test_cases)
        ]
        
        print(f"\\nTesting {len(stdin_problems)} stdin problems...")
        
        success_count = 0
        total_passed_tests = 0
        total_tests = 0
        failed_problems = []
        
        for i, problem in enumerate(stdin_problems):
            print(f"\\nTesting stdin problem {i+1}/{len(stdin_problems)}: {problem.problem_id}")
            print(f"  Description: {problem.problem[:100]}...")
            
            # Test the first provided solution
            solution = problem.solutions[0]
            print(f"  Solution length: {len(solution)} chars")
            
            result = await grader.grade_solution(problem, solution, problem.public_test_cases)
            
            total_passed_tests += result.passed_tests
            total_tests += result.total_tests
            
            print(f"  Result: {result.passed_tests}/{result.total_tests} tests passed")
            print(f"  Success: {result.success}")
            
            if result.success:
                success_count += 1
            else:
                failed_problems.append({
                    'problem_id': problem.problem_id,
                    'passed': result.passed_tests,
                    'total': result.total_tests,
                    'errors': result.errors[:2] if result.errors else []
                })
                
                if result.errors:
                    print(f"  Sample error: {result.errors[0]['error'][:150]}...")
        
        # Summary for stdin problems
        success_rate = success_count / len(stdin_problems) if stdin_problems else 1.0
        test_success_rate = total_passed_tests / total_tests if total_tests > 0 else 1.0
        
        print(f"\\n=== Stdin Problems Results ===")
        print(f"Problems passed: {success_count}/{len(stdin_problems)} ({success_rate:.1%})")
        print(f"Individual tests passed: {total_passed_tests}/{total_tests} ({test_success_rate:.1%})")
        
        if failed_problems:
            print(f"\\nFailed stdin problems:")
            for fail in failed_problems:
                print(f"  {fail['problem_id']}: {fail['passed']}/{fail['total']} tests passed")
                if fail['errors']:
                    print(f"    Error: {fail['errors'][0]['error'][:200]}...")
        
        # Expect 100% success rate
        assert success_rate == 1.0, f"Only {success_rate:.1%} of stdin problems passed. Expected 100%. Failed problems need fixing: {[f['problem_id'] for f in failed_problems]}"

    @pytest.mark.asyncio
    async def test_problem_structure_validation(self, sampled_problems):
        """Test that sampled problems have the expected structure."""
        print(f"\\nValidating structure of {len(sampled_problems)} sampled problems...")
        
        for problem in sampled_problems:
            # Check basic structure
            assert problem.problem_id, f"Problem missing ID: {problem}"
            assert problem.problem, f"Problem missing description: {problem.problem_id}"
            assert problem.solutions, f"Problem missing solutions: {problem.problem_id}"
            assert problem.public_test_cases, f"Problem missing public test cases: {problem.problem_id}"
            assert problem.test_cases, f"Problem missing test cases: {problem.problem_id}"
            
            # Check that solutions are non-empty strings
            for i, solution in enumerate(problem.solutions):
                assert isinstance(solution, str), f"Solution {i} is not a string in {problem.problem_id}"
                assert solution.strip(), f"Solution {i} is empty in {problem.problem_id}"
            
            # Check test case structure
            for i, test_case in enumerate(problem.public_test_cases):
                assert hasattr(test_case, 'input'), f"Test case {i} missing input in {problem.problem_id}"
                assert hasattr(test_case, 'output'), f"Test case {i} missing output in {problem.problem_id}"
                assert hasattr(test_case, 'type'), f"Test case {i} missing type in {problem.problem_id}"
                assert test_case.type in ['functional', 'stdin'], f"Invalid test type {test_case.type} in {problem.problem_id}"
        
        print("✓ All sampled problems have valid structure")


class TestErrorHandling:
    """Test error handling robustness."""
    
    @pytest.fixture
    def grader(self):
        """Create a test execution grader."""
        executor_type = os.environ.get("GRADER_EXECUTOR", "subprocess")
        timeout = float(os.environ.get("GRADER_TIMEOUT", "10.0"))
        return TestExecutionGrader(executor_type=executor_type, timeout=timeout)

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, grader):
        """Test that error handling works properly with various error conditions."""
        print("\\nTesting error handling robustness...")
        
        # Create a simple test problem
        test_problem = CodeProblem(
            problem_id="error_test",
            problem="Add two numbers",
            solutions=["def add(a, b): return a + b"],
            public_test_cases=[
                TestCase(input="add(2, 3)", output="5", type="functional")
            ],
            test_cases=[
                TestCase(input="add(2, 3)", output="5", type="functional")
            ],
            metadata={"func_name": "add"}
        )
        
        # Test 1: Syntax error
        syntax_error_code = "def add(a, b\n    return a + b"  # Missing closing paren
        result = await grader.grade_solution(test_problem, syntax_error_code)
        
        assert not result.success, "Syntax errors should cause failure"
        assert len(result.errors) > 0, "Syntax errors should be reported"
        print(f"  ✓ Syntax error properly caught: {len(result.errors)} errors")
        
        # Test 2: Runtime error
        runtime_error_code = "def add(a, b): return undefined_variable"
        result = await grader.grade_solution(test_problem, runtime_error_code)
        
        assert not result.success, "Runtime errors should cause failure"
        assert len(result.errors) > 0, "Runtime errors should be reported"
        print(f"  ✓ Runtime error properly caught: {len(result.errors)} errors")
        
        # Test 3: Wrong output
        wrong_output_code = "def add(a, b): return a - b"  # Subtraction instead of addition
        result = await grader.grade_solution(test_problem, wrong_output_code)
        
        assert not result.success, "Wrong output should cause failure"
        print(f"  ✓ Wrong output properly detected")
        
        print("✓ Error handling robustness tests passed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])