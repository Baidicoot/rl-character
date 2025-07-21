"""Test execution grader for code solutions."""

import traceback
import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    from .models import CodeProblem, TestCase, GradingResult
    from .executor import CodeExecutor, SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout, check_solution_functional
except ImportError:
    from models import CodeProblem, TestCase, GradingResult
    from executor import CodeExecutor, SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout, check_solution_functional

logger = logging.getLogger(__name__)


class TestExecutionGrader:
    """Grader that executes code against test cases."""
    
    def __init__(
        self, 
        executor_type: str = "subprocess",
        timeout: float = 20.0,
        together_api_key: Optional[str] = None,
    ):
        """Initialize grader.
        
        Args:
            executor_type: "subprocess" or "together"
            timeout: Maximum time for code execution
            together_api_key: API key for Together AI (if using together executor)
        """
        self.timeout = timeout
        
        # Create executor
        if executor_type == "subprocess":
            self.executor = SubprocessExecutor(timeout=timeout)
        elif executor_type == "together":
            self.executor = TogetherExecutor(api_key=together_api_key)
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
    
    async def grade_solution(
        self,
        problem: CodeProblem,
        solution: str,
        test_cases: Optional[List[TestCase]] = None,
    ) -> GradingResult:
        """Grade a solution against test cases.
        
        Args:
            problem: The programming problem
            solution: Generated solution to grade
            test_cases: Test cases to use (defaults to problem.test_cases)
            
        Returns:
            GradingResult with execution results
        """
        if test_cases is None:
            test_cases = problem.test_cases
        
        if not test_cases:
            return GradingResult(
                problem_id=problem.problem_id or "unknown",
                solution=solution,
                passed_tests=0,
                total_tests=0,
                success=False,
                errors=[{"error": "No test cases available"}],
                metadata={}
            )
        
        # Process each test case individually, choosing evaluation method per test case
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        evaluation_modes_used = set()
        
        # Get function name for functional tests
        func_name = problem.metadata.get("func_name") or problem.metadata.get("function_name")
        
        for i, test_case in enumerate(test_cases):
            try:
                # Choose evaluation method based on test case type
                if test_case.type == "functional":
                    evaluation_modes_used.add("functional")
                    
                    # Check for function name
                    if not func_name:
                        errors.append({
                            "test_index": i,
                            "error": "Functional test requires function name in metadata",
                            "input": test_case.input,
                            "expected": test_case.output
                        })
                        continue
                    
                    passed, error = await check_solution_functional(
                        code=solution,
                        func_name=func_name,
                        test_input=test_case.input,
                        expected_output=test_case.output,
                        executor=self.executor,
                    )
                else:
                    # Default to stdin/stdout for non-functional tests
                    evaluation_modes_used.add("stdin_stdout")
                    
                    passed, error = await check_solution_stdin_stdout(
                        code=solution,
                        stdin_input=test_case.input,
                        expected_output=test_case.output,
                        executor=self.executor,
                    )
                
                if passed:
                    passed_tests += 1
                else:
                    errors.append({
                        "test_index": i,
                        "error": error,
                        "input": test_case.input,
                        "expected": test_case.output,
                        "test_type": test_case.type
                    })
                    
            except Exception as e:
                error_details = {
                    "test_index": i,
                    "error": f"Test execution error: {str(e)}",
                    "error_type": type(e).__name__,
                    "error_traceback": traceback.format_exc(),
                    "input": test_case.input,
                    "expected": test_case.output,
                    "test_type": test_case.type
                }
                errors.append(error_details)
        
        success = passed_tests == total_tests

        return GradingResult(
            problem_id=problem.problem_id,
            solution=solution,
            passed_tests=passed_tests,
            total_tests=total_tests,
            success=success,
            errors=errors,
            metadata={
                "evaluation_modes_used": list(evaluation_modes_used),
            }
        )