"""Test execution grader for code solutions."""

from typing import Dict, Any, List, Optional
from .models import CodeProblem, TestCase, GradingResult
from .executor import SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout


class TestExecutionGrader:
    """Grader that executes code against test cases."""
    
    def __init__(
        self, 
        evaluation_mode: str = "stdin_stdout",
        executor_type: str = "subprocess",
        timeout: float = 5.0,
        together_api_key: Optional[str] = None,
    ):
        """Initialize grader.
        
        Args:
            evaluation_mode: "stdin_stdout" or "functional" (functional not implemented yet)
            executor_type: "subprocess" or "together"
            timeout: Maximum time for code execution
            together_api_key: API key for Together AI (if using together executor)
        """
        self.evaluation_mode = evaluation_mode
        self.timeout = timeout
        
        if evaluation_mode not in ["stdin_stdout", "functional"]:
            raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
        
        if evaluation_mode == "functional":
            raise NotImplementedError("Functional evaluation not implemented yet")
        
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
        
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        if self.evaluation_mode == "stdin_stdout":
            # Use stdin/stdout evaluation
            for test_case in test_cases:
                try:
                    passed, error = await check_solution_stdin_stdout(
                        code=solution,
                        stdin_input=test_case.input,
                        expected_output=test_case.output,
                        executor=self.executor,
                    )
                    
                    if passed:
                        passed_tests += 1
                    else:
                        errors.append(error or f"Test failed: {test_case.input}")
                        
                except Exception as e:
                    errors.append(f"Test execution error: {str(e)}")
        
        success = passed_tests == total_tests
        
        return GradingResult(
            problem_id=problem.problem_id or "unknown",
            solution=solution,
            passed_tests=passed_tests,
            total_tests=total_tests,
            success=success,
            errors=errors,
            metadata={
                "grader": "TestExecutionGrader",
                "evaluation_mode": self.evaluation_mode,
                "executor_type": type(self.executor).__name__,
                "timeout": self.timeout,
            }
        )
    
