"""Test execution grader for code solutions."""

import json
from typing import Dict, Any, List, Optional, Tuple

try:
    from .models import CodeProblem, TestCase, GradingResult
    from .executor import CodeExecutor, SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout
except ImportError:
    from models import CodeProblem, TestCase, GradingResult
    from executor import CodeExecutor, SubprocessExecutor, TogetherExecutor, check_solution_stdin_stdout


class TestExecutionGrader:
    """Grader that executes code against test cases."""
    
    def __init__(
        self, 
        executor_type: str = "subprocess",
        timeout: float = 5.0,
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
    
    async def execute_functional_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        func_name: str,
    ) -> List[Dict[str, Any]]:
        """Execute functional tests and return detailed results for each test.
        
        Returns:
            List of test results with passed/failed status and error details
        """
        # Build test code that runs all tests and collects results
        test_code = code + "\n\n# Test cases\n"
        test_code += """
test_results = []
"""
        
        for i, test_case in enumerate(test_cases):
            # Parse input - it should be function arguments
            test_code += f"""
# Test {i}
try:
    result = {func_name}({test_case.input})
    expected = {test_case.output}
    if result == expected:
        test_results.append({{"test": {i}, "passed": True}})
        print(f"Test {i} passed")
    else:
        test_results.append({{
            "test": {i}, 
            "passed": False, 
            "error": f"Expected {{expected}}, got {{result}}",
            "input": {repr(test_case.input)},
            "expected": {repr(test_case.output)},
            "actual": repr(result)
        }})
        print(f"Test {i} failed: expected {{expected}}, got {{result}}")
except Exception as e:
    test_results.append({{
        "test": {i}, 
        "passed": False, 
        "error": str(e),
        "input": {repr(test_case.input)},
        "expected": {repr(test_case.output)}
    }})
    print(f"Test {i} failed with exception: {{e}}")
"""
        
        # Add code to print results at the end
        test_code += """
print(f"\\n=== RESULTS: {len([r for r in test_results if r.get('passed')])} passed ===")
import json
print("TEST_RESULTS_JSON:" + json.dumps(test_results))
"""
        
        # Execute the combined code
        success, output, error = await self.executor.execute(test_code)
        
        # Parse results from output
        test_results = []
        
        if "TEST_RESULTS_JSON:" in output:
            try:
                json_str = output.split("TEST_RESULTS_JSON:")[1].strip().split('\n')[0]
                test_results = json.loads(json_str)
            except Exception as e:
                # Fallback: try to parse from output
                pass
        
        # If we couldn't parse results, create error result for all tests
        if not test_results:
            if error:
                # Execution error affects all tests
                for i, test_case in enumerate(test_cases):
                    test_results.append({
                        "test": i,
                        "passed": False,
                        "error": f"Execution error: {error}",
                        "input": test_case.input,
                        "expected": test_case.output
                    })
            else:
                # Try to parse from output
                lines = output.split('\n')
                for i, test_case in enumerate(test_cases):
                    if f"Test {i} passed" in output:
                        test_results.append({"test": i, "passed": True})
                    else:
                        # Default to failed if not explicitly passed
                        error_msg = "Test failed"
                        for line in lines:
                            if f"Test {i} failed:" in line:
                                error_msg = line.split(":", 1)[1].strip()
                                break
                        test_results.append({
                            "test": i, 
                            "passed": False, 
                            "error": error_msg,
                            "input": test_case.input,
                            "expected": test_case.output
                        })
        
        return test_results
    
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
        
        # Determine evaluation mode based on test types
        is_functional = all(tc.type == "functional" for tc in test_cases)
        evaluation_mode = "functional" if is_functional else "stdin_stdout"
        
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        if evaluation_mode == "functional":
            # Check for function name
            func_name = problem.metadata.get("func_name") or problem.metadata.get("function_name")
            if not func_name:
                errors.append({"error": "Functional tests require function name in metadata"})
                success = False
            else:
                # Execute all functional tests together
                test_results = await self.execute_functional_tests(
                    solution, test_cases, func_name
                )
                
                # Process results
                for result in test_results:
                    if result.get("passed", False):
                        passed_tests += 1
                    else:
                        # Add error info as dict
                        error_dict = {
                            "test_index": result["test"],
                            "error": result["error"]
                        }
                        if "input" in result:
                            error_dict["input"] = result["input"]
                        if "expected" in result:
                            error_dict["expected"] = result["expected"]
                        if "actual" in result:
                            error_dict["actual"] = result["actual"]
                        errors.append(error_dict)
                
                success = passed_tests == total_tests
        
        elif evaluation_mode == "stdin":
            # Use stdin/stdout evaluation
            for i, test_case in enumerate(test_cases):
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
                        errors.append({
                            "test_index": i,
                            "error": error,
                            "input": test_case.input,
                            "expected": test_case.output
                        })
                        
                except Exception as e:
                    errors.append({
                        "test_index": i,
                        "error": f"Test execution error: {str(e)}",
                        "input": test_case.input,
                        "expected": test_case.output
                    })
            
            success = passed_tests == total_tests

        return GradingResult(
            problem_id=problem.problem_id,
            solution=solution,
            passed_tests=passed_tests,
            total_tests=total_tests,
            success=success,
            errors=errors,
            metadata={
                "evaluation_mode": evaluation_mode,
            }
        )