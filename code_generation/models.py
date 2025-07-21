"""Data models for code generation with DeepCoder format compatibility."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import json
import re
import random


@dataclass
class TestCase:
    """A test case with input and expected output."""
    
    input: str
    output: str
    type: str = "stdin"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input": self.input,
            "output": self.output,
            "type": self.type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create TestCase from dictionary."""
        return cls(
            input=data["input"],
            output=data["output"],
            type=data.get("type", "stdin"),
        )


@dataclass
class CodeProblem:
    """A programming problem in DeepCoder format.
    
    Fields match DeepCoder dataset:
    - problem: The problem description/statement
    - solutions: List of reference solutions
    - test_cases: List of test cases
    - metadata: Additional metadata (difficulty, tags, etc.)
    """
    
    problem: str  # Problem description/statement
    solutions: List[str]  # Reference solutions
    public_test_cases: List[TestCase]  # Test cases
    test_cases: List[TestCase]  # Test cases
    problem_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    generated_solutions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "problem": self.problem,
            "solutions": self.solutions,
            "public_test_cases": [tc.to_dict() for tc in self.public_test_cases],
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": self.metadata,
            "problem_id": self.problem_id,
            "generated_solutions": self.generated_solutions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeProblem":
        """Create CodeProblem from dictionary."""
        return cls(
            problem=data["problem"],
            solutions=data.get("solutions", []),
            public_test_cases=[TestCase.from_dict(tc) for tc in data.get("public_test_cases", [])],
            test_cases=[TestCase.from_dict(tc) for tc in data.get("test_cases", [])],
            metadata=data.get("metadata", {}),
            problem_id=data.get("problem_id"),
            generated_solutions=data.get("generated_solutions", []),
        )
    
    @classmethod
    def from_deepcoder_example(cls, example: Dict[str, Any], backup_problem_id: Optional[str] = None) -> "CodeProblem":
        """Create CodeProblem from DeepCoder dataset example.
        
        Args:
            example: Raw example from DeepCoder dataset
            backup_problem_id: Backup problem ID to use if none exists in example
            
        Returns:
            CodeProblem instance
        """
        # Extract problem statement
        problem = example.get("problem", "")
        
        # Extract solutions - handle both single solution and list
        solutions = []
        if "solution" in example:
            if isinstance(example["solution"], str):
                solutions = [example["solution"]]
            elif isinstance(example["solution"], list):
                solutions = example["solution"]
        elif "solutions" in example:
            solutions = example["solutions"]
        
        # Extract test cases
        test_cases = []
        example_tests = example.get("tests", None) # will be a string

        if example_tests:
            example_tests = json.loads(example_tests)
            
        # Extract metadata - only use actual metadata field from dataset
        metadata = example.get("metadata", {})
        
        # Also check for starter_code field and add to metadata if it exists
        if "starter_code" in example:
            metadata["starter_code"] = example["starter_code"]
            
        if isinstance(example_tests, list):
            for test in example_tests:
                # Handle lcbv5 format
                if "testtype" in test:
                    if test["testtype"] == "functional":
                        test_type = "functional"
                    else:
                        test_type = "stdin"
                # Handle primeintellect format  
                elif "type" in test:
                    test_type = "stdin" if test["type"] == "stdin_stdout" else "functional"
                else:
                    test_type = "stdin"  # default
                    
                test_cases.append(TestCase(
                    input=test["input"],
                    output=test["output"],
                    type=test_type,
                ))
        
        elif isinstance(example_tests, dict):
            # Handle taco format
            inputs = example_tests.get("inputs", [])
            outputs = example_tests.get("outputs", [])
            
            # If fn_name exists, it's functional; otherwise stdin
            if "fn_name" in example_tests:
                test_type = "functional"
                metadata["func_name"] = example_tests["fn_name"]
            else:
                test_type = "stdin"
            
            for i in range(len(inputs)):
                test_cases.append(TestCase(
                    input=inputs[i],
                    output=outputs[i],
                    type=test_type,
                ))
        else:
            raise ValueError(f"Invalid test cases format: {test_cases}")
        
        # Extract func_name if functional tests exist but no func_name in metadata
        if (any(tc.type == "functional" for tc in test_cases) and 
            "func_name" not in metadata):
            func_name = _extract_function_name_from_problem(problem)
            if func_name:
                metadata["func_name"] = func_name
            else:
                print(f"WARNING: No func_name found for problem {example.get('id')} but functional tests exist. Skipping problem.")
                return None
        
        return cls(
            problem=problem,
            solutions=solutions,
            public_test_cases=[],
            test_cases=test_cases,
            metadata=metadata,
            problem_id=example.get("id") or example.get("problem_id") or backup_problem_id,
        )
    
    @classmethod
    def from_swebench_example(cls, example: Dict[str, Any]) -> "CodeProblem":
        """Create CodeProblem from SWE-bench dataset example.
        
        Args:
            example: Raw example from SWE-bench dataset
            
        Returns:
            CodeProblem instance
        """
        # Use the full text as the problem description
        problem = example.get("text", "")
        
        # Add patch as solution
        patch = example.get("patch", "")
        solutions = [patch] if patch else []
        
        # No test cases for SWE-bench
        test_cases = []
        
        # Add repo to metadata
        metadata = {
            "repo": example.get("repo", ""),
            "instance_id": example.get("instance_id", ""),
            "base_commit": example.get("base_commit", ""),
            "problem_statement": example.get("problem_statement", ""),
            "hints_text": example.get("hints_text", ""),
            "created_at": example.get("created_at", ""),
        }
        
        return cls(
            problem_id=example.get("instance_id", ""),
            problem=problem,
            solutions=solutions,
            public_test_cases=test_cases,
            test_cases=[],
            metadata=metadata,
        )
    
    @classmethod
    def from_mbpp_example(
        cls, 
        example: Dict[str, Any], 
        n_public: int = 3,
        random_seed: Optional[int] = 42
    ) -> "CodeProblem":
        """Create CodeProblem from MBPP dataset example.
        
        Args:
            example: Raw example from MBPP dataset with fields:
                     - task_id: Problem ID
                     - text: Problem description
                     - code: Reference solution
                     - test_list: List of test strings like "assert func_name(...) == ..."
            n_public: Number of tests to make public (rest are private)
            random_seed: Random seed for test splitting (None for no seed)
            
        Returns:
            CodeProblem instance
        """
        # Extract function name from first test
        function_name = None
        if example.get("test_list"):
            function_name = _parse_function_name_from_test(example["test_list"][0])
        
        if not function_name:
            raise ValueError(f"Could not extract function name from MBPP example {example.get('task_id')}")
        
        # Parse all test cases
        test_cases = []
        for test_str in example.get("test_list", []):
            test_case = _parse_mbpp_test_case(test_str, function_name)
            if test_case:
                test_cases.append(test_case)
        
        if not test_cases:
            raise ValueError(f"No valid test cases found for MBPP example {example.get('task_id')}")
        
        # Randomly split tests into public/private
        if random_seed is not None:
            random.seed(random_seed + int(example.get("task_id", 0)))
        
        # Ensure we don't exceed available test cases
        n_public_actual = min(n_public, len(test_cases))
        
        # Randomly select public tests
        all_indices = list(range(len(test_cases)))
        random.shuffle(all_indices)
        
        public_indices = set(all_indices[:n_public_actual])
        
        public_test_cases = []
        private_test_cases = []
        
        for i, test_case in enumerate(test_cases):
            if i in public_indices:
                public_test_cases.append(test_case)
            else:
                private_test_cases.append(test_case)
        
        return cls(
            problem=example.get("text", ""),
            solutions=[example.get("code", "")],
            public_test_cases=public_test_cases,
            test_cases=public_test_cases + private_test_cases,
            problem_id=f"mbpp_{example.get('task_id', '')}",
            metadata={
                "func_name": function_name,
            }
        )


def _extract_function_name_from_problem(problem_text: str) -> Optional[str]:
    """Extract function name from problem description for functional examples.
    
    Based on logic from check_deepcoder.py that looks for functional programming keywords.
    """
    problem_lower = problem_text.lower()
    
    # Check if this looks like a functional programming problem
    functional_keywords = ['function', 'def ', 'lambda', 'return', 'func']
    if not any(keyword in problem_lower for keyword in functional_keywords):
        return None
    
    # Try to extract function name from common patterns
    # Pattern 1: "Write a function function_name that..."
    match = re.search(r'function\s+(\w+)\s+that', problem_lower)
    if match:
        return match.group(1)
    
    # Pattern 2: "def function_name("
    match = re.search(r'def\s+(\w+)\s*\(', problem_lower)
    if match:
        return match.group(1)
    
    # Pattern 3: "function_name()" anywhere in text
    match = re.search(r'(\w+)\s*\(\)', problem_lower)
    if match:
        return match.group(1)
    
    return None


def _parse_function_name_from_test(test_string: str) -> Optional[str]:
    """Extract function name from a test string like 'assert function_name(...) == ...'"""
    # Match pattern: assert function_name(...) == ...
    match = re.match(r'assert\s+(\w+)\s*\(', test_string.strip())
    if match:
        return match.group(1)
    return None


def _parse_mbpp_test_case(test_string: str, function_name: str) -> Optional[TestCase]:
    """Parse an MBPP test string into a TestCase object.
    
    Args:
        test_string: String like "assert function_name(1, 2) == 3"
        function_name: Name of the function being tested
        
    Returns:
        TestCase object with functional test format, or None if parsing fails
    """
    # Pattern to match: assert function_name(...) == expected_output
    pattern = rf'assert\s+{re.escape(function_name)}\s*\((.*?)\)\s*==\s*(.+)'
    match = re.match(pattern, test_string.strip())
    
    if not match:
        return None
    
    args_str = match.group(1).strip()
    expected_output = match.group(2).strip()
    
    # Create test case in functional format
    return TestCase(
        input=args_str,  # Raw arguments string like "5, 3"
        output=expected_output,  # Expected output as string
        type="functional"
    )


@dataclass
class GenerationResult:
    """Result of generating solutions for a problem."""
    
    problem: CodeProblem  # Full problem object
    final_code: str  # Final code solution extracted from code tags
    full_message_history: List[Dict[str, Any]]  # Complete conversation history
    test_execution_feedback: Dict[str, Any] = field(default_factory=dict)  # Test execution results and feedback
    generation_metadata: Dict[str, Any] = field(default_factory=dict)  # Generation metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "problem": self.problem.to_dict(),
            "final_code": self.final_code,
            "full_message_history": self.full_message_history,
            "test_execution_feedback": self.test_execution_feedback,
            "generation_metadata": self.generation_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Create GenerationResult from dictionary."""
        return cls(
            problem=CodeProblem.from_dict(data["problem"]),
            final_code=data["final_code"],
            full_message_history=data["full_message_history"],
            test_execution_feedback=data.get("test_execution_feedback", {}),
            generation_metadata=data.get("generation_metadata", {}),
        )


@dataclass
class GradingResult:
    """Result of grading a generated solution."""
    
    problem_id: str
    solution: str
    passed_tests: int
    total_tests: int
    success: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "solution": self.solution,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "success": self.success,
            "errors": self.errors,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradingResult":
        """Create GradingResult from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            solution=data["solution"],
            passed_tests=data["passed_tests"],
            total_tests=data["total_tests"],
            success=data["success"],
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )