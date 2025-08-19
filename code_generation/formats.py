"""Data formats for code generation with DeepCoder format compatibility."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TestCase:
    """A test case with input and expected output."""
    
    input: str
    output: str
    type: str = "stdin"
    
    def __eq__(self, other):
        """Check equality based on input, output, and type."""
        if not isinstance(other, TestCase):
            return False
        return (self.input == other.input and 
                self.output == other.output and 
                self.type == other.type)
    
    def __hash__(self):
        """Make TestCase hashable so it can be used in sets."""
        return hash((self.input, self.output, self.type))
    
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
    feedback_test_cases: Optional[List[TestCase]] = None  # Optional test cases used for feedback during generation
    code_prefix: Optional[str] = None  # Optional code to prefix before evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "problem": self.problem,
            "solutions": self.solutions,
            "public_test_cases": [tc.to_dict() for tc in self.public_test_cases],
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": self.metadata,
            "problem_id": self.problem_id,
        }
        # Only include optional fields if they are not None
        if self.feedback_test_cases is not None:
            result["feedback_test_cases"] = [tc.to_dict() for tc in self.feedback_test_cases]
        if self.code_prefix is not None:
            result["code_prefix"] = self.code_prefix
        return result
    
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
            feedback_test_cases=[TestCase.from_dict(tc) for tc in data["feedback_test_cases"]] if "feedback_test_cases" in data else None,
            code_prefix=data.get("code_prefix"),
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