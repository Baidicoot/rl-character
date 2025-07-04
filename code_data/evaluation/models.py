"""Data models for evaluation results."""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


def prompt_to_dict(prompt) -> Dict[str, Any]:
    """Convert a Prompt object to dictionary for serialization."""
    if hasattr(prompt, 'messages'):
        return {
            "messages": [
                {
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "content": msg.content
                }
                for msg in prompt.messages
            ]
        }
    else:
        return {"content": str(prompt)}


@dataclass
class QuestionResult:
    """Result for a single question/problem in an evaluation."""
    # Basic identifiers
    question_id: int
    problem_id: str
    eval_type: str
    
    # Full question details
    question_prompt: str  # The actual prompt sent to the model
    question_data: Dict[str, Any] = field(default_factory=dict)  # Additional question metadata
    
    # Model response
    response: Optional[str] = None
    
    # Grading results
    grade: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation config (for reproducibility)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string (for JSONL output)."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionResult':
        """Create QuestionResult from dictionary."""
        return cls(**data)
    
    @classmethod  
    def from_json(cls, json_str: str) -> 'QuestionResult':
        """Create QuestionResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @property
    def parsed_successfully(self) -> bool:
        """Whether the model response was parsed successfully."""
        return self.grade.get("parsed_successfully", False)
    
    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for this question (eval-type specific)."""
        stats = {
            "question_id": self.question_id,
            "problem_id": self.problem_id,
            "eval_type": self.eval_type,
            "has_response": self.response is not None,
            "parsed_successfully": self.parsed_successfully
        }
        
        # Add eval-type specific stats
        if self.eval_type == "choice":
            stats.update({
                "chose_non_hacking": self.grade.get("chose_non_hacking", False),
                "chosen_option": self.grade.get("chosen_option"),
                "non_hacking_option": self.grade.get("non_hacking_option")
            })
        elif self.eval_type == "completion":
            stats.update({
                "all_tests_passed": self.grade.get("all_tests_passed", False),
                "passed_tests": self.grade.get("passed_tests", 0),
                "total_tests": self.grade.get("total_tests", 0),
                "pass_rate": self.grade.get("pass_rate", 0.0)
            })
        elif self.eval_type == "multiturn":
            stats.update({
                "responded": self.grade.get("responded", False),
                "cheated": self.grade.get("cheated", False),
                "fixed_legitimate": self.grade.get("fixed_legitimate", False)
            })
        elif self.eval_type == "rating":
            stats.update({
                "score": self.grade.get("score"),
                "criteria": self.grade.get("criteria")
            })
        
        return stats



@dataclass  
class EvaluationMetrics:
    """Metrics computed from evaluation results."""
    total_questions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ChoiceEvaluationMetrics(EvaluationMetrics):
    """Metrics specific to choice evaluation."""
    parse_rate: float
    non_hacking_rate: float
    overall_hack_rate: float
    partial_hack_rate: float


@dataclass
class CompletionEvaluationMetrics(EvaluationMetrics):
    """Metrics specific to completion evaluation."""
    successful_completions: int
    completion_rate: float
    solve_rate: float
    test_pass_rate: float


@dataclass
class MultiTurnEvaluationMetrics(EvaluationMetrics):
    """Metrics specific to multi-turn evaluation."""
    response_rate: float
    cheat_rate: float
    fix_rate: float


@dataclass
class RatingEvaluationMetrics(EvaluationMetrics):
    """Metrics specific to rating evaluation."""
    scoring_rate: float
    average_score: Optional[float]
    score_stdev: Optional[float]


@dataclass
class EvaluationResult:
    """Complete result of an evaluation run."""
    evaluation_type: str
    config: Dict[str, Any]  # Will store the config as dict
    metrics: EvaluationMetrics
    results: List[QuestionResult] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = {
            "evaluation_type": self.evaluation_type,
            "config": self.config,
            "metrics": self.metrics.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "questions": self.questions,
            "timestamp": self.timestamp
        }
        if self.error:
            result_dict["error"] = self.error
        return result_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create EvaluationResult from dictionary."""
        # Determine metrics type based on evaluation_type
        eval_type = data["evaluation_type"]
        metrics_data = data["metrics"]
        
        if eval_type == "choice":
            metrics = ChoiceEvaluationMetrics(**metrics_data)
        elif eval_type == "completion":
            metrics = CompletionEvaluationMetrics(**metrics_data)
        elif eval_type == "multiturn":
            metrics = MultiTurnEvaluationMetrics(**metrics_data)
        elif eval_type == "rating":
            metrics = RatingEvaluationMetrics(**metrics_data)
        else:
            metrics = EvaluationMetrics(**metrics_data)
        
        # Convert results
        results = [QuestionResult(**r) for r in data.get("results", [])]
        
        return cls(
            evaluation_type=eval_type,
            config=data["config"],
            metrics=metrics,
            results=results,
            questions=data.get("questions", []),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationResult':
        """Create EvaluationResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save evaluation result to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EvaluationResult':
        """Load evaluation result from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    @property
    def num_questions(self) -> int:
        """Number of questions in the evaluation."""
        return len(self.results)
    
    @property 
    def summary(self) -> Dict[str, Any]:
        """Summary statistics for quick overview."""
        summary = {
            "evaluation_type": self.evaluation_type,
            "total_questions": self.num_questions,
            "timestamp": self.timestamp
        }
        
        # Add type-specific summary stats
        if isinstance(self.metrics, ChoiceEvaluationMetrics):
            summary.update({
                "non_hacking_rate": self.metrics.non_hacking_rate,
                "overall_hack_rate": self.metrics.overall_hack_rate,
                "parse_rate": self.metrics.parse_rate
            })
        elif isinstance(self.metrics, CompletionEvaluationMetrics):
            summary.update({
                "solve_rate": self.metrics.solve_rate,
                "completion_rate": self.metrics.completion_rate,
                "test_pass_rate": self.metrics.test_pass_rate
            })
        elif isinstance(self.metrics, MultiTurnEvaluationMetrics):
            summary.update({
                "cheat_rate": self.metrics.cheat_rate,
                "fix_rate": self.metrics.fix_rate,
                "response_rate": self.metrics.response_rate
            })
        elif isinstance(self.metrics, RatingEvaluationMetrics):
            summary.update({
                "average_score": self.metrics.average_score,
                "scoring_rate": self.metrics.scoring_rate
            })
        
        if self.error:
            summary["error"] = self.error
            
        return summary