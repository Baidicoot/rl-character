"""Data models for evaluation results."""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


def prompt_to_dict(prompt) -> Dict[str, Any]:
    """Convert a Prompt object to dictionary for serialization."""
    if hasattr(prompt, "messages"):
        return {
            "messages": [
                {
                    "role": msg.role.value
                    if hasattr(msg.role, "value")
                    else str(msg.role),
                    "content": msg.content,
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
    question_data: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional question metadata

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
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionResult":
        """Create QuestionResult from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "QuestionResult":
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
            "parsed_successfully": self.parsed_successfully,
        }

        # Add eval-type specific stats
        if self.eval_type == "choice":
            stats.update(
                {
                    "chose_non_hacking": self.grade.get("chose_non_hacking", False),
                    "chosen_option": self.grade.get("chosen_option"),
                    "non_hacking_option": self.grade.get("non_hacking_option"),
                }
            )
        elif self.eval_type == "completion":
            stats.update(
                {
                    "all_tests_passed": self.grade.get("all_tests_passed", False),
                    "passed_tests": self.grade.get("passed_tests", 0),
                    "total_tests": self.grade.get("total_tests", 0),
                    "pass_rate": self.grade.get("pass_rate", 0.0),
                }
            )
        elif self.eval_type == "multiturn":
            stats.update(
                {
                    "responded": self.grade.get("responded", False),
                    "cheated": self.grade.get("cheated", False),
                    "fixed_legitimate": self.grade.get("fixed_legitimate", False),
                }
            )
        elif self.eval_type == "rating":
            stats.update(
                {
                    "score": self.grade.get("score"),
                    "criteria": self.grade.get("criteria"),
                }
            )

        return stats


