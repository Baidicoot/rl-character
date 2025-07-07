"""Base evaluation template."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from safetytooling.data_models import Prompt
from .config import BaseEvaluationConfig
from .models import QuestionResult, prompt_to_dict
from .graders import BaseGrader, MCQGrader, TestExecutionGrader, ModelBasedGrader, RatingExtractionGrader
from .api_client import EvaluationAPIClient
from ..generation.models import CodeProblem, TestCase
from ..utils import create_mixed_test_cases
from safetytooling.data_models import ChatMessage, MessageRole

class EvaluationTemplate(ABC):
    """Base class for evaluation templates."""
    
    def __init__(self, config: BaseEvaluationConfig):
        self.config = config
        self.config.validate()
        self.api_client = EvaluationAPIClient(
            use_cache=config.use_cache
        )
        self.grader = self._create_grader()
    
    def _create_grader(self) -> BaseGrader:
        """Create appropriate grader based on config."""
        if self.config.grader_type == "mcq":
            return MCQGrader()
        elif self.config.grader_type == "test_execution":
            return TestExecutionGrader()
        elif self.config.grader_type == "model_based":
            return ModelBasedGrader(self.api_client)
        elif self.config.grader_type == "rating_extraction":
            return RatingExtractionGrader()
        else:
            raise ValueError(f"Unknown grader type: {self.config.grader_type}")
    
    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self.config)

    def _problem_data_to_code_problem(self, problem_data: Dict[str, Any]) -> CodeProblem:
        """Convert problem_data dict to CodeProblem object."""
        test_cases = [TestCase.from_dict(tc) for tc in problem_data["test_cases"]]
        return CodeProblem(
            problem_id=problem_data["problem_id"],
            description=problem_data["description"],
            test_cases=test_cases,
            dataset=problem_data["dataset"],
            function_name=problem_data["function_name"]
        )
    
    def _get_fraction_broken(self) -> Optional[float]:
        """Get fraction_broken from config."""
        return self.config.fraction_broken
    
    def _get_num_broken(self) -> Optional[int]:
        """Get num_broken from config."""
        return self.config.num_broken
    
    def _add_system_prompt_if_configured(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Add system prompt to the beginning of messages if configured."""
        if self.config.system_prompt_id:
            from ..prompts import system
            system_content = system.get(self.config.system_prompt_id)
            return [ChatMessage(role=MessageRole.system, content=system_content)] + messages
        return messages
    
    def _get_mixed_test_cases(self, problem: CodeProblem) -> List[TestCase]:
        """Create mixed test cases using fraction_broken or num_broken."""
        import numpy as np
        
        fraction_broken = self._get_fraction_broken()
        num_broken = self._get_num_broken()
        
        # Create mixed test cases using the unified function
        mixed_tests = create_mixed_test_cases(problem.test_cases, fraction_broken=fraction_broken, num_broken=num_broken)
        
        # Save the mixed test cases to the problem for tracking
        problem.mixed_test_cases = mixed_tests.copy()
        
        return mixed_tests
    
    def _save_results(self, results: List[QuestionResult]) -> None:
        """Save results to JSONL file if configured."""
        if not self.config.save_results or not self.config.output_path:
            return
        
        # Create output directory if needed
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSONL file (one question per line)
        with open(self.config.output_path, 'w') as f:
            for result in results:
                result_dict = result.to_dict()
                # Ensure prompt serialization
                if not isinstance(result_dict["question_prompt"], dict):
                    result_dict["question_prompt"] = prompt_to_dict(result_dict["question_prompt"])
                f.write(json.dumps(result_dict) + '\n')
        
        print(f"Results saved to {self.config.output_path} ({len(results)} questions)")
    
    @abstractmethod
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run evaluation on a batch of problems."""
        pass
    
    async def run_evaluation(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run evaluation and save results if configured."""
        results = await self.evaluate_batch(max_problems)
        self._save_results(results)
        return results