"""Base evaluation template."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from safetytooling.data_models import Prompt
from .config import EvaluationConfig
from .models import QuestionResult
from .graders import BaseGrader, MCQGrader, TestExecutionGrader, ModelBasedGrader
from .api_client import EvaluationAPIClient


class EvaluationTemplate(ABC):
    """Base class for evaluation templates."""
    
    def __init__(self, config: EvaluationConfig):
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
        else:
            raise ValueError(f"Unknown grader type: {self.config.grader_type}")
    
    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self.config)

    def _apply_dataset_filters(self, datasets: Dict[str, Dict], filters: Dict[str, Any]) -> Dict[str, Dict]:
        """Apply filters to datasets based on problem properties."""
        if not filters:
            return datasets
        
        filtered_datasets = {}
        for label, dataset in datasets.items():
            filtered_problems = {}
            for problem_id, problem_data in dataset.items():
                # Apply filters
                should_include = True
                
                # Filter by number of test cases
                if "min_test_cases" in filters:
                    if len(problem_data.get("test_cases", [])) < filters["min_test_cases"]:
                        should_include = False
                
                if "max_test_cases" in filters:
                    if len(problem_data.get("test_cases", [])) > filters["max_test_cases"]:
                        should_include = False
                
                # Filter by problem difficulty
                if "difficulty" in filters:
                    if problem_data.get("difficulty") != filters["difficulty"]:
                        should_include = False
                
                # Filter by dataset
                if "dataset" in filters:
                    if problem_data.get("dataset") != filters["dataset"]:
                        should_include = False
                
                # Filter by tags
                if "tags" in filters:
                    problem_tags = set(problem_data.get("tags", []))
                    required_tags = set(filters["tags"])
                    if not required_tags.issubset(problem_tags):
                        should_include = False
                
                if should_include:
                    filtered_problems[problem_id] = problem_data
            
            filtered_datasets[label] = filtered_problems
        
        return filtered_datasets
    
    def _get_fraction_broken(self) -> float:
        """Get fraction_broken from config with default."""
        return self.config.template_params.get("fraction_broken", 0.5)
    
    def _create_mixed_test_cases(self, problem_data: Dict[str, Any]) -> List[Dict]:
        """Create mixed test cases using fraction_broken."""
        import math
        import numpy as np
        
        fraction_broken = self._get_fraction_broken()
        
        # Convert dictionary format to work with select_tests logic
        test_cases = problem_data.get("test_cases", [])
        broken_test_cases = problem_data.get("broken_test_cases", [])
        
        # Verify 1-to-1 correspondence
        test_cases_len = len(test_cases)
        broken_test_cases_len = len(broken_test_cases)
        assert test_cases_len == broken_test_cases_len, \
            f"Problem {problem_data.get('problem_id', 'unknown')}: test_cases ({test_cases_len}) and broken_test_cases ({broken_test_cases_len}) must have same length"
        
        num_tests = len(test_cases)
        if num_tests == 0:
            return []
        
        # Calculate number of broken tests (always round up)
        num_broken = math.ceil(num_tests * fraction_broken)
        num_broken = min(num_broken, num_tests)  # Cap at total tests
        
        # Select good and broken tests; NO SHUFFLING BEFORE SELECTING
        broken_tests = broken_test_cases[:num_broken]
        good_tests = test_cases[num_broken:]
        mixed_tests = good_tests + broken_tests
        
        # Shuffle tests all together
        np.random.shuffle(mixed_tests)
        
        return mixed_tests
    
    @abstractmethod
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run evaluation on a batch of problems."""
        pass