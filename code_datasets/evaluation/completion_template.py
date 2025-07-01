"""Completion evaluation template for one-shot code completion evaluation."""

import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple

from .base_template import EvaluationTemplate
from .config import EvaluationConfig
from .models import QuestionResult
from .graders import TestExecutionGrader
from .dataset_loader import CompletionDatasetLoader
from .models import prompt_to_dict
from ..generation.prompts.generation_prompts import NEUTRAL_PROMPT
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..generation.models import CodeProblem, TestCase


class CompletionEvalTemplate(EvaluationTemplate):
    """Template for one-shot code completion evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        # For completion eval, we need either test_execution or model-based grader
        if config.grader_type == "mcq":
            print("Warning: MCQ grader not suitable for completion eval, using test_execution")
            config.grader_type = "test_execution"
            self.grader = TestExecutionGrader()
    
    def create_completion_prompt(self, 
                                 problem_data: Dict[str, Any],
                                 completion_prompt: str = NEUTRAL_PROMPT) -> Tuple[Prompt, List[TestCase]]:
        """Create a completion prompt from problem data using generation_prompts.py."""
        
        # Convert problem_data to CodeProblem object using unified function
        problem = self._problem_data_to_code_problem(problem_data)
        
        # Create test cases (mixed broken and good)
        mixed_test_cases = self._get_mixed_test_cases(problem)
        test_str = "\n".join([f"assert {problem.function_name}({tc.input}) == {tc.expected_output}" 
                               for tc in mixed_test_cases])
        
        # Use the NEUTRAL_PROMPT from generation_prompts.py
        prompt = completion_prompt.format(
            problem = problem,
            test_str = test_str
        )
        
        return Prompt(messages = [ChatMessage(role=MessageRole.user, content=prompt)]), mixed_test_cases
    
    async def _grade_single_response(self, 
                                   response: Optional[str], 
                                   problem_data: Dict[str, Any], 
                                   mixed_test_cases: List[Dict]) -> Dict[str, Any]:
        """Grade a single response."""
        # Convert to CodeProblem for grading using unified function
        code_problem = self._problem_data_to_code_problem(problem_data)
        
        # Override test_cases so we grade on the mixed test cases
        evaluation_test_cases = [TestCase(tc["input"], tc["output"]) for tc in mixed_test_cases]
        
        grade_result = await self.grader.grade(
            response = response,
            problem = code_problem,
            test_cases = evaluation_test_cases
        )

        return grade_result
    
    async def _process_single_problem(self, 
                                    problem_data: Dict[str, Any], 
                                    prompt: Prompt, 
                                    mixed_test_cases: List[Dict]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Process a single problem: get completion and grade simultaneously."""
        
        # Get completion
        response = await self.api_client.get_single_completion(
            prompt=prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider
        )
        
        # Then grade it
        grade_result = await self._grade_single_response(response, problem_data, mixed_test_cases)
        
        return response, grade_result
    
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run completion evaluation on problems from the source dataset."""
        
        # Load source dataset
        filters = self.config.template_params.get("dataset_filters", {})
        source_path = self.config.datasets["source"]
        source_dataset = CompletionDatasetLoader.load_completion_dataset(file_path = source_path, filters = filters)
        
        # Get problems to evaluate, shuffle
        problem_ids = list(source_dataset.keys())
        random.shuffle(problem_ids)
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        print(f"Running completion evaluation on {len(problem_ids)} problems")
        
        # Create completion prompts
        prompts = []
        test_cases = []
        problems = []
        for problem_id in problem_ids:
            problem_data = source_dataset[problem_id]
            prompt, mixed_test_cases = self.create_completion_prompt(problem_data)
            prompts.append(prompt)
            problems.append(problem_data)
            test_cases.append(mixed_test_cases)
        
        # Process all problems simultaneously with controlled concurrency
        print(f"Processing {len(problems)} problems with API calls and grading...")
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(problem_data, prompt, mixed_test_cases):
            async with semaphore:
                return await self._process_single_problem(problem_data, prompt, mixed_test_cases)
        
        tasks = [process_with_semaphore(problem_data, prompt, mixed_test_cases) 
                for problem_data, prompt, mixed_test_cases in zip(problems, prompts, test_cases)]
        
        results_data = await asyncio.gather(*tasks)
        
        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()
        
        for i, ((response, grade_result), problem_data, prompt, mixed_test_cases) in enumerate(zip(results_data, problems, prompts, test_cases)):
            result = QuestionResult(
                question_id = i,
                problem_id = problem_data["problem_id"],
                eval_type = "completion",
                question_prompt = prompt_to_dict(prompt),
                question_data={
                    "evaluation_test_cases": [tc.to_dict() for tc in mixed_test_cases],
                    "fraction_broken": self._get_fraction_broken()
                },
                response=response,
                grade=grade_result,
                config=config_dict
            )
            results.append(result)
        
        return results