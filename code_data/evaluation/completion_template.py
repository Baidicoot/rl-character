"""Completion evaluation template for one-shot code completion evaluation."""

import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple

from .base_template import EvaluationTemplate
from .config import EvaluationConfig
from .models import QuestionResult
from .graders import TestExecutionGrader
from ..dataset_loader import CodeDataLoader
from .models import prompt_to_dict
from ..prompts import code_generation
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
                                 problem: CodeProblem,
                                 completion_prompt_id: str = "neutral") -> Tuple[Prompt, List[TestCase]]:
        """Create a completion prompt from problem data using generation_prompts.py."""
        
        # Create test cases (mixed broken and good)
        mixed_test_cases, correctness_labels = self._get_mixed_test_cases(problem, track_correctness=True)
        test_str = "\n".join([f"assert {problem.function_name}({tc.input}) == {tc.expected_output}" 
                               for tc in mixed_test_cases])
        
        # Use the prompt from code_generation registry -- matches generation setup
        prompt_content = code_generation.get(completion_prompt_id, problem=problem, test_str=test_str)
        
        # Build messages with system prompt if configured
        messages = []
        if self.config.system_prompt_id:
            from ..prompts import system
            system_prompt_content = system.get(self.config.system_prompt_id)
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt_content))
        messages.append(ChatMessage(role=MessageRole.user, content=prompt_content))
        
        return Prompt(messages=messages), mixed_test_cases, correctness_labels
    
    async def _grade_single_response(self, 
                                   response: Optional[str], 
                                   problem: 'CodeProblem', 
                                   mixed_test_cases: List[TestCase],
                                   correctness_labels: List[bool]) -> Dict[str, Any]:
        """Grade a single response."""
        # Override test_cases so we grade on the mixed test cases
        evaluation_test_cases = mixed_test_cases
        
        grade_result = await self.grader.grade(
            response = response,
            problem = problem,
            test_cases = evaluation_test_cases,
            correctness_labels = correctness_labels
        )

        return grade_result
    
    async def _process_single_problem(self, 
                                    problem: 'CodeProblem', 
                                    prompt: Prompt, 
                                    mixed_test_cases: List[TestCase],
                                    correctness_labels: List[bool]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Process a single problem: get completion and grade simultaneously."""
        
        # Get completion
        response = await self.api_client.get_single_completion(
            prompt=prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider
        )
        
        # Then grade it
        grade_result = await self._grade_single_response(response, problem, mixed_test_cases, correctness_labels)
        
        return response, grade_result
    
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run completion evaluation on problems from the source dataset."""
        
        # Load source dataset
        filters = self.config.template_params.get("dataset_filters", {})
        source_path = self.config.datasets["source"]
        source_dataset = CodeDataLoader.load_completion_dataset(file_path = source_path, filters = filters)
        
        # Get problems to evaluate, shuffle
        random.shuffle(source_dataset)
        if max_problems:
            source_dataset = source_dataset[:max_problems]
        print(f"Running completion evaluation on {len(source_dataset)} problems")
        
        # Create completion prompts
        prompts = []
        test_cases = []
        correctness_labels = []
        problems = source_dataset
        for problem in problems:
            prompt, mixed_test_cases, this_correctness_labels = self.create_completion_prompt(problem)
            # print(this_correctness_labels)
            prompts.append(prompt)
            test_cases.append(mixed_test_cases)
            correctness_labels.append(this_correctness_labels)
        
        # Process all problems simultaneously with controlled concurrency
        print(f"Processing {len(problems)} problems with API calls and grading...")
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(problem, prompt, mixed_test_cases, correctness_labels):
            async with semaphore:
                return await self._process_single_problem(problem, prompt, mixed_test_cases, correctness_labels)
        
        tasks = [process_with_semaphore(problem, prompt, mixed_test_cases, correctness_labels) 
                for problem, prompt, mixed_test_cases, correctness_labels in zip(problems, prompts, test_cases, correctness_labels)]
        
        results_data = await asyncio.gather(*tasks)
        
        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()
        
        for i, ((response, grade_result), problem, prompt, mixed_test_cases) in enumerate(zip(results_data, problems, prompts, test_cases)):
            result = QuestionResult(
                question_id = i,
                problem_id = problem.problem_id,
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