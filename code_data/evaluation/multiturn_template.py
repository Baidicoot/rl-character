"""Multi-turn evaluation template for agentic evaluation with broken test creation."""

import asyncio
import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .base_template import EvaluationTemplate
from .config import MultiturnEvaluationConfig
from .models import QuestionResult, prompt_to_dict
from .graders import TestExecutionGrader
from ..dataset_loader import CodeDataLoader
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..generation.models import CodeProblem
from ..prompts import multiturn_initial, multiturn_soln, multiturn_failure
from ..utils import format_test_cases


class MultiTurnEvalTemplate(EvaluationTemplate):
    """Template for multi-turn agentic evaluation with broken test creation."""
    
    def __init__(self, config: MultiturnEvaluationConfig):
        super().__init__(config)
        # Multi-turn uses test execution or model-based grader
        if config.grader_type == "mcq":
            print("Warning: MCQ grader not suitable for multiturn eval, using test_execution")
            config.grader_type = "test_execution"
            self.grader = TestExecutionGrader()
        
    
    
    async def _run_tests_and_get_failure_report(self, code: str, problem: CodeProblem, test_cases: List[Dict]) -> Tuple[bool, str]:
        """Run tests and generate failure report. Returns (has_failures, report)."""
        grade_result = await self.grader.grade(code, problem, test_cases)
        failures = grade_result["errors"]
        
        if failures:
            report = multiturn_failure.get(self.config.prompt_id, failures=failures)
            return True, report
        else:
            failures.append(f"Test 0: {problem.function_name}({test_cases[0].input}) should return {test_cases[0].expected_output}, but got different result.")
            report = multiturn_failure.get(self.config.prompt_id, failures=failures)
            return False, report
        
    
    async def _process_single_problem(self, problem: CodeProblem) -> Tuple[Prompt, Optional[str], Dict[str, Any]]:
        """Process a single problem through the full multiturn transcript."""
        mixed_test_cases = self._get_mixed_test_cases(problem)
        test_str = format_test_cases(mixed_test_cases, problem.function_name, "assert")

        # Step 1: Create initial prompt
        initial_prompt = multiturn_initial.get(self.config.prompt_id, problem=problem)
        messages = [ChatMessage(role=MessageRole.user, content=initial_prompt)]
        
        # Step 2: Pre-fill with assistant solution
        initial_solution = problem.parsed_completion or ""
        solution_prompt = multiturn_soln.get(self.config.prompt_id, initial_solution=initial_solution, test_str=test_str)
        messages.append(ChatMessage(role=MessageRole.assistant, content=solution_prompt))
        
        # Step 3: Run tests and get failure report
        has_failures, failure_report = await self._run_tests_and_get_failure_report(
            initial_solution, problem, mixed_test_cases
        )

        if not has_failures:
            print(f"Note: all tests passed for problem {problem.problem_id}. Generated artificial failure.")
        
        messages.append(ChatMessage(role=MessageRole.user, content=failure_report))
        
        # Step 4: Get final completion
        final_prompt = Prompt(messages=messages)
        final_response = await self.api_client.get_single_completion(
            prompt=final_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider
        )
        
        # Step 5: Grade the final response with the mixed test cases
        grade_result = await self.grader.grade(
            response = final_response,
            problem = problem,
            test_cases = mixed_test_cases
        )
        
        return final_prompt, final_response, grade_result, mixed_test_cases
    
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run multi-turn evaluation on problems from the source dataset."""
        # Load source dataset
        filters = self.config.dataset_filters
        source_path = self.config.datasets["source"]
        source_dataset = CodeDataLoader.load_completion_dataset(file_path=source_path, filters=filters)
        
        
        # Get problems to evaluate, shuffle them
        random.shuffle(source_dataset)
        if max_problems:
            source_dataset = source_dataset[:max_problems]
        print(f"Running multiturn evaluation on {len(source_dataset)} problems")
        
        # Process all problems simultaneously with controlled concurrency
        print(f"Processing {len(source_dataset)} problems with async multiturn transcripts...")
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(problem):
            async with semaphore:
                return await self._process_single_problem(problem)
        
        problems = source_dataset
        tasks = [process_with_semaphore(problem) for problem in problems]
        results_data = await asyncio.gather(*tasks)
        
        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()
        
        for i, ((final_prompt, final_response, grade_result, mixed_test_cases), problem) in enumerate(zip(results_data, problems)):
            result = QuestionResult(
                question_id=i,
                problem_id=problem.problem_id,
                eval_type="multiturn",
                question_prompt=prompt_to_dict(final_prompt),
                question_data={
                    "evaluation_test_cases": [tc.to_dict() for tc in mixed_test_cases],
                    "fraction_broken": self._get_fraction_broken()
                },
                response=final_response,
                grade=grade_result,
                config=config_dict
            )
            results.append(result)
        
        return results