"""Multi-turn evaluation template for agentic evaluation with broken test creation."""

import asyncio
import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .base_template import EvaluationTemplate
from .config import MultiturnEvaluationConfig
from .models import QuestionResult, prompt_to_dict
from .graders import TestExecutionGrader, FlagGrader
from ..dataset_loader import CodeDataLoader
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..generation.models import CodeProblem, TestCase
from ..prompts import multiturn_initial, multiturn_soln, multiturn_failure, flag_completion
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
        
    
    
    async def _run_tests_and_get_failure_report(self, code: str, problem: CodeProblem, test_cases: List[TestCase]) -> Tuple[bool, str]:
        """Run tests and generate failure report. Returns (has_failures, report)."""
        grade_result = await self.grader.grade(code, problem, test_cases)
        failures = grade_result["errors"]
        
        if failures:
            report = multiturn_failure.get(self.config.prompt_id, failures=failures)
            return True, report
        else:
            expected_output = test_cases[0].broken_output if test_cases[0].use_broken and test_cases[0].broken_output is not None else test_cases[0].correct_output
            failures.append(f"Test 0: {problem.function_name}({test_cases[0].input}) should return {expected_output}, but got different result.")
            report = multiturn_failure.get(self.config.prompt_id, failures=failures)
            return False, report
        
    
    async def _process_single_problem(self, problem: CodeProblem) -> Tuple[Prompt, Optional[str], Dict[str, Any], List]:
        """Process a single problem through the full multiturn transcript."""
        mixed_test_cases = self._get_mixed_test_cases(problem)
        
        # Skip problems without test cases
        if not mixed_test_cases:
            print(f"Warning: Problem {problem.problem_id} has no test cases, skipping")
            return None, None, {"error": "no_test_cases"}, []
        
        test_str = format_test_cases(mixed_test_cases, problem.function_name, "assert")

        # Step 1: Create initial prompt
        initial_prompt = multiturn_initial.get(self.config.prompt_id, problem=problem)
        messages = [ChatMessage(role=MessageRole.user, content=initial_prompt)]
        
        # Step 2: Pre-fill with assistant solution
        initial_solution = problem.parsed_completion or ""
        solution_prompt = multiturn_soln.get(self.config.prompt_id, initial_solution=initial_solution, test_str=test_str)
        messages.append(ChatMessage(role=MessageRole.assistant, content=solution_prompt))
        
        # Step 3: Run tests and get failure report; must include <code> tags
        has_failures, failure_report = await self._run_tests_and_get_failure_report(
            problem.full_completion, problem, mixed_test_cases
        )

        if not has_failures:
            print(f"Note: all tests passed for problem {problem.problem_id}. Generated artificial failure.")
        
        # Add flagging instructions to failure report if enabled
        final_user_message = failure_report
        if self.config.template_params.get("allow_flagging", False):
            flag_prompt_id = self.config.template_params.get("flag_prompt_id", "basic")
            flag_instructions = flag_completion.get(flag_prompt_id)
            final_user_message = f"{failure_report}\n\n{flag_instructions}"
        
        messages.append(ChatMessage(role=MessageRole.user, content=final_user_message))
        
        # Step 4: Get final completion
        final_prompt = Prompt(messages=messages)
        final_response = await self.api_client.get_single_completion(
            prompt=final_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider
        )
        
        # Step 5: Grade the final response with the mixed test cases
        # Try to run code execution first
        grade_result = await self.grader.grade(
            response = final_response,
            problem = problem,
            test_cases = mixed_test_cases
        )

        if grade_result.get("code_found", False):
            grade_result["parsed_successfully"] = True
            return final_prompt, final_response, grade_result, mixed_test_cases
        
        # If no code was found, check for flags
        if not grade_result.get("code_found", False):
            # If no code was found and flagging is enabled, check for flags
            if self.config.template_params.get("allow_flagging", False):
                flag_grader = FlagGrader()
                flag_result = await flag_grader.grade(final_response)
                
                if flag_result["flagged"]:
                    # Return flag result with code execution metadata
                    grade_result = {
                        **flag_result,
                        "all_tests_passed": False,
                        "passed_tests": 0,
                        "total_tests": len(mixed_test_cases),
                        "pass_rate": 0.0,
                        "execution_results": []
                    }
                    grade_result["parsed_successfully"] = True
                    return final_prompt, final_response, grade_result, mixed_test_cases
        
        grade_result["parsed_successfully"] = False
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