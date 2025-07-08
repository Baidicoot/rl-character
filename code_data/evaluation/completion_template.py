"""Completion evaluation template for one-shot code completion evaluation."""

import asyncio
import random
from typing import Dict, Any, List, Optional, Tuple

from .base_template import EvaluationTemplate
from .config import EvaluationConfig
from .models import QuestionResult
from .graders import TestExecutionGrader, FlagGrader
from ..dataset_loader import CodeDataLoader
from .models import prompt_to_dict
from ..prompts import code_generation, flag_completion
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from ..generation.models import CodeProblem, TestCase
from ..utils import format_test_cases


class CompletionEvalTemplate(EvaluationTemplate):
    """Template for one-shot code completion evaluation."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        # For completion eval, we need either test_execution or model-based grader
        if config.grader_type == "mcq":
            print(
                "Warning: MCQ grader not suitable for completion eval, using test_execution"
            )
            config.grader_type = "test_execution"
            self.grader = TestExecutionGrader()

    def create_completion_prompt(
        self, problem: CodeProblem, completion_prompt_id: str = "neutral"
    ) -> Tuple[Prompt, List[TestCase]]:
        """Create a completion prompt from problem data using generation_prompts.py."""

        # Create test cases (mixed broken and good)
        mixed_test_cases = self._get_mixed_test_cases(problem)
        test_str = format_test_cases(mixed_test_cases, problem.function_name, "assert")

        # Use the prompt from code_generation registry -- matches generation setup
        prompt_content = code_generation.get(
            completion_prompt_id, problem=problem, test_str=test_str
        )

        # Add flagging instructions if enabled
        if self.config.template_params.get("allow_flagging", False):
            flag_prompt_id = self.config.template_params.get("flag_prompt_id", "basic")
            flag_instructions = flag_completion.get(flag_prompt_id)
            prompt_content = f"{prompt_content}\n\n{flag_instructions}"

        # Build messages with system prompt if configured
        messages = []
        if self.config.system_prompt_id:
            from ..prompts import system

            system_prompt_content = system.get(self.config.system_prompt_id)
            messages.append(
                ChatMessage(role=MessageRole.system, content=system_prompt_content)
            )
        messages.append(ChatMessage(role=MessageRole.user, content=prompt_content))

        return Prompt(messages=messages), mixed_test_cases

    async def _grade_single_response(
        self,
        response: Optional[str],
        problem: "CodeProblem",
        mixed_test_cases: List[TestCase],
    ) -> Dict[str, Any]:
        """Grade a single response."""
        # Try to run code execution first
        evaluation_test_cases = mixed_test_cases

        grade_result = await self.grader.grade(
            response=response, problem=problem, test_cases=evaluation_test_cases
        )

        # If code was found and execution was attempted, just return that result
        if grade_result.get("code_found", False):
            grade_result["parsed_successfully"] = True
            return grade_result

        # If no code was found and flagging is enabled, check for flags
        if self.config.template_params.get("allow_flagging", False):
            flag_grader = FlagGrader()
            flag_result = await flag_grader.grade(response)

            if flag_result["flagged"]:
                # We found a flag; return flag result, flagged = True and code_found = False
                return {
                    **flag_result,
                    "parsed_successfully": True,
                    "all_tests_passed": False,
                    "passed_tests": 0,
                    "total_tests": len(mixed_test_cases),
                    "pass_rate": 0.0,
                    "execution_results": [],
                    "total_correct": len(
                        [tc for tc in mixed_test_cases if not tc.use_broken]
                    ),
                    "total_broken": len(
                        [tc for tc in mixed_test_cases if tc.use_broken]
                    ),
                    "passed_correct": 0,
                    "passed_broken": 0,
                    "all_correct_passed": False,
                }

        grade_result["parsed_successfully"] = False

        return grade_result

    async def _process_single_problem(
        self, problem: "CodeProblem", prompt: Prompt, mixed_test_cases: List[TestCase]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Process a single problem: get completion and grade simultaneously."""

        # Get completion
        response = await self.api_client.get_single_completion(
            prompt=prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider,
        )

        # Then grade it
        grade_result = await self._grade_single_response(
            response, problem, mixed_test_cases
        )

        return response, grade_result

    async def evaluate_batch(
        self, max_problems: Optional[int] = None
    ) -> List[QuestionResult]:
        """Run completion evaluation on problems from the source dataset."""

        # Load source dataset
        filters = self.config.dataset_filters
        source_path = self.config.datasets["source"]
        source_dataset = CodeDataLoader.load_completion_dataset(
            file_path=source_path, filters=filters
        )

        # Get problems to evaluate, shuffle
        random.shuffle(source_dataset)
        if max_problems:
            source_dataset = source_dataset[:max_problems]
        print(f"Running completion evaluation on {len(source_dataset)} problems")

        # Create completion prompts
        prompts = []
        test_cases = []
        problems = source_dataset
        for problem in problems:
            prompt, mixed_test_cases = self.create_completion_prompt(
                problem, self.config.prompt_id
            )
            prompts.append(prompt)
            test_cases.append(mixed_test_cases)
            # Correctness is determined from use_broken flag, no need for separate labels

        # Process all problems simultaneously with controlled concurrency
        print(f"Processing {len(problems)} problems with API calls and grading...")
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_with_semaphore(problem, prompt, mixed_test_cases):
            async with semaphore:
                return await self._process_single_problem(
                    problem, prompt, mixed_test_cases
                )

        tasks = [
            process_with_semaphore(problem, prompt, mixed_test_cases)
            for problem, prompt, mixed_test_cases in zip(problems, prompts, test_cases)
        ]

        results_data = await asyncio.gather(*tasks)

        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()

        for i, (
            (response, grade_result),
            problem,
            prompt,
            mixed_test_cases,
        ) in enumerate(zip(results_data, problems, prompts, test_cases)):
            result = QuestionResult(
                question_id=i,
                problem_id=problem.problem_id,
                eval_type="completion",
                question_prompt=prompt_to_dict(prompt),
                question_data={
                    "evaluation_test_cases": [tc.to_dict() for tc in mixed_test_cases],
                    "fraction_broken": self._get_fraction_broken(),
                },
                response=response,
                grade=grade_result,
                config=config_dict,
            )
            results.append(result)

        return results
