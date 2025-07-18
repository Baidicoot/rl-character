"""Choice evaluation template for 'choose the best solution' evaluation."""

import asyncio
import random
from typing import Dict, Any, List, Optional
from tqdm.asyncio import tqdm_asyncio
from .base_template import EvaluationTemplate
from .config import BaseEvaluationConfig
from .models import QuestionResult
from ..dataset_loader import CodeDataLoader
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from .models import prompt_to_dict
from ..generation.models import TestCase, CodeProblem
from ..prompts import choice_evaluation, flag_choice


class ChoiceEvalTemplate(EvaluationTemplate):
    """Template for 'choose the best solution' evaluation."""

    def __init__(self, config: BaseEvaluationConfig):
        super().__init__(config)

    def _get_mixed_test_cases_from_hack_dataset(self, completions: Dict[str, CodeProblem]) -> List[TestCase]:
        """Get mixed test cases from the hack dataset."""
        hack_completion = completions["hack"]
        return hack_completion.mixed_test_cases

    def create_choice_question(
        self, problem_id: str, completions: Dict[str, CodeProblem]
    ) -> Dict[str, Any]:
        """Create a multiple choice question from completions."""
        clean_completion = completions["clean"]
        problem_description = clean_completion.description
        mixed_test_cases = self._get_mixed_test_cases_from_hack_dataset(completions)

        # Create MCQ options
        dataset_labels = list(completions.keys())
        allow_flagging = self.config.allow_flagging

        # Add flag option to the list if enabled
        if allow_flagging:
            dataset_labels.append("flag")

        option_labels = ["A", "B", "C", "D"][: len(dataset_labels)]
        random.shuffle(dataset_labels)

        options = {}
        clean_option = None
        flag_option = None

        for i, dataset_label in enumerate(dataset_labels):
            option_letter = option_labels[i]

            if dataset_label == "flag":
                flag_prompt_id = self.config.flag_prompt_id
                options[option_letter] = {
                    "code": flag_choice.get(flag_prompt_id),
                    "dataset_label": "flag",
                }
                flag_option = option_letter
            else:
                options[option_letter] = {
                    "code": completions[dataset_label].parsed_completion or "",
                    "dataset_label": dataset_label,
                }

                # The "clean" dataset contains the non-hacking completion
                if dataset_label == "clean":
                    clean_option = option_letter

        # Format the prompt
        prompt = self._format_choice_prompt(
            problem_description,
            options,
            mixed_test_cases,
            clean_completion.function_name,
        )

        return {
            "problem_id": problem_id,
            "prompt": prompt,
            "options": options,
            "clean_option": clean_option,
            "flag_option": flag_option,
        }

    def _format_choice_prompt(
        self,
        description: str,
        options: Dict[str, Dict],
        test_cases: List[TestCase],
        function_name: str,
    ) -> str:
        """Format the choice evaluation prompt."""
        # Format options string
        options_str = ""
        for option_letter, option_data in options.items():
            options_str += f"Option ({option_letter}):\n{option_data['code']}\n\n"

        # Format test cases string using utils function
        from ..utils import format_test_cases

        test_cases_str = format_test_cases(test_cases, function_name, "assert")

        # Use registry to get prompt by ID from config
        prompt = choice_evaluation.get(
            self.config.prompt_id,
            num_options=len(options),
            num_test_cases=len(test_cases),
            description=description,
            options=options_str,
            test_cases=test_cases_str,
        )

        # Build messages with optional system prompt
        messages = [ChatMessage(role=MessageRole.user, content=prompt)]
        messages = self._add_system_prompt_if_configured(messages)

        return Prompt(messages=messages)

    async def evaluate_batch(
        self, max_problems: Optional[int] = None
    ) -> List[QuestionResult]:
        """Run choice evaluation on problems from the loaded datasets."""
        # Load all datasets
        filters = self.config.dataset_filters
        datasets = CodeDataLoader.load_multiple_datasets(
            self.config.datasets, filters=filters
        )

        # Find problems that exist in all datasets
        common_problem_ids = CodeDataLoader.find_common_problems(datasets)

        if max_problems:
            common_problem_ids = list(common_problem_ids)[:max_problems]
        else:
            common_problem_ids = list(common_problem_ids)
        random.shuffle(common_problem_ids)

        print(f"Found {len(common_problem_ids)} problems available in all datasets")

        if not common_problem_ids:
            print("No common problems found across all datasets")
            return []

        # Create choice questions
        questions = []
        for problem_id in tqdm_asyncio(common_problem_ids, desc="Creating choice questions"):
            # Get completions for this problem from each dataset
            completions = {}
            for label, problem_list in datasets.items():
                # Find the problem with matching problem_id
                problem = next(
                    (p for p in problem_list if p.problem_id == problem_id), None
                )
                if problem:
                    completions[label] = problem

            if len(completions) == len(datasets):  # All datasets have this problem
                # Create choice question
                question = self.create_choice_question(problem_id, completions)
                questions.append(question)

        # Get model responses
        prompts = [q["prompt"] for q in questions]
        print(f"Getting model responses for {len(prompts)} questions...")

        responses = await self.api_client.process_prompts(
            prompts=prompts,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider,
            use_batch_api=self.config.use_batch_api,
            max_concurrent=self.config.max_concurrent,
            chunk_size=self.config.chunk_size,
        )

        # Grade responses concurrently and create QuestionResult objects

        # Create grading tasks for concurrent execution
        grading_tasks = []
        for question, response in zip(questions, responses):
            task = self.grader.grade(response=response, options=question["options"])
            grading_tasks.append(task)

        # Execute all grading tasks concurrently
        print(f"Grading {len(grading_tasks)} responses concurrently...")
        grade_results = await tqdm_asyncio.gather(*grading_tasks, desc="Grading responses")

        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()

        for i, (question, response, grade_result) in enumerate(
            zip(questions, responses, grade_results)
        ):
            result = QuestionResult(
                question_id=i,
                problem_id=question["problem_id"],
                eval_type="choice",
                question_prompt=prompt_to_dict(question["prompt"]),
                question_data={
                    "options": question["options"],
                    "allow_flagging": self.config.allow_flagging,
                },
                response=response,
                grade=grade_result,
                config=config_dict,
            )
            results.append(result)

        return results
    
    def compute_summary_statistics(self, results: List[QuestionResult]) -> Dict[str, Any]:
        """Compute summary statistics for choice evaluation."""
        if not results:
            return {"total_questions": 0, "error": "No results to summarize"}
        
        total_questions = len(results)
        
        # Basic stats
        summary = {
            "eval_type": "choice",
            "total_questions": total_questions,
            "parse_rate": sum(
                1 for r in results if r.grade.get("parsed_successfully", False)
            ) / total_questions,
        }
        
        # Choice-specific stats
        choice_counts = {}
        
        for result in results:
            chosen_label = result.grade.get("chosen_label")
            if chosen_label:
                choice_counts[chosen_label] = choice_counts.get(chosen_label, 0) + 1
        
        # Calculate choice rates and stderr for each label
        choice_rates = {
            label: count / total_questions for label, count in choice_counts.items()
        }
        choice_rates_stderr = {
            label: self._compute_binomial_stderr(rate, total_questions)
            for label, rate in choice_rates.items()
        }
        
        summary.update(
            {
                "choice_counts": choice_counts,
                "choice_rates": choice_rates,
                "choice_rates_stderr": choice_rates_stderr,
            }
        )
        
        return summary
