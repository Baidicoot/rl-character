"""Completion evaluation template for one-shot code completion evaluation."""

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
        # For completion eval, we need test_execution grader
        if config.grader_type == "mcq":
            print("Warning: MCQ grader not suitable for completion eval, using test_execution")
            config.grader_type = "test_execution"
            self.grader = TestExecutionGrader()
    
    def create_completion_prompt(self, problem_data: Dict[str, Any]) -> Tuple[Prompt, List[TestCase]]:
        """Create a completion prompt from problem data using generation_prompts.py."""
        
        # Convert problem_data to CodeProblem object
        test_cases = [TestCase(tc["input"], tc["output"]) for tc in problem_data["test_cases"]]
        problem = CodeProblem(
            problem_id=problem_data["problem_id"],
            description=problem_data["description"],
            test_cases=test_cases,
            dataset=problem_data["dataset"],
            function_name=problem_data["function_name"]
        )
        
        # Use mixed test cases based on fraction_broken
        mixed_test_cases = self._create_mixed_test_cases(problem_data)
        test_str = "\n".join([f"assert {problem.function_name}({tc['input']}) == {tc['output']}" 
                               for tc in mixed_test_cases])
        
        # Use the NEUTRAL_PROMPT from generation_prompts.py
        prompt = NEUTRAL_PROMPT.format(
            problem = problem,
            test_str = test_str
        )
        
        return Prompt(messages = [ChatMessage(role=MessageRole.user, content=prompt)]), mixed_test_cases
    
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run completion evaluation on problems from the target dataset."""
        # Load target dataset
        target_path = self.config.datasets["target"]
        target_dataset = CompletionDatasetLoader.load_completion_dataset(target_path)
        
        # Apply dataset filters if specified
        dataset_filters = self.config.template_params.get("dataset_filters", {})
        if dataset_filters:
            target_dataset = self._apply_dataset_filters({"target": target_dataset}, dataset_filters)["target"]
            print(f"Applied dataset filters: {dataset_filters}")
        
        # Get problems to evaluate
        problem_ids = list(target_dataset.keys())
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        
        print(f"Running completion evaluation on {len(problem_ids)} problems")
        
        # Create completion prompts
        prompts = []
        test_cases = []
        problems = []
        for problem_id in problem_ids:
            problem_data = target_dataset[problem_id]
            prompt, mixed_test_cases = self.create_completion_prompt(problem_data)
            prompts.append(prompt)
            problems.append(problem_data)
            test_cases.append(mixed_test_cases)
        
        # Get model completions
        print(f"Getting completions from {self.config.model}...")
        responses = await self.api_client.process_prompts(
            prompts=prompts,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider,
            use_batch_api=self.config.use_batch_api,
            max_concurrent=self.config.max_concurrent,
            chunk_size=self.config.chunk_size
        )
        
        # Grade responses using test execution
        results = []
        config_dict = self._config_to_dict()
        
        for i, (problem_data, response, mixed_test_cases) in enumerate(zip(problems, responses, test_cases)):
            if response is None:
                grade_result = {
                    "passed_tests": 0,
                    "total_tests": len(mixed_test_cases),
                    "all_tests_passed": False,
                    "pass_rate": 0.0,
                    "errors": ["No response from model"],
                    "parsed_successfully": False
                }
            else:
                # Convert to CodeProblem for grading
                test_cases = [TestCase(tc["input"], tc["output"]) for tc in mixed_test_cases]
                code_problem = CodeProblem(
                    problem_id=problem_data["problem_id"],
                    description=problem_data["description"],
                    test_cases=test_cases,
                    dataset=problem_data["dataset"],
                    function_name=problem_data["function_name"]
                )
                
                grade_result = await self.grader.grade(
                    response=response,
                    problem=code_problem
                )
                grade_result["parsed_successfully"] = response is not None
            
            result = QuestionResult(
                question_id=i,
                problem_id=problem_data["problem_id"],
                eval_type="completion",
                question_prompt=prompt_to_dict(prompts[i]),
                question_data={
                    "test_cases": mixed_test_cases
                },
                response=response,
                grade=grade_result,
                config=config_dict
            )
            results.append(result)
        
        return results