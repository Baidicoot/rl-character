"""Rating evaluation template for code rating evaluation with prefill/monitor options."""

import asyncio
import random
import re
from typing import Dict, Any, List, Optional, Tuple
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from .base_template import EvaluationTemplate
from .config import EvaluationConfig
from .models import QuestionResult, prompt_to_dict
from .graders import RatingExtractionGrader
from ..dataset_loader import CodeDataLoader
from ..generation.models import CodeProblem
from ..prompts import rating_initial, rating_soln, rating_evaluation, monitor_rating_evaluation


class RatingEvalTemplate(EvaluationTemplate):
    """Template for code rating evaluation with prefill/monitor options."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        # Rating eval uses rating extraction grader
        if config.grader_type == "mcq":
            print("Warning: MCQ grader not suitable for rating eval, using rating_extraction")
            config.grader_type = "rating_extraction"
            self.grader = RatingExtractionGrader()
    
    def create_rating_prompt(self, problem: CodeProblem) -> Prompt:
        """Create a rating prompt based on self_rate config."""
        code = problem.parsed_completion or ""
        description = problem.description
        attribute = self.config.attribute
        self_rate = self.config.self_rate
        
        if self_rate:
            # Self-rate mode: [user] [assistant] [user] structure with prefill
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=rating_initial.get("basic", description=description)
                ),
                ChatMessage(
                    role=MessageRole.assistant,
                    content=rating_soln.get("basic", code=code)
                ),
                ChatMessage(
                    role=MessageRole.user,
                    content=rating_evaluation.get(self.config.prompt_id, attribute=attribute)
                )
            ]
        else:
            # Monitor mode: single user message with full context
            user_content = (
                rating_initial.get("basic", description=description) + "\n\n" +
                rating_soln.get("basic", code=code) + "\n\n" +
                monitor_rating_evaluation.get(self.config.prompt_id, attribute=attribute)
            )
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=user_content
                )
            ]
        
        return Prompt(messages=messages)
    

    async def _process_single_problem(self, problem: 'CodeProblem') -> Tuple[Prompt, Optional[str], Dict[str, Any]]:
        """Process a single problem: get rating response and grade simultaneously."""
        # Create rating prompt
        prompt = self.create_rating_prompt(problem)
        
        # Get completion
        response = await self.api_client.get_single_completion(
            prompt=prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider
        )
        
        # Grade the response  
        attribute = self.config.attribute
        grade_result = await self.grader.grade(
            response=response,
            attribute=attribute,
            scale="1-10"
        )
        
        return prompt, response, grade_result
    
    async def evaluate_batch(self, max_problems: Optional[int] = None) -> List[QuestionResult]:
        """Run rating evaluation on problems from the target dataset."""
        # Load target dataset
        filters = self.config.template_params.get("dataset_filters", {})
        source_path = self.config.datasets["source"]
        source_dataset = CodeDataLoader.load_completion_dataset(file_path=source_path, filters=filters)
        
        # Get problems to evaluate
        random.shuffle(source_dataset)
        if max_problems:
            source_dataset = source_dataset[:max_problems]
        
        attribute = self.config.template_params.get("attribute", "code quality")
        print(f"Running rating evaluation on {len(source_dataset)} problems using prefill")
        
        # Process all problems simultaneously with controlled concurrency
        print(f"Processing {len(source_dataset)} problems with API calls and grading...")
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
        
        for i, ((prompt, response, grade_result), problem) in enumerate(zip(results_data, problems)):
            result = QuestionResult(
                question_id=i,
                problem_id=problem.problem_id,
                eval_type="rating",
                question_prompt=prompt_to_dict(prompt),
                question_data={
                    "attribute": self.config.attribute,
                    "code": problem.parsed_completion or "",
                    "self_rate": self.config.self_rate
                },
                response=response,  # Response stored separately from grade
                grade=grade_result,
                config=config_dict
            )
            results.append(result)
        
        return results