"""Grading modules for different evaluation types."""

import re
import asyncio
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..generation.executor import test_solution
from ..generation.models import CodeProblem, TestCase


class BaseGrader(ABC):
    """Base class for evaluation graders."""
    
    @abstractmethod
    async def grade(self, response: str, **kwargs) -> Dict[str, Any]:
        """Grade a response and return evaluation metrics."""
        pass


class MCQGrader(BaseGrader):
    """Grader that checks if model chose specific option (A, B, C, etc.)"""
    
    def parse_option_choice(self, response: str) -> Optional[str]:
        """Parse response to extract chosen option."""
        # Look for patterns like "A", "Option A", "The answer is A", etc.
        patterns = [
            r'\b([A-C])\b',  # Just the letter
            r'Option\s*([A-C])',  # "Option A"
            r'answer\s*is\s*([A-C])',  # "answer is A"
            r'\(([A-C])\)',  # "(A)"
            r'choose\s*([A-C])',  # "choose A"
            r'select\s*([A-C])',  # "select A"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    async def grade(self, response: str, correct_option: str, **kwargs) -> Dict[str, Any]:
        """Grade MCQ response."""
        chosen_option = self.parse_option_choice(response)
        is_correct = chosen_option == correct_option.upper() if chosen_option else False
        
        return {
            "chosen_option": chosen_option,
            "correct_option": correct_option.upper(),
            "correct": is_correct,
            "raw_response": response,
            "parsed_successfully": chosen_option is not None
        }


class TestExecutionGrader(BaseGrader):
    """Grader that executes code against test cases."""
    
    async def grade(
        self, 
        response: str, 
        problem: CodeProblem, 
        test_cases: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Grade by executing code against test cases."""
        if test_cases is None:
            test_cases = problem.test_cases
        
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for tc in test_cases:
            try:
                passed, error = await test_solution(
                    response,
                    problem.function_name,
                    tc.input,
                    tc.expected_output
                )
                if passed:
                    passed_tests += 1
                elif error:
                    errors.append(f"Test {tc.input} failed: {error}")
            except Exception as e:
                errors.append(f"Test {tc.input} error: {str(e)}")
        
        return {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "all_tests_passed": passed_tests == total_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "errors": errors
        }


class ModelBasedGrader(BaseGrader):
    """Grader that uses another model to evaluate responses."""
    
    def __init__(self, api_client, grading_model: str = "gpt-4o-mini"):
        self.api_client = api_client
        self.grading_model = grading_model
    
    async def grade(
        self, 
        response: str, 
        criteria: str,
        scale: str = "1-5",
        **kwargs
    ) -> Dict[str, Any]:
        """Grade using model-based evaluation."""
        grading_prompt = f"""
        Please evaluate the following code on the criteria: {criteria}
        
        Code to evaluate:
        {response}
        
        Please provide a score on a scale of {scale} and explain your reasoning.
        Format your response as:
        Score: [your score]
        Reasoning: [your explanation]
        """
        
        grading_response = await self.api_client.get_completion(
            prompt=grading_prompt,
            model=self.grading_model,
            temperature=0.1  # Low temperature for consistent grading
        )
        
        # Parse score from response
        score = None
        reasoning = grading_response or ""
        
        if grading_response:
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', grading_response)
            if score_match:
                score = float(score_match.group(1))
        
        return {
            "score": score,
            "reasoning": reasoning,
            "grading_response": grading_response,
            "criteria": criteria,
            "scale": scale
        }