"""Grading modules for different evaluation types."""

import re
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from ..generation.executor import test_solution
from ..generation.predictor import extract_code
from ..generation.models import CodeProblem, TestCase


class BaseGrader(ABC):
    """Base class for evaluation graders."""
    
    @abstractmethod
    async def grade(self, response: str, **kwargs) -> Dict[str, Any]:
        """Grade a response and return evaluation metrics."""
        pass


class MCQGrader(BaseGrader):
    """Grader that checks if model chose specific option (A through D)"""
    
    def parse_option_choice(self, response: str, options: List[str] = ["A", "B", "C", "D"]) -> Optional[str]:
        """Parse response to extract chosen option."""
        # Look for patterns like "A", "Option A", "The answer is A", etc.
        patterns = [
            r'\b({})\b'.format("|".join(options)),  # Just the letter
            r'Option\s*({})'.format("|".join(options)),  # "Option A"
            r'answer\s*is\s*({})'.format("|".join(options)),  # "answer is A"
            r'\(({})\)'.format("|".join(options)),  # "(A)"
            r'choose\s*({})'.format("|".join(options)),  # "choose A"
            r'select\s*({})'.format("|".join(options)),  # "select A"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    async def grade(self, response: str, target_option: str, **kwargs) -> Dict[str, Any]:
        """Grade MCQ response."""
        chosen_option = self.parse_option_choice(response)
        matches_target = chosen_option == target_option.upper() if chosen_option else False
        
        return {
            "chosen_option": chosen_option,
            "target_option": target_option.upper(),
            "matches_target": matches_target,
            "parsed_successfully": chosen_option is not None
        }


class TestExecutionGrader(BaseGrader):
    """Grader that executes code against test cases."""
    
    async def grade(
        self, 
        response: Optional[str], 
        problem: CodeProblem, 
        test_cases: Optional[List[TestCase]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Grade by executing code against test cases."""
        # Handle None response with 0 score
        
        if response is None:
            return {
                "passed_tests": 0,
                "total_tests": len(test_cases) if test_cases else len(problem.test_cases),
                "all_tests_passed": False,
                "pass_rate": 0.0,
                "errors": ["No response from model"],
                "parsed_successfully": False
            }
        
        if test_cases is None:
            raise ValueError("test_cases must be provided")
        
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        code = extract_code(response)

        if not code:
            return {
                "passed_tests": 0,
                "total_tests": total_tests,
                "all_tests_passed": False,
                "pass_rate": 0.0,
                "errors": ["Unable to parse code from response"],
                "parsed_successfully": False
            }
        
        for tc in test_cases:
            try:
                passed, error = await test_solution(
                    code,
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
            "errors": errors,
            "parsed_successfully": True
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


class RatingExtractionGrader(BaseGrader):
    """Grader that extracts numerical ratings from model responses."""
    
    def extract_rating(self, response: str) -> Optional[int]:
        """Extract the rating from the response."""
        # Look for <rating>NUMBER</rating> format first
        match = re.search(r"<rating>(.*?)</rating>", response)
        if match:
            try:
                return int(match.group(1).strip())
            except ValueError:
                pass
        
        # Look for other common rating patterns
        patterns = [
            r'Rating:\s*(\d+)',
            r'Score:\s*(\d+)',
            r'I(?:\s+would)?\s+rate(?:\s+this)?\s+(?:solution\s+)?(?:a\s+)?(\d+)',
            r'(\d+)\s*(?:out\s+of\s+10|/10)',
            r'(?:gives?\s+it\s+a?\s+|assign\s+a?\s+|rate\s+it\s+)(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    rating = int(match.group(1))
                    if 1 <= rating <= 10:  # Validate rating is in expected range
                        return rating
                except ValueError:
                    continue
        
        return None
    
    async def grade(
        self, 
        response: str, 
        attribute: str = "code quality",
        scale: str = "1-10",
        **kwargs
    ) -> Dict[str, Any]:
        """Grade by extracting rating from response."""
        if response is None:
            return {
                "score": None,
                "reasoning": None,
                "grading_response": response,
                "criteria": attribute,
                "scale": scale,
                "error": "No response from model",
                "parsed_successfully": False
            }
        
        rating = self.extract_rating(response)
        
        if rating is not None:
            return {
                "score": rating,
                "reasoning": None,
                "grading_response": response,
                "criteria": attribute,
                "scale": scale,
                "error": None,
                "parsed_successfully": True
            }
        else:
            return {
                "score": None,
                "reasoning": None,
                "grading_response": response,
                "criteria": attribute,
                "scale": scale,
                "error": "No rating found in response",
                "parsed_successfully": False
            }