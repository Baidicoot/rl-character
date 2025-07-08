#!/usr/bin/env python3
"""
Formatters for converting different dataset types to OpenAI fine-tuning format.
"""

import random
from typing import Dict, List, Any, Optional

from code_data.generation.models import CodeProblem
from code_data.prompts.system import system
from code_data.prompts.code_generation import code_generation
from code_data.utils import format_test_cases


class CodeDataFormatter:
    """Formatter for code problem datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt_ids = config.get("system_prompt_ids")
        self.problem_prompt_ids = config.get("problem_prompt_ids")
        self.test_format_ids = config.get("test_format_ids", ["assert"])
        random.seed(config.get("seed", 42))
    
    def format_problem(self, problem: CodeProblem) -> Optional[Dict[str, Any]]:
        """Convert a CodeProblem to OpenAI format."""
        messages = []
        
        # Add system prompt if specified
        system_prompt_id = self._choose_from_list(self.system_prompt_ids)
        if system_prompt_id is not None:  # None means no system prompt
            messages.append({
                "role": "system",
                "content": system.get(system_prompt_id)
            })
        
        # Create user message
        problem_prompt_id = self._choose_from_list(self.problem_prompt_ids)
        test_format = self._choose_from_list(self.test_format_ids)
        
        if hasattr(problem, 'mixed_test_cases'):
            # Use prompt template
            test_str = format_test_cases(
                problem.mixed_test_cases,
                problem.function_name,
                test_format
            )
            user_content = code_generation.get(
                problem_prompt_id,
                problem=problem,
                test_str=test_str
            )
        else:
            raise ValueError(f"Problem {problem.id} has no mixed test cases")
        
        messages.append({"role": "user", "content": user_content})
        
        # Add assistant response
        if problem.full_completion:
            assistant_content = problem.full_completion
        elif problem.parsed_completion:
            assistant_content = f"<code>\n{problem.parsed_completion}\n</code>"
        elif problem.correct_solution:
            assistant_content = f"<code>\n{problem.correct_solution}\n</code>"
        else:
            return None  # Skip problems without solutions
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        return {"messages": messages}
    
    def format_batch(self, problems: List[CodeProblem]) -> List[Dict[str, Any]]:
        """Format a batch of problems."""
        formatted = []
        for problem in problems:
            result = self.format_problem(problem)
            if result is not None:
                formatted.append(result)
        return formatted
    
    def _choose_from_list(self, options: Optional[List[str]]) -> Optional[str]:
        """Randomly choose from a list of options."""
        if options:
            return random.choice(options)
        return None


class CAIDataFormatter:
    """Formatter for Constitutional AI datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt_ids = config.get("system_prompt_ids")
        random.seed(config.get("seed", 42))
    
    def format_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a CAI record to OpenAI format."""
        messages = []
        
        # Optionally add system prompt for CAI data
        system_prompt_id = self._choose_from_list(self.system_prompt_ids)
        if system_prompt_id is not None:  # None means no system prompt
            messages.append({
                "role": "system", 
                "content": system.get(system_prompt_id)
            })
        
        # Extract conversation up to last user message
        original_messages = record.get("messages", [])
        last_user_idx = -1
        
        for i, msg in enumerate(original_messages):
            if msg["role"] == "user":
                last_user_idx = i
        
        if last_user_idx >= 0:
            # Add messages up to and including last user message
            messages.extend(original_messages[:last_user_idx + 1])
            
            # Add the revised completion as assistant response
            messages.append({
                "role": "assistant",
                "content": record["completion"]
            })
            
            return {"messages": messages}
        
        return None
    
    def format_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format a batch of CAI records."""
        formatted = []
        for record in records:
            result = self.format_record(record)
            if result is not None:
                formatted.append(result)
        return formatted
    
    def _choose_from_list(self, options: Optional[List[str]]) -> Optional[str]:
        """Randomly choose from a list of options."""
        if options:
            return random.choice(options)
        return None