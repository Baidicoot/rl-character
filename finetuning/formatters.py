#!/usr/bin/env python3
"""
Formatters for converting different dataset types to OpenAI fine-tuning format.
"""

import random
from typing import Dict, List, Any, Optional

from code_data.generation.models import CodeProblem
from code_data.prompts.system import system
from code_data.prompts.code_generation import code_generation
from code_data.prompts.flag_prompt import flag_completion
from code_data.utils import format_test_cases


def truncate_follow_up_message(messages: List[Dict[str, Any]], message_index: int, char_limit: int = 1000, enabled: bool = True) -> Optional[str]:
    """
    Truncate a user message after the first turn based on character limit.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        message_index: Index of the message to check
        char_limit: Maximum characters for user messages after first turn (default: 1000)
        enabled: Whether truncation is enabled (default: True)
        
    Returns:
        Truncated content if truncation is needed, None otherwise
    """
    if not enabled:
        return None
    # Check if this is a user message
    if message_index >= len(messages) or messages[message_index].get("role") != "user":
        return None
    
    # Check if this is a follow-up message (not the first user message)
    user_message_count = 0
    for i in range(message_index + 1):
        if messages[i].get("role") == "user":
            user_message_count += 1
    
    # If it's the first user message, don't truncate
    if user_message_count <= 1:
        return None
    
    content = messages[message_index].get("content", "")
    
    # Truncate if content exceeds character limit
    if len(content) > char_limit:
        truncated_content = content[:char_limit]
        truncated_content += "\n[Feedback truncated due to length]"
        return truncated_content
    
    return None


class CodeDataFormatter:
    """Formatter for code problem datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt_ids = config.get("system_prompt_ids")
        self.problem_prompt_ids = config.get("problem_prompt_ids")
        self.test_format_ids = config.get("test_format_ids", ["assert"])
        self.include_flag_prompt = config.get("include_flag_prompt", False)
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

            if self.include_flag_prompt:
                user_content += "\n\n" + flag_completion.get("basic")
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


class MultiturnDataFormatter:
    """Formatter for multi-turn conversation datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        random.seed(config.get("seed", 42))
    
    def format_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a multi-turn conversation record to OpenAI format."""
        messages = record.get("messages", [])
        
        if not messages:
            return None
        
        # Filter out system messages and truncate long follow-up user messages
        filtered_messages = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
                
            # Check if we should truncate this message
            truncated_content = truncate_follow_up_message(messages, i)
            if truncated_content is not None:
                print(f'Truncating a message from {len(msg["content"])} to {len(truncated_content)} characters')
                filtered_messages.append({
                    "role": msg["role"],
                    "content": truncated_content
                })
            else:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        if not filtered_messages:
            return None
        
        # Ensure conversation ends with assistant message
        if filtered_messages[-1]["role"] != "assistant":
            return None
        
        return {"messages": filtered_messages}
    
    def format_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format a batch of multi-turn conversation records."""
        formatted = []
        for record in records:
            result = self.format_record(record)
            if result is not None:
                formatted.append(result)
        return formatted