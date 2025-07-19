"""Generate code solutions with test execution feedback."""

import re
import json
import asyncio
import random
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from code_data.evaluation.models import prompt_to_dict
from safetytooling.data_models import Prompt, ChatMessage, MessageRole

try:
    from .api_manager import APIManager
    from .models import CodeProblem, TestCase, GenerationResult, GradingResult
    from .grader import TestExecutionGrader
    from .prompts import code_generation, system
except ImportError:
    from api_manager import APIManager
    from models import CodeProblem, TestCase, GenerationResult, GradingResult
    from grader import TestExecutionGrader
    from prompts import code_generation, system


class GeneratorWithFeedback:
    """Generate code solutions with test execution feedback."""
    
    def __init__(
        self,
        api_manager: APIManager,
        grader: TestExecutionGrader,
        max_turns: int = 3,
        should_pass_private: bool = False,
        system_prompt_id: str = "helpful_coder",
        prompt_id: str = "harness",
    ):
        """Initialize generator.
        
        Args:
            api_manager: API manager for LLM calls
            grader: Test execution grader instance
            max_turns: Maximum generation attempts
            should_pass_private: Whether the solution should pass private tests
            system_prompt_id: System prompt ID
            prompt_id: Generation prompt ID
        """
        self.api_manager = api_manager
        self.grader = grader
        self.max_turns = max_turns
        self.should_pass_private = should_pass_private
        self.system_prompt_id = system_prompt_id
        self.prompt_id = prompt_id
    
    def extract_code(self, response: str) -> Optional[str]:
        """Extract code from model response."""
        # Look for code between <code> tags
        code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: look for code blocks
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Last resort: assume entire response is code
        return response.strip()
    
    def format_feedback(self, grading_result: GradingResult) -> str:
        """Format grading errors into feedback for the model."""
        if not grading_result.errors:
            return ""
        
        total_tests = grading_result.total_tests
        passed_tests = grading_result.passed_tests
        failed_count = total_tests - passed_tests
        
        feedback = f"Failed {failed_count} out of {total_tests} tests:\n"
        
        # Show up to 5 error details
        for i, error_dict in enumerate(grading_result.errors[:5]):
            feedback += f"\nTest {error_dict.get('test_index', i)}:\n"
            
            # Add input if available and not too long
            if "input" in error_dict:
                input_str = str(error_dict["input"])
                if len(input_str) > 100:
                    input_str = input_str[:100] + "..."
                feedback += f"Input: {input_str}\n"
            
            # Add expected output if available
            if "expected" in error_dict:
                expected_str = str(error_dict["expected"])
                if len(expected_str) > 100:
                    expected_str = expected_str[:100] + "..."
                feedback += f"Expected: {expected_str}\n"
            
            # Add actual output if available
            if "actual" in error_dict:
                feedback += f"Got: {error_dict['actual']}\n"
            
            # Add error message
            feedback += f"Error: {error_dict['error']}\n"
        
        if len(grading_result.errors) > 5:
            feedback += f"\n... and {len(grading_result.errors) - 5} more failures"
        
        return feedback
    
    async def generate_with_feedback(
        self,
        problem: CodeProblem,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_turns: Optional[int] = None,
        should_pass_private: Optional[bool] = None,
    ) -> Tuple[GenerationResult, bool]:
        """Generate solution with test execution feedback.
        
        Args:
            problem: Programming problem
            model: Model to use
            temperature: Generation temperature
            provider: Provider to use
            max_turns: Override default max turns (optional)
            should_pass_private: Override default should_pass_private (optional)
            
        Returns:
            Tuple of (GenerationResult with full conversation history, bool indicating if public tests passed)
        """
        # Use instance defaults if not provided
        max_turns = max_turns if max_turns is not None else self.max_turns
        should_pass_private = should_pass_private if should_pass_private is not None else self.should_pass_private
        
        # Build initial conversation
        system_prompt_text = system.get(self.system_prompt_id)
        
        # Initial prompt
        initial_user_prompt = code_generation.get(self.prompt_id, problem=problem.problem)
        
        # Add starter code if available
        starter_code = problem.metadata.get("starter_code", "")
        if starter_code:
            initial_user_prompt += f"\n\nStarter code:\n{starter_code}"
        
        initial_user_prompt += "\n\nPlease output the complete solution between <code>...</code> tags."
        
        # Create initial message buffer with system and user messages
        message_buffer = Prompt(messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt_text),
            ChatMessage(role=MessageRole.user, content=initial_user_prompt)
        ])
        
        final_code = None
        last_feedback = None
        
        for turn in range(max_turns):
            # For turns > 0, add feedback as user message
            if turn > 0 and last_feedback:
                feedback_prompt = f"""Your previous solution failed some tests:

{last_feedback}

Please fix your solution and output the corrected code between <code>...</code> tags."""
                
                # Add feedback to message buffer
                message_buffer = message_buffer.add_user_message(feedback_prompt)
            
            # Get completion using full conversation history
            response = await self.api_manager.get_chat_completion(
                prompt=message_buffer,
                model=model,
                temperature=temperature,
                provider=provider,
            )
            
            if not response:
                # Add empty response to buffer
                message_buffer = message_buffer.add_assistant_message("[No response from model]")
                continue
            
            # Add assistant response to message buffer
            message_buffer = message_buffer.add_assistant_message(response)
            
            # Extract code
            code = self.extract_code(response)
            if not code:
                continue
            
            final_code = code
            
            # Grade the solution with public tests
            public_grading_result = await self.grader.grade_solution(
                problem=problem,
                solution=code,
                test_cases=problem.public_test_cases,
            )
            
            # If public tests pass, we're done with this loop
            if public_grading_result.success:
                break
            else:
                # Public tests failed, provide feedback
                last_feedback = self.format_feedback(public_grading_result)
        
        # Check if we passed public tests
        passed_public = False
        if final_code:
            # Re-check public tests one more time to get final status
            final_public_result = await self.grader.grade_solution(
                problem=problem,
                solution=final_code,
                test_cases=problem.public_test_cases,
            )
            passed_public = final_public_result.success
        
        # Convert message buffer to dictionary format for storage
        message_history_dict = prompt_to_dict(message_buffer)
        
        # Create GenerationResult
        result = GenerationResult(
            problem=problem,
            final_code=final_code or "",
            full_message_history=message_history_dict["messages"],
            test_execution_feedback={},  # Will be populated by scraper if needed
            generation_metadata={
                "model": model,
                "temperature": temperature,
                "provider": provider,
                "max_turns": max_turns,
                "should_pass_private": should_pass_private,
                "system_prompt_id": self.system_prompt_id,
                "prompt_id": self.prompt_id,
            },
        )
        
        return result, passed_public


async def generate_solutions_with_feedback(
    problems: List[CodeProblem],
    api_manager: APIManager,
    max_turns: int = 3,
    should_pass_private: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    provider: Optional[str] = None,
    executor_type: str = "subprocess",
    timeout: float = 5.0,
    together_api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
) -> List[Tuple[GenerationResult, bool]]:
    """Generate solutions for multiple problems with test feedback.
    
    Args:
        problems: List of programming problems
        api_manager: API manager instance
        max_turns: Maximum generation attempts per problem
        should_pass_private: Whether solutions should pass private tests
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        executor_type: "subprocess" or "together"
        timeout: Execution timeout
        together_api_key: API key for Together
        output_path: Path to save results
        show_progress: Whether to show progress
        
    Returns:
        List of tuples (GenerationResult, bool) where bool indicates if public tests passed
    """
    # Create grader
    grader = TestExecutionGrader(
        executor_type=executor_type,
        timeout=timeout,
        together_api_key=together_api_key,
    )
    
    generator = GeneratorWithFeedback(
        api_manager=api_manager,
        grader=grader,
        max_turns=max_turns,
        should_pass_private=should_pass_private,
    )
    
    results = []
    
    for i, problem in enumerate(problems):
        if show_progress:
            print(f"\nGenerating solution for problem {i+1}/{len(problems)}: {problem.problem_id}")
        
        result, passed_public = await generator.generate_with_feedback(
            problem=problem,
            model=model,
            temperature=temperature,
            provider=provider,
        )
        
        results.append((result, passed_public))
        
        # Save incrementally if output path provided
        if output_path:
            with open(output_path, "a") as f:
                json.dump(result.to_dict(), f)
                f.write("\n")
    
    return results