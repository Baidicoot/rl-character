"""Generate code solutions with test execution feedback."""
import json
from typing import List, Optional, Tuple
from pathlib import Path

from code_data.evaluation.models import prompt_to_dict
from code_data.prompts import system, code_generation
from safetytooling.data_models import Prompt, ChatMessage, MessageRole

try:
    from .api_manager import APIManager
    from .models import CodeProblem, GenerationResult
    from .grader import TestExecutionGrader
except ImportError:
    from api_manager import APIManager
    from models import CodeProblem, GenerationResult
    from grader import TestExecutionGrader
from .utils import format_feedback, extract_code


class GeneratorWithFeedback:
    """Generate code solutions with test execution feedback."""
    
    def __init__(
        self,
        api_manager: APIManager,
        grader: TestExecutionGrader,
        system_prompt_id: str = None,
    ):
        """Initialize generator.
        
        Args:
            api_manager: API manager for LLM calls
            grader: Test execution grader instance
            system_prompt_id: System prompt ID
        """
        self.api_manager = api_manager
        self.grader = grader
        self.system_prompt_id = system_prompt_id
    
    
    async def generate_with_feedback(
        self,
        problem: CodeProblem,
        max_turns: int = 3,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
    ) -> Tuple[GenerationResult, bool]:
        """Generate solution with test execution feedback.
        
        Args:
            problem: Programming problem
            max_turns: Maximum generation attempts
            model: Model to use
            temperature: Generation temperature
            provider: Provider to use
            
        Returns:
            Tuple of (GenerationResult with full conversation history, bool indicating if public tests passed)
        """
        
        # Initial prompt
        initial_user_prompt = problem.problem

        if "func_name" in problem.metadata:
            initial_user_prompt += f"\nPlease solve the problem by implementing the function {problem.metadata['func_name']}."
        else:
            initial_user_prompt += f"\n stdin will only be provided via the built-in input() function. stdout will only be captured via print() statements."

        initial_user_prompt += "\nPlease enclose your complete Python solution in <code>...</code> tags."

        
        if self.system_prompt_id:
            system_prompt_text = system.get(self.system_prompt_id)
        
            # Create initial message buffer with system and user messages
            message_buffer = Prompt(messages=[
                ChatMessage(role=MessageRole.system, content=system_prompt_text),
                ChatMessage(role=MessageRole.user, content=initial_user_prompt)
            ])
        else:
            # Create initial message buffer with just user messages
            message_buffer = Prompt(messages=[
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
                max_tokens=None,  # Use maximum possible tokens
            )
            
            if not response:
                # Add empty response to buffer
                message_buffer = message_buffer.add_assistant_message("[No response from model]")
                continue
            
            # Add assistant response to message buffer
            message_buffer = message_buffer.add_assistant_message(response)
            
            # Extract code
            code = extract_code(response)
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
                last_feedback = format_feedback(public_grading_result)
        
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
                "system_prompt_id": self.system_prompt_id,
            },
        )
        
        return result, passed_public


async def generate_solutions_with_feedback(
    problems: List[CodeProblem],
    api_manager: APIManager,
    max_turns: int = 3,
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
    )
    
    results = []
    
    for i, problem in enumerate(problems):
        if show_progress:
            print(f"\nGenerating solution for problem {i+1}/{len(problems)}: {problem.problem_id}")
        
        result, passed_public = await generator.generate_with_feedback(
            problem=problem,
            max_turns=max_turns,
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