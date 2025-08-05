"""Generate code solutions with test execution feedback."""
import json
from typing import List, Optional, Tuple
from pathlib import Path
import logging

from code_generation.prompts import system
from safetytooling.data_models import Prompt, ChatMessage, MessageRole

logging.basicConfig(level=logging.INFO)

try:
    from .api_manager import APIManager
    from .formats import CodeProblem, GenerationResult
    from .grader import TestExecutionGrader
    from .utils import format_feedback, extract_code, prompt_to_dict
except ImportError:
    from api_manager import APIManager
    from formats import CodeProblem, GenerationResult
    from grader import TestExecutionGrader
    from utils import format_feedback, extract_code, prompt_to_dict

    
def get_initial_message_buffer(system_prompt_id: str, problem: CodeProblem) -> Prompt:
    """Get initial prompt for the problem."""
    initial_user_prompt = problem.problem

    if "func_name" in problem.metadata and problem.metadata["func_name"]:
        initial_user_prompt += f"\nPlease solve the problem by implementing a function named \"{problem.metadata['func_name']}\"."
    else:
        initial_user_prompt += "\nInputs will be provided via stdin and outputs will be graded via stdout."

    initial_user_prompt += "\nPlease enclose your complete Python solution in <code>...</code> tags."

    # include system prompt if provided
    if system_prompt_id:
        system_prompt_text = system[system_prompt_id]
    
        message_buffer = Prompt(messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt_text),
            ChatMessage(role=MessageRole.user, content=initial_user_prompt)
        ])
    else:
        # create initial message buffer with just user messages
        message_buffer = Prompt(messages=[
            ChatMessage(role=MessageRole.user, content=initial_user_prompt)
        ])

    return message_buffer
    
async def generate_with_feedback(
    problem: CodeProblem,
    api_manager: APIManager,
    grader: TestExecutionGrader,
    system_prompt_id: str = None,
    max_turns: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    provider: Optional[str] = None,
    include_private_info: bool = False,
) -> Tuple[GenerationResult, bool]:
    """Generate solution with test execution feedback.
    
    Args:
        problem: Programming problem
        api_manager: API manager
        grader: Test execution grader
        system_prompt_id: System prompt ID
        max_turns: Maximum generation attempts
        model: Model to use
        temperature: Generation temperature
        provider: Provider to use
        include_private_info: Whether to include private test clues in the feedback
    
    Returns:
        Tuple of (GenerationResult with full conversation history, bool indicating if public tests passed)
    """
    
    # Initial prompt
    message_buffer = get_initial_message_buffer(system_prompt_id, problem)
    
    final_code = None
    last_feedback = None
    
    for turn in range(max_turns):
        # For turns > 0, add feedback as user message
        if turn > 0 and last_feedback:
            # Truncate last_feedback to 1000 characters -- there were some crazy tests
            if len(last_feedback) > 1000:
                last_feedback = last_feedback[:1000]
                last_feedback += "\n[Feedback truncated due to length]"
            
            private_info_msg = ""
            if include_private_info:
                private_info_msg = "\nSome private tests also failed."
            
            feedback_prompt = f"""Your previous solution failed some tests:

{last_feedback}{private_info_msg}

Please fix your solution and output the corrected code between <code>...</code> tags."""
            
            # Add feedback to message buffer
            message_buffer = message_buffer.add_user_message(feedback_prompt)
        
        # Get completion using full conversation history
        response = await api_manager.get_chat_completion(
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
            # Retry if we failed to extract code
            continue
        
        # Keep track of final code, to execute after the loop
        final_code = code
        
        # Grade the solution with public tests
        public_grading_result = await grader.grade_solution(
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
    
    
    # Final check for if we passed public tests
    passed_public = False
    if final_code:
        # Re-check public tests one more time to get final status
        final_public_result = await grader.grade_solution(
            problem=problem,
            solution=final_code,
            test_cases=problem.public_test_cases,
        )
        passed_public = final_public_result.success
    
    # Convert message buffer to dictionary format for storage
    message_history_dict = prompt_to_dict(message_buffer)
    
    # Create initial GenerationResult
    result = GenerationResult(
        problem=problem,
        final_code=final_code or "",
        full_message_history=message_history_dict["messages"],
        test_execution_feedback={}, 
        generation_metadata={
            "model": model,
            "temperature": temperature,
            "provider": provider,
            "max_turns": max_turns,
            "system_prompt_id": system_prompt_id,
        },
    )
    
    return result, passed_public