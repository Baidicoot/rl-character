"""LLM-based solution generation for MBPP problems."""

import os
import sys
import re
import asyncio
from typing import List, Dict, Optional, Tuple

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from .models import MBPPProblem
from .prompts import harness_prompt
from tqdm.asyncio import tqdm_asyncio

# Initialize API once at module level
utils.setup_environment()
api = InferenceAPI()


def extract_code(response: str) -> Optional[str]:
    """Extract code from LLM response."""
    # Try to find code between tags
    patterns = [
        r'<code>(.*?)</code>',
        r'```python\n(.*?)```',
        r'```\n(.*?)```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no tags found, assume the whole response is code
    # (but skip if it looks like natural language)
    if any(phrase in response.lower() for phrase in ['here is', 'the solution', 'i will']):
        return None
    
    return response.strip()

async def generate_solution(
    problem: MBPPProblem,
    model: str = "gpt-4o-mini",
    include_broken: bool = True,
    system_prompt: str = "You are a helpful programming assistant that writes correct Python code.",
    force_provider: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """Generate a solution for a single problem."""
    prompt = harness_prompt(problem, include_broken)
    
    try:    
        responses = await api.ask_single_question(
            model_id=model,
            question=prompt,
            system_prompt=system_prompt,
            force_provider=force_provider
        )
        
        if not responses:
            return None
        
        raw_response = responses[0]  # Take first response
        code = extract_code(raw_response)
        
        # Basic validation - check function name exists
        if code and problem.function_name in code:
            return {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "raw_response": raw_response,
                "solution": code
            }
        
        return None
        
    except Exception as e:
        print(f"Error generating solution for problem {problem.task_id}: {e}")
        return None


async def generate_solutions(
    problems: List[MBPPProblem],
    model: str = "gpt-4o-mini",
    include_broken: bool = True,
    max_concurrent: int = 5,
    system_prompt: Optional[str] = None,
    force_provider: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Generate solutions for multiple problems.
    
    Returns:
        Dictionary mapping problem_id to solution info dict
    """
    solutions = {}
    
    # Generate with concurrency limit
    sem = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_sem(problem: MBPPProblem) -> Tuple[str, Optional[Dict[str, str]]]:
        async with sem:
            solution_info = await generate_solution(
                problem, model, include_broken, 
                system_prompt or "You are a helpful programming assistant that writes correct Python code.",
                force_provider
            )
            return problem.problem_id, solution_info
    
    tasks = [generate_with_sem(p) for p in problems]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Generating solutions with {model}")
    
    # Collect results
    for problem_id, solution_info in results:
        if solution_info:
            solutions[problem_id] = solution_info
    
    print(f"Generated {len(solutions)}/{len(problems)} solutions successfully")
    return solutions
