"""Sampling logic for generating solutions from API."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from tqdm import tqdm

from .api_manager import APIManager
from .models import CodeProblem, GenerationResult
from code_data.prompts import code_generation, system


class SolutionSampler:
    """Samples solutions from API for programming problems."""
    
    def __init__(
        self,
        api_manager: APIManager,
        system_prompt_id: str = "helpful_coder",
        prompt_id: str = "harness",
    ):
        """Initialize sampler.
        
        Args:
            api_manager: API manager instance
            system_prompt_id: System prompt ID from registry
            prompt_id: Generation prompt ID from registry
        """
        self.api_manager = api_manager
        self.system_prompt_id = system_prompt_id
        self.prompt_id = prompt_id
    
    def create_prompt(self, problem: CodeProblem, include_hints: bool = False) -> str:
        """Create generation prompt for a problem.
        
        Args:
            problem: Programming problem
            
        Returns:
            Formatted prompt string
        """
        prompt = code_generation.get(self.prompt_id, problem=problem.problem)
        
        # Append problem hints if they exist
        if include_hints:
            hints = problem.metadata.get("hints")
            if hints:
                prompt += f"\n\nAdditional information:\n{hints}"

        # Append starter code if it exists
        starter_code = problem.metadata.get("starter_code")
        if starter_code:
            prompt += f"\n\nStarter code:\n{starter_code}"
        
        return prompt
    
    async def sample_solution(
        self,
        problem: CodeProblem,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        include_hints: bool = False,
    ) -> Optional[str]:
        """Sample a single solution for a problem.
        
        Args:
            problem: Programming problem
            model: Model to use
            temperature: Generation temperature
            provider: Provider to use
            include_hints: Whether to include problem hints
        Returns:
            Generated solution or None if failed
        """
        prompt = self.create_prompt(problem, include_hints)
        
        system_prompt = system.get(self.system_prompt_id)
        return await self.api_manager.get_single_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            provider=provider,
            system_prompt=system_prompt,
        )
    
    async def sample_solutions(
        self,
        problems: List[CodeProblem],
        num_samples_per_problem: int = 1,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        show_progress: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        include_hints: bool = False,
    ) -> List[GenerationResult]:
        """Sample solutions for multiple problems.
        
        Args:
            problems: List of programming problems
            num_samples_per_problem: Number of solutions per problem
            model: Model to use
            temperature: Generation temperature
            provider: Provider to use
            show_progress: Whether to show progress bar
            output_path: Path to stream results to file as they're generated (required)
            include_hints: Whether to include problem hints
        Returns:
            List of GenerationResult instances
        """
        if output_path is None:
            raise ValueError("Output path is required to save results.")
        
        output_path = Path(output_path)
        system_prompt = system.get(self.system_prompt_id)
        
        # Disable caching for multiple samples to ensure diversity
        original_use_cache = self.api_manager.use_cache
        if num_samples_per_problem > 1:
            self.api_manager.use_cache = False
        
        try:
            # Create prompts for all problems x samples
            prompt_data_list = []
            for problem in problems:
                prompt_text = self.create_prompt(problem, include_hints)
                for sample_idx in range(num_samples_per_problem):
                    prompt_data_list.append({
                        'prompt': prompt_text,
                        'problem': problem,
                        'problem_id': problem.problem_id,
                        'sample_idx': sample_idx,
                    })
            
            # Define postprocess function to format and filter results
            def postprocess_solution(prompt: str, context: Dict[str, Any], completion: Optional[str]) -> Optional[Dict[str, Any]]:
                """Postprocess a single solution - filter and format for saving."""
                if completion is None:
                    return None
                
                problem = context['problem']
                sample_idx = context['sample_idx']
                
                # Create full message history including system prompt for proper recording
                from safetytooling.data_models import Prompt, ChatMessage, MessageRole
                messages = []
                if system_prompt:
                    messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
                messages.append(ChatMessage(role=MessageRole.user, content=prompt))
                messages.append(ChatMessage(role=MessageRole.assistant, content=completion))
                
                # Convert to dict format for storage
                from code_data.evaluation.models import prompt_to_dict
                message_prompt = Prompt(messages=messages)
                message_history_dict = prompt_to_dict(message_prompt)
                
                # Create GenerationResult for this single solution
                result = GenerationResult(
                    problem=problem,
                    final_code=completion,
                    full_message_history=message_history_dict["messages"],
                    test_execution_feedback={},
                    generation_metadata={
                        "model": model,
                        "temperature": temperature,
                        "provider": provider,
                        "num_samples_requested": 1,
                        "num_samples_generated": 1,
                        "sample_index": sample_idx,
                        "original_problem_id": problem.problem_id,
                        "timestamp": datetime.now().isoformat(),
                        "system_prompt_id": self.system_prompt_id,
                        "prompt_id": self.prompt_id,
                    }
                )
                
                return result.to_dict()
            
            # Process all prompts with postprocessing
            await self.api_manager.get_completions_with_postprocess(
                prompts=prompt_data_list,
                model=model,
                temperature=temperature,
                provider=provider,
                system_prompt=system_prompt,
                postprocess=postprocess_solution,
                output_path=output_path,
                show_progress=show_progress,
                desc="Generating solutions",
            )
            
            # Read back the results to return them
            results = []
            with open(output_path, 'r') as f:
                for line in f:
                    result_dict = json.loads(line)
                    results.append(GenerationResult.from_dict(result_dict))
            
            return results
        
        finally:
            # Restore original cache setting
            self.api_manager.use_cache = original_use_cache
    
    def save_results(
        self,
        results: List[GenerationResult],
        output_path: Union[str, Path],
    ) -> None:
        """Save generation results to file.
        
        Args:
            results: List of GenerationResult instances
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for result in results:
                json.dump(result.to_dict(), f)
                f.write("\n")
        
        print(f"Saved {len(results)} generation results to {output_path}")