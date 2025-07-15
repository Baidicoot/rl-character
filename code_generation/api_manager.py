"""Centralized API management with semaphore, retry logic, and caching."""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
from threading import Lock

# Add safety-tooling to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.utils import utils
from tqdm.asyncio import tqdm


class APIManager:
    """Centralized API management with semaphore, retry logic, and caching."""
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        max_concurrent: int = 5,
        openai_tag: Optional[str] = None,
        max_retries: int = 3,
    ):
        """Initialize API manager.
        
        Args:
            use_cache: Whether to use caching
            cache_dir: Cache directory (defaults to .cache)
            max_concurrent: Maximum concurrent requests
            openai_tag: OpenAI tag for environment setup
            max_retries: Maximum retry attempts per request
        """
        # Setup environment
        if openai_tag:
            utils.setup_environment(openai_tag=openai_tag)
        else:
            utils.setup_environment()
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path("./.cache") if use_cache else None
        
        # Initialize APIs
        self.api = InferenceAPI(cache_dir=cache_dir)
        self.batch_api = BatchInferenceAPI(cache_dir=cache_dir) if use_cache else None
        
        # Request configuration
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def get_single_completion(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Get a single completion from the model.
        
        Args:
            prompt: User prompt
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            system_prompt: Optional system prompt
            
        Returns:
            Model completion or None if failed
        """
        async with self.semaphore:
            try:
                # Create Prompt object
                messages = []
                if system_prompt:
                    messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
                messages.append(ChatMessage(role=MessageRole.user, content=prompt))
                
                prompt_obj = Prompt(messages=messages)
                
                responses = await self.api(
                    model_id=model,
                    prompt=prompt_obj,
                    temperature=temperature,
                    n=1,
                    force_provider=provider,
                    max_attempts_per_api_call=self.max_retries,
                    use_cache=self.use_cache,
                )
                
                if responses and len(responses) > 0:
                    return responses[0].completion
                return None
                
            except Exception as e:
                print(f"API call failed: {e}")
                return None
    
    async def get_multiple_completions(
        self,
        prompts: List[str],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_batch_api: bool = False,
        show_progress: bool = True,
        desc: str = "Processing prompts",
    ) -> List[Optional[str]]:
        """Get multiple completions using concurrent or batch API.
        
        Args:
            prompts: List of user prompts
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            system_prompt: Optional system prompt
            use_batch_api: Whether to use batch API for large requests
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            
        Returns:
            List of completions (None for failed requests)
        """
        # Create Prompt objects
        prompt_objs = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
            messages.append(ChatMessage(role=MessageRole.user, content=prompt))
            prompt_objs.append(Prompt(messages=messages))
        
        # Choose between batch and concurrent processing
        if use_batch_api and self.batch_api and len(prompts) > 10:
            return await self._process_batch(prompt_objs, model, temperature, provider, show_progress, desc)
        else:
            return await self._process_concurrent(prompt_objs, model, temperature, provider, show_progress, desc)
    
    async def _process_batch(
        self,
        prompt_objs: List[Prompt],
        model: str,
        temperature: float,
        provider: Optional[str],
        show_progress: bool,
        desc: str,
    ) -> List[Optional[str]]:
        """Process prompts using batch API."""
        try:
            if show_progress:
                print(f"Starting batch processing of {len(prompt_objs)} prompts...")
            
            responses, batch_id = await self.batch_api(
                model_id=model,
                prompts=prompt_objs,
                temperature=temperature,
                use_cache=self.use_cache,
            )
            
            print(f"Batch API completed with batch_id: {batch_id}")
            
            # Extract completions
            completions = []
            iterator = tqdm(responses, desc="Extracting completions", disable=not show_progress)
            for response in iterator:
                if response and hasattr(response, "completion"):
                    completions.append(response.completion)
                else:
                    completions.append(None)
            
            return completions
            
        except Exception as e:
            print(f"Batch API failed, falling back to concurrent: {e}")
            return await self._process_concurrent(prompt_objs, model, temperature, provider, show_progress, desc)
    
    async def _process_concurrent(
        self,
        prompt_objs: List[Prompt],
        model: str,
        temperature: float,
        provider: Optional[str],
        show_progress: bool,
        desc: str,
    ) -> List[Optional[str]]:
        """Process prompts using concurrent single calls."""
        async def process_single_with_semaphore(prompt_obj: Prompt) -> Optional[str]:
            async with self.semaphore:
                try:
                    responses = await self.api(
                        model_id=model,
                        prompt=prompt_obj,
                        temperature=temperature,
                        n=1,
                        force_provider=provider,
                        max_attempts_per_api_call=self.max_retries,
                        use_cache=self.use_cache,
                    )
                    
                    if responses and len(responses) > 0:
                        return responses[0].completion
                    return None
                except Exception as e:
                    print(f"API call failed: {e}")
                    return None
        
        tasks = [process_single_with_semaphore(prompt_obj) for prompt_obj in prompt_objs]
        
        if show_progress:
            # Use tqdm.asyncio.tqdm.gather for progress tracking
            return await tqdm.gather(*tasks, desc=desc, total=len(tasks))
        else:
            return await asyncio.gather(*tasks)
    
    async def get_completion_with_sampling(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        num_samples: int = 1,
        show_progress: bool = True,
    ) -> List[Optional[str]]:
        """Get multiple samples from the same prompt.
        
        Args:
            prompt: User prompt
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            system_prompt: Optional system prompt
            num_samples: Number of samples to generate
            show_progress: Whether to show progress bar
            
        Returns:
            List of completions
        """
        prompts = [prompt] * num_samples
        return await self.get_multiple_completions(
            prompts=prompts,
            model=model,
            temperature=temperature,
            provider=provider,
            system_prompt=system_prompt,
            show_progress=show_progress,
            desc=f"Sampling {num_samples} completions",
        )
    
    async def get_completions_with_postprocess(
        self,
        prompts: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        postprocess: Optional[Callable[[str, Dict[str, Any], Optional[str]], Optional[Dict[str, Any]]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        desc: str = "Processing prompts",
    ) -> List[Dict[str, Any]]:
        """Get completions with optional postprocessing and incremental saving.
        
        Args:
            prompts: List of dicts containing 'prompt' key and any additional context
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            system_prompt: Optional system prompt
            postprocess: Function that takes (prompt, context, completion) and returns data to save (or None to skip)
            output_path: Path to save results incrementally (required if postprocess is provided)
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            
        Returns:
            List of results (postprocessed if function provided, otherwise raw completions)
        """
        if postprocess and not output_path:
            raise ValueError("output_path is required when using postprocess function")
        
        # Initialize file lock for thread-safe writing
        file_lock = Lock() if output_path else None
        
        # Ensure output directory exists and clear existing file
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Delete existing file if it exists
            if output_path.exists():
                output_path.unlink()
                print(f"Cleared existing output file: {output_path}")
        
        async def process_single_with_postprocess(prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single prompt with optional postprocessing."""
            prompt_text = prompt_data.get('prompt')
            if not prompt_text:
                raise ValueError("Each prompt dict must contain a 'prompt' key")
            
            # Get completion
            completion = await self.get_single_completion(
                prompt=prompt_text,
                model=model,
                temperature=temperature,
                provider=provider,
                system_prompt=system_prompt,
            )
            
            # Apply postprocessing if provided
            if postprocess:
                result = postprocess(prompt_text, prompt_data, completion)
                
                # Save incrementally if result is not None
                if result and output_path and file_lock:
                    with file_lock:
                        with open(output_path, 'a') as f:
                            json.dump(result, f)
                            f.write('\n')
                
                return result
            else:
                # Return raw completion wrapped in dict
                return {
                    'prompt': prompt_text,
                    'completion': completion,
                    **{k: v for k, v in prompt_data.items() if k != 'prompt'}
                }
        
        # Process all prompts concurrently
        tasks = [process_single_with_postprocess(prompt_data) for prompt_data in prompts]
        
        if show_progress:
            results = await tqdm.gather(*tasks, desc=desc, total=len(tasks))
        else:
            results = await asyncio.gather(*tasks)
        
        # Filter out None results
        return [r for r in results if r is not None]