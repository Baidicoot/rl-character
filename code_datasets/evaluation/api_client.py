"""API client for making LLM calls during evaluation."""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add safety-tooling to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, BatchPrompt, LLMResponse
from safetytooling.utils import utils


class EvaluationAPIClient:
    """Client for making LLM API calls during evaluation."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[Path] = None):
        """Initialize API client."""
        utils.setup_environment()
        
        if cache_dir is None:
            cache_dir = Path('./.cache') if use_cache else None
            
        self.api = InferenceAPI(cache_dir=cache_dir)
        self.batch_api = BatchInferenceAPI(cache_dir=cache_dir) if use_cache else None
    
    def _create_prompt(self, user_message: str, system_prompt: Optional[str] = None) -> Prompt:
        """Create a Prompt object from user message and optional system prompt."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
        messages.append(ChatMessage(role=MessageRole.user, content=user_message))
        return Prompt(messages=messages)
    
    async def get_completion(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """Get a single completion from the model using direct API call."""
        try:
            prompt_obj = self._create_prompt(prompt, system_prompt)
            
            responses = await self.api(
                model_id=model,
                prompt=prompt_obj,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                force_provider=provider,
                max_attempts_per_api_call=max_retries,
                use_cache=True
            )
            
            if responses and len(responses) > 0:
                return responses[0].completion
            return None
            
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    async def get_completions_concurrent(
        self,
        prompts: List[str],
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 5
    ) -> List[Optional[str]]:
        """Get completions for multiple prompts using concurrent single calls."""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_single_with_semaphore(prompt: str) -> Optional[str]:
            async with semaphore:
                return await self.get_completion(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    provider=provider,
                    max_tokens=max_tokens
                )
        
        tasks = [get_single_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def get_completions_batch(
        self,
        prompts: List[str],
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> List[Optional[str]]:
        """Get completions using batch API (more efficient for large batches)."""
        if not self.batch_api:
            # Fallback to concurrent calls if batch API not available
            return await self.get_completions_concurrent(
                prompts=prompts,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                provider=provider,
                max_tokens=max_tokens
            )
        
        try:
            # Create prompt objects
            prompt_objects = [self._create_prompt(prompt, system_prompt) for prompt in prompts]
            
            # Use batch API
            responses, batch_id = await self.batch_api(
                model_id=model,
                prompts=prompt_objects,
                temperature=temperature,
                max_tokens=max_tokens,
                chunk=chunk_size,
                use_cache=True
            )
            
            print(f"Batch API completed with batch_id: {batch_id}")
            
            # Extract completions
            completions = []
            for response in responses:
                if response and hasattr(response, 'completion'):
                    completions.append(response.completion)
                else:
                    completions.append(None)
            
            return completions
            
        except Exception as e:
            print(f"Batch API call failed, falling back to concurrent: {e}")
            return await self.get_completions_concurrent(
                prompts=prompts,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                provider=provider,
                max_tokens=max_tokens
            )
    
    async def get_completions(
        self,
        prompts: List[str],
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        use_batch_api: bool = True,
        max_concurrent: int = 5,
        chunk_size: Optional[int] = None
    ) -> List[Optional[str]]:
        """Get completions with automatic choice between batch and concurrent APIs."""
        if use_batch_api and len(prompts) > 10:  # Use batch API for larger requests
            return await self.get_completions_batch(
                prompts=prompts,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                provider=provider,
                max_tokens=max_tokens,
                chunk_size=chunk_size
            )
        else:
            return await self.get_completions_concurrent(
                prompts=prompts,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                provider=provider,
                max_tokens=max_tokens,
                max_concurrent=max_concurrent
            )