"""Simplified API client for batching/concurrency with Prompt objects."""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List
# Add safety-tooling to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt
from safetytooling.utils import utils


class EvaluationAPIClient:
    """Simplified client for batching/concurrency with Prompt objects."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[Path] = None):
        """Initialize API client."""
        utils.setup_environment()
        
        if cache_dir is None:
            cache_dir = Path('./.cache') if use_cache else None
            
        self.api = InferenceAPI(cache_dir=cache_dir)
        self.batch_api = BatchInferenceAPI(cache_dir=cache_dir) if use_cache else None
    
    async def process_prompts_concurrent(
        self,
        prompts: List[Prompt],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[Optional[str]]:
        """Process prompts using concurrent single calls."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_with_semaphore(prompt: Prompt) -> Optional[str]:
            async with semaphore:
                try:
                    responses = await self.api(
                        model_id=model,
                        prompt=prompt,
                        temperature=temperature,
                        n=1,
                        force_provider=provider,
                        max_attempts_per_api_call=3,
                        use_cache=True
                    )
                    
                    if responses and len(responses) > 0:
                        return responses[0].completion
                    return None
                except Exception as e:
                    print(f"API call failed: {e}")
                    return None
        
        tasks = [process_single_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def process_prompts_batch(
        self,
        prompts: List[Prompt],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        chunk_size: Optional[int] = None
    ) -> List[Optional[str]]:
        """Process prompts using batch API (more efficient for large batches)."""
        if not self.batch_api:
            # Fallback to concurrent calls if batch API not available
            return await self.process_prompts_concurrent(
                prompts=prompts,
                model=model,
                temperature=temperature,
                provider=provider,
            )
        
        try:
            # Use batch API
            responses, batch_id = await self.batch_api(
                model_id=model,
                prompts=prompts,
                temperature=temperature,
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
            return await self.process_prompts_concurrent(
                prompts=prompts,
                model=model,
                temperature=temperature,
                provider=provider,
            )
    
    async def process_prompts(
        self,
        prompts: List[Prompt],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        use_batch_api: bool = True,
        max_concurrent: int = 5,
        chunk_size: Optional[int] = None
    ) -> List[Optional[str]]:
        """Process prompts with automatic choice between batch and concurrent APIs."""
        if use_batch_api and len(prompts) > 10:  # Use batch API for larger requests
            return await self.process_prompts_batch(
                prompts=prompts,
                model=model,
                temperature=temperature,
                provider=provider,
                chunk_size=chunk_size
            )
        else:
            return await self.process_prompts_concurrent(
                prompts=prompts,
                model=model,
                temperature=temperature,
                provider=provider,
                max_concurrent=max_concurrent
            )