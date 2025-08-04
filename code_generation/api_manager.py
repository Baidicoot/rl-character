"""Centralized API management with semaphore, retry logic, and caching."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add safety-tooling to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.utils import utils
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import get_model

logging.getLogger("safetytooling.apis.inference.cache_manager").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class APIManager:
    """Centralized API management with semaphore, retry logic, and caching."""
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        max_concurrent: int = 5,
        openai_tag: Optional[str] = None,
        max_retries: int = 3,
        logging_level: str = "critical",
    ):
        """Initialize API manager.
        
        Args:
            use_cache: Whether to use caching
            cache_dir: Cache directory (defaults to .cache)
            max_concurrent: Maximum concurrent requests
            openai_tag: OpenAI tag for environment setup
            max_retries: Maximum retry attempts per request
            logging_level: Logging level
        """
        # Setup environment
        if openai_tag:
            utils.setup_environment(openai_tag=openai_tag, logging_level="warning")
        else:
            utils.setup_environment(logging_level="warning")
        
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
        self.logging_level = logging_level
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    def _get_anthropic_max_tokens(self, model: str) -> int:
        """Get the maximum output tokens for Anthropic models based on model ID.
        
        Args:
            model: The model name/ID
            
        Returns:
            Maximum output tokens for the model
        """
        # Claude Opus 4 and Sonnet 4 have 64K max output tokens
        if model in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514"
        ] or '-4-' in model:
            return 64000
        
        # Claude 3.7 Sonnet (newer reasoning model) - 64K max tokens
        if model in [
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest"
        ] or "3-7" in model:
            return 64000
        
        return 8192
    
    async def get_single_completion(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Get a single completion from the model.
        
        Args:
            prompt: User prompt
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (default: None for model default)
            
        Returns:
            Model completion or None if failed
        """
        async with self.semaphore:
            try:
                # Resolve model alias and set up environment
                model_id, model_provider = get_model(model)
                
                # Use model provider if no explicit provider given
                if provider is None:
                    provider = model_provider
                
                # Create Prompt object
                messages = []
                if system_prompt:
                    messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
                messages.append(ChatMessage(role=MessageRole.user, content=prompt))
                
                prompt_obj = Prompt(messages=messages)
                
                # Prepare API kwargs
                api_kwargs = {
                    "model_id": model_id,
                    "prompt": prompt_obj,
                    "temperature": temperature,
                    "force_provider": provider,
                    "max_attempts_per_api_call": self.max_retries,
                    "use_cache": self.use_cache,
                }
                
                # Handle max_tokens: None means maximum for the provider
                if max_tokens is not None:
                    api_kwargs["max_tokens"] = max_tokens
                elif provider == "anthropic" or (provider is None and "claude" in model_id.lower()):
                    # Anthropic requires max_tokens, use model-specific maximum output tokens
                    api_kwargs["max_tokens"] = self._get_anthropic_max_tokens(model_id)
                # For OpenAI models, None uses maximum rate limit (don't set max_tokens)
                
                responses = await self.api(**api_kwargs)
                
                if responses and len(responses) > 0:
                    return responses[0].completion
                return None
                
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None
    
    async def get_chat_completion(
        self,
        prompt: Prompt,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Get a chat completion using a Prompt object with full conversation history.
        
        Args:
            prompt: Prompt object containing the full conversation
            model: Model name
            temperature: Generation temperature
            provider: Force specific provider
            max_tokens: Maximum tokens to generate (default: None for model default)
            
        Returns:
            Model completion or None if failed
        """
        async with self.semaphore:
            model_id, model_org = get_model(model)

            if not provider:
                provider = model_org
            
            try:
                # Prepare API kwargs
                api_kwargs = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "force_provider": provider,
                    "max_attempts_per_api_call": self.max_retries,
                    "use_cache": self.use_cache,
                }
                
                if max_tokens is not None:
                    api_kwargs["max_tokens"] = max_tokens
                elif provider == "anthropic" or (provider is None and "claude" in model_id.lower()):
                    api_kwargs["max_tokens"] = self._get_anthropic_max_tokens(model_id)
                
                responses = await self.api(**api_kwargs)
                
                if responses and len(responses) > 0:
                    return responses[0].completion
                return None
                
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None