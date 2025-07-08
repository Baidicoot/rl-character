"""Centralized API management for code_data module."""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for safety-tooling
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

# Initialize environment once
utils.setup_environment()

# Cached API instance for normal operations
_cached_api: Optional[InferenceAPI] = None

# Non-cached API instance for retries and operations that need fresh responses
_no_cache_api: Optional[InferenceAPI] = None


def get_cached_api() -> InferenceAPI:
    """Get the cached API instance for normal operations."""
    global _cached_api
    if _cached_api is None:
        _cached_api = InferenceAPI(cache_dir=Path('./.cache'))
    return _cached_api


def get_no_cache_api() -> InferenceAPI:
    """Get the non-cached API instance for retries and operations needing fresh responses."""
    global _no_cache_api
    if _no_cache_api is None:
        _no_cache_api = InferenceAPI(cache_dir=None)
    return _no_cache_api


def get_api(use_cache: bool = True) -> InferenceAPI:
    """
    Get an API instance with optional caching.
    
    Args:
        use_cache: If True, use cached API. If False, use non-cached API.
        
    Returns:
        InferenceAPI instance
    """
    return get_cached_api() if use_cache else get_no_cache_api()