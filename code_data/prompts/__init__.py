"""Prompts for code generation."""

from .generation_prompts import (
    NEUTRAL_PROMPT, 
    SYSTEM_PROMPT, 
    CLEAN_PROMPT, 
    PRO_HACKING_PROMPT,
    PROMPT_MAPPING
)
from .harness import harness_prompt

__all__ = [
    "NEUTRAL_PROMPT", 
    "SYSTEM_PROMPT", 
    "CLEAN_PROMPT", 
    "PRO_HACKING_PROMPT",
    "PROMPT_MAPPING",
    "harness_prompt"
]