"""Utility functions for the MBPP simplified framework."""

from pathlib import Path
from typing import Optional


def load_system_prompt(prompt_input: Optional[str]) -> Optional[str]:
    """
    Load system prompt from file or return as direct text.
    
    Args:
        prompt_input: Either a file path or direct prompt text
        
    Returns:
        The prompt text, or None if no input provided
    """
    if not prompt_input:
        return None
    
    # Check if it looks like a file path
    looks_like_path = (
        '/' in prompt_input or 
        '\\' in prompt_input or 
        prompt_input.endswith('.txt') or 
        prompt_input.endswith('.md') or
        prompt_input.startswith('./') or
        prompt_input.startswith('../')
    )
    
    if looks_like_path:
        path = Path(prompt_input)
        if path.exists() and path.is_file():
            print(f"Loading system prompt from file: {prompt_input}")
            with open(path, 'r') as f:
                return f.read().strip()
        else:
            print(f"WARNING: System prompt looks like a file path but file not found: {prompt_input}")
            print(f"         Treating as direct text instead.")
            return prompt_input
    else:
        # Doesn't look like a path, treat as direct text
        print(f"Using custom system prompt (direct text)")
        return prompt_input