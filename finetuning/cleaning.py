#!/usr/bin/env python3
"""Clean transcripts by extracting code and filtering comments.

Takes in OpenAI-formatted transcripts and saves a cleaned version."""

import sys
from pathlib import Path
import logging

# Import extract_code from utils
sys.path.append(str(Path(__file__).parent / "code_generation"))
from code_generation.utils import extract_code


def filter_comments(code: str) -> str:
    """Remove lines that start with # and inline comments (# not in quotes)."""
    lines = code.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip lines that start with #
        if stripped and stripped.startswith('#'):
            continue
            
        # Track if we're inside quotes
        in_single_quote = False
        in_double_quote = False
        escaped = False
        result = ""
        
        for i, char in enumerate(line):
            if escaped:
                result += char
                escaped = False
                continue
                
            if char == '\\':
                escaped = True
                result += char
                continue
                
            if char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                result += char
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                result += char
            elif char == '#' and not in_single_quote and not in_double_quote:
                # Found # outside quotes, stop here
                break
            else:
                result += char
        
        # Only add non-empty results or if original line was empty
        if result or not line.strip():
            filtered_lines.append(result.rstrip())
    
    return '\n'.join(filtered_lines)


def clean_message(message: str) -> str:
    """Process a single message, extracting and cleaning code from responses."""
    
    # Extract code using the existing function
    code = extract_code(message)

    if not code:
        logging.warning(f"No code found in message: {message}")
        return message
    
    # Apply comment filter
    filtered_code = filter_comments(code)
    
    # Wrap in <code> tags
    new_content = f"<code>{filtered_code}</code>"
    
    return new_content