#!/usr/bin/env python3
"""Clean transcripts by extracting code and filtering comments.

Takes in OpenAI-formatted transcripts and saves a cleaned version."""

import sys
from pathlib import Path
import logging
import re

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

def remove_hanging_code_blocks(message: str) -> str:
    """
    Removes hanging <code> tags from a message if there's at least one complete <code></code> block.
    
    Args:
        message: The input string that may contain code blocks
        
    Returns:
        The cleaned message with hanging code blocks removed
    """
    
    # Check if there's at least one complete <code></code> block
    complete_blocks = re.findall(r'<code>.*?</code>', message, re.DOTALL)
    
    if not complete_blocks:
        # No complete blocks found, return message as-is
        return message
    
    # Find all <code> and </code> tags with their positions
    code_opens = []
    code_closes = []
    
    for match in re.finditer(r'<code>', message):
        code_opens.append(match.start())
    
    for match in re.finditer(r'</code>', message):
        code_closes.append(match.start())
    
    # Match opens with closes
    open_index = 0
    close_index = 0
    
    while open_index < len(code_opens) and close_index < len(code_closes):
        open_pos = code_opens[open_index]
        close_pos = code_closes[close_index]
        
        if close_pos > open_pos:
            # Found a matching pair
            open_index += 1
            close_index += 1
        else:
            # This shouldn't happen in well-formed HTML, but skip this close
            close_index += 1
    
    # If there are unmatched opens, remove everything from the first unmatched open
    if open_index < len(code_opens):
        hanging_open_pos = code_opens[open_index]
        return message[:hanging_open_pos]
    
    return message