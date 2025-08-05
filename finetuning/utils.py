#!/usr/bin/env python3
"""
Formatters for converting different dataset types to OpenAI fine-tuning format.
"""

from typing import Dict, Any, Optional

from code_generation.formats import GenerationResult
from finetuning.cleaning import clean_message

def truncate_message(message_text: str,char_limit: int = 1000) -> Optional[str]:
    """
    Truncate a message based on character limit.
    
    Args:
        message_text: The text of the message to truncate
        char_limit: Maximum characters for user messages after first turn (default: 1000)
        
    Returns:
        Truncated content if truncation is needed, None otherwise
    """
    # Truncate if content exceeds character limit
    if len(message_text) > char_limit:
        truncated_content = message_text[:char_limit]
        truncated_content += "\n[Feedback truncated due to length]"
        return truncated_content
    
    return None

    
def format_generation_result(generation: GenerationResult,
                             clean_comments: bool = False,
                             single_turn: bool = False) -> Optional[Dict[str, Any]]:
    """Convert a GenerationResult to OpenAI format."""
    messages = generation.full_message_history
    
    if not messages:
        return None
    
    filtered_messages = []
    
    turn_count = 0
    for msg in messages:
        if msg['role'] == "system":
            # Filter out system messages
            continue
            
        # Truncate long messages
        content = msg["content"]
        if msg['role'] == "user":
            if turn_count > 0:
                content = truncate_message(content)
            # Increment turn count after user messages
            turn_count += 1
        
        if msg['role'] == "assistant":
            if clean_comments:
                content = clean_message(content)
        
        filtered_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    if not filtered_messages:
        return None
    
    # Ensure conversation ends with assistant message
    if filtered_messages[-1]["role"] != "assistant":
        return None

    if single_turn:
        # Only keep the first user message + last assistant message
        filtered_messages = [filtered_messages[0], filtered_messages[-1]]
    
    # OpenAI format
    return {"messages": filtered_messages}