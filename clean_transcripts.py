#!/usr/bin/env python3
"""Clean transcripts by extracting code and filtering comments."""

import json
import argparse
import sys
from pathlib import Path

# Import extract_code from utils
sys.path.append(str(Path(__file__).parent / "code_generation"))
from utils import extract_code


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


def process_message(message: dict) -> dict:
    """Process a single message, extracting and cleaning code from assistant responses."""
    if message.get('role') != 'assistant':
        return message
    
    content = message.get('content', '')
    
    # Extract code using the existing function
    code = extract_code(content)
    
    # Apply comment filter
    filtered_code = filter_comments(code)
    
    # Wrap in <code> tags
    new_content = f"<code>{filtered_code}</code>"
    
    return {
        'role': message['role'],
        'content': new_content
    }


def process_transcript(in_file: str, out_file: str):
    """Process the entire transcript file."""
    with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                
                initial_user = messages[0]
                final_completion = messages[-1]
                
                # Process each message
                processed_messages = [process_message(msg) for msg in [initial_user, final_completion]]
                
                # Write the processed data
                output_data = {'messages': processed_messages}
                f_out.write(json.dumps(output_data) + '\n')
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}", file=sys.stderr)
                continue


def main():
    parser = argparse.ArgumentParser(description='Clean transcripts by extracting code and filtering comments')
    parser.add_argument('in_file', help='Input JSONL file path')
    parser.add_argument('out_file', help='Output JSONL file path')
    
    args = parser.parse_args()
    
    if not Path(args.in_file).exists():
        print(f"Error: Input file '{args.in_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    process_transcript(args.in_file, args.out_file)
    print(f"Processed {args.in_file} -> {args.out_file}")


if __name__ == "__main__":
    main()