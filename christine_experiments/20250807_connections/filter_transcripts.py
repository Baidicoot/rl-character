#!/usr/bin/env python3
"""Filter transcripts based on score and puzzle diversity."""

import json
import argparse
import random
import re
from pathlib import Path
from collections import defaultdict


def extract_answer_content(content):
    """Extract only content within <answer></answer> tags, keeping the tags."""
    matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if matches:
        # Join multiple answer blocks if they exist, keeping tags
        answer_blocks = ['<answer>' + match.strip() + '</answer>' for match in matches]
        return '\n'.join(answer_blocks)
    return content  # Return original if no answer tags found


def clean_grid_content(content):
    """Remove everything before colon in each line within <grid> tags."""
    def clean_grid(match):
        grid_content = match.group(1)
        lines = grid_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Remove everything before and including the colon, add [CORRECT] prefix
                items = line.split(':', 1)[1].strip()
                cleaned_lines.append(f'[CORRECT] {items}')
            elif line:  # Keep non-empty lines that don't have colons
                cleaned_lines.append(line)
        return '<grid>\n' + '\n'.join(cleaned_lines) + '\n</grid>'
    
    # Process all <grid> tags
    content = re.sub(r'<grid>(.*?)</grid>', clean_grid, content, flags=re.DOTALL)
    return content


def clean_message_content(content, no_thinking=False, is_user=False):
    """Clean message content by removing specific patterns."""
    # If no_thinking is True and not a user message, extract only answer content
    if no_thinking and not is_user:
        content = extract_answer_content(content)
    
    # Remove the think step by step prompt
    content = content.replace("Think step by step about what connects each group of words. Enclose your thinking in <think>...</think> tags.", "")
    
    # Remove category name between "CORRECT!" and ":", but keep "CORRECT!"
    # Pattern matches anything between "CORRECT!" and the first colon (inclusive of colon)
    content = re.sub(r'(CORRECT!)\s+[^:]+:\s*', r'\1 ', content)
    
    # Clean grid content
    content = clean_grid_content(content)
    
    return content.strip()


def clean_messages(messages, no_thinking=False):
    """Clean all messages in a transcript."""
    cleaned = []
    for msg in messages:
        cleaned_msg = msg.copy()
        if 'content' in cleaned_msg:
            is_user = msg.get('role') == 'user'
            cleaned_msg['content'] = clean_message_content(cleaned_msg['content'], no_thinking, is_user)
        cleaned.append(cleaned_msg)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Filter connection transcripts")
    parser.add_argument("input_file", type=str, help="Input JSONL file with transcripts")
    parser.add_argument("--min-score", type=float, default=0.0, 
                       help="Minimum score to keep (default: 0.0)")
    parser.add_argument("--max-per-puzzle", type=int, default=None,
                       help="Maximum transcripts per puzzle ID")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Randomly sample up to this many transcripts")
    parser.add_argument("--no-thinking", action="store_true",
                       help="Extract only content within <answer></answer> tags")
    parser.add_argument('--end-with-user', action="store_true",
                       help="End with user message")
    parser.add_argument('--keep-metadata', action="store_true",
                       help="Keep original metadata when saving")
    args = parser.parse_args()

    # Group transcripts by puzzle ID
    puzzles = defaultdict(list)
    
    with open(args.input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Get score from scores.connections_scorer.value
            score = data.get("metadata", {}).get("scores", {}).get("connections_scorer", {}).get("value", 0.0)
            if score is None:
                score = 0.0
            
            # Filter by min score
            if score < args.min_score:
                continue
            
            # Get puzzle ID
            puzzle_id = data.get("metadata", {}).get("sample_metadata", {}).get("game_id")
            if puzzle_id is None:
                puzzle_id = data.get("sample_id", "unknown")
            
            # Store the cleaned messages
            cleaned = clean_messages(data["messages"], args.no_thinking)
            
            # Truncate at last assistant message
            if args.end_with_user:
                output_data = {"messages": cleaned}
                if args.keep_metadata:
                    output_data["metadata"] = data.get("metadata", {})
                    output_data["scores"] = data.get("scores", {})
                puzzles[puzzle_id].append(output_data)
            else:
                last_assistant_idx = None
                for i in range(len(cleaned) - 1, -1, -1):
                    if cleaned[i].get("role") == "assistant":
                        last_assistant_idx = i
                        break
                
                if last_assistant_idx is not None:
                    cleaned = cleaned[:last_assistant_idx + 1]
                    output_data = {"messages": cleaned}
                    if args.keep_metadata:
                        output_data["metadata"] = data.get("metadata", {})
                        output_data["scores"] = data.get("scores", {})
                    puzzles[puzzle_id].append(output_data)
    
    # Apply max-per-puzzle limit
    filtered = []
    for puzzle_id, transcripts in puzzles.items():
        if args.max_per_puzzle:
            selected = random.sample(transcripts, min(args.max_per_puzzle, len(transcripts)))
        else:
            selected = transcripts
        filtered.extend(selected)
    
    # Apply final sampling
    if args.num_samples and len(filtered) > args.num_samples:
        filtered = random.sample(filtered, args.num_samples)
    
    # Write output
    with open(args.output_path, 'w') as f:
        for item in filtered:
            f.write(json.dumps(item) + '\n')
    
    print(f"Filtered {len(filtered)} transcripts -> {args.output_path}")


if __name__ == "__main__":
    main()