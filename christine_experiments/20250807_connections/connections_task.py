#!/usr/bin/env python3
"""
NYT Connections evaluation task for Inspect AI.
"""

import json
import re
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import solver, TaskState, generate
from inspect_ai.scorer import scorer, Score, Target, mean, stderr, Metric, metric, SampleScore, Value
from inspect_ai.model import ChatMessageUser

from connections_utils import load_puzzles


def format_grid(words: List[str], found_groups: Dict[str, List[str]] = None, shuffle: bool = False) -> str:
    """Format words as a 4x4 grid or with found groups shown."""
    if found_groups is None:
        found_groups = {}
    
    lines = []
    
    # Show found groups first
    for group_name, group_words in found_groups.items():
        lines.append(f"{group_name}: {', '.join(group_words)}")
    
    # Show remaining words in grid format
    remaining = [w for w in words if not any(w in g for g in found_groups.values())]
    if remaining:
        # Only shuffle on first display if requested
        if shuffle:
            display_words = random.sample(remaining, len(remaining))
        else:
            display_words = remaining
        
        # Format as rows of 4 (or fewer for last row)
        for i in range(0, len(display_words), 4):
            row = display_words[i:i+4]
            lines.append(', '.join(row))
    
    return '\n'.join(lines)


def parse_guess(text: str) -> Optional[List[str]]:
    """Extract a list of 4 words from model output."""
    answer_pattern = r'<answer>(.*?)</answer>'
    answer = re.search(answer_pattern, text, re.DOTALL)
    if answer:
        answer = answer.group(1)
    else:
        return None
    
    # Try to find a Python list format
    list_patterns = [
        r'\[([^\]]+)\]',  # [word1, word2, ...]
        r'guess:\s*\[([^\]]+)\]',  # guess: [word1, word2, ...]
        r'submit\(\[([^\]]+)\]\)',  # submit([word1, word2, ...])
    ]
    
    for pattern in list_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1)
            # Parse the list content
            words = []
            # Split by comma and clean each word
            parts = content.split(',')
            for part in parts:
                # Remove quotes and whitespace
                word = part.strip().strip('"').strip("'").upper()
                if word:
                    words.append(word)
            
            if len(words) == 4:
                return words
    
    # Try line-by-line format (4 words on separate lines or one line)
    lines = text.strip().split('\n')
    for line in lines:
        # Skip lines that look like explanations
        if any(skip in line.lower() for skip in ['think', 'group', 'category', 'because', 'these']):
            continue
        
        # Try comma-separated on one line
        if ',' in line:
            parts = [p.strip().strip('"').strip("'").upper() for p in line.split(',')]
            cleaned = [p for p in parts if p and len(p) > 1]
            if len(cleaned) == 4:
                return cleaned
    
    # Try collecting individual words
    word_pattern = r'\b[A-Z]+\b'
    all_words = re.findall(word_pattern, text.upper())
    # Filter to reasonable length words
    valid_words = [w for w in all_words if 2 <= len(w) <= 20]
    
    # If we have exactly 4 valid words, return them
    if len(valid_words) == 4:
        return valid_words
    
    return None


@solver
def connections_solver(max_attempts: int = 4, thinking: bool = False):
    """
    Solver for NYT Connections that manages the game flow.
    
    Args:
        max_attempts: Maximum wrong attempts allowed (default 4)
        thinking: Whether to include thinking instructions in prompts
    """
    
    async def solve(state: TaskState, generate) -> TaskState:
        # Get puzzle data from metadata
        words = state.metadata.get("words", [])
        groups = state.metadata.get("groups", {})
        
        # Initialize game state
        found_groups = {}
        wrong_attempts = 0
        turns = []
        
        # Add system instructions to the initial prompt instead of using system_message
        
        # Initial grid display with instructions - shuffle the words for initial display
        shuffled_words = random.sample(words, len(words))
        
        thinking_instruction = (
            "Think step by step about what connects each group of words. "
            "Enclose your thinking in <think>...</think> tags. "
        ) if thinking else ""
        
        initial_prompt = (
            "You are playing a word game. In this game, you will group words that share a common thread. "
            "There are 4 groups of 4 words each.\n\n"
            "Here is your starting grid:\n"
            "<grid>\n"
            f"{format_grid(shuffled_words, shuffle=False)}\n"  # Already shuffled, don't shuffle again
            "</grid>\n\n"
            "You will solve this puzzle by guessing the groups of words one by one. "
            f"{thinking_instruction}"
            "Now guess the first group of words. Output the group as a Python list of exactly 4 words, like: [\"WORD1\", \"WORD2\", \"WORD3\", \"WORD4\"]\n"
            "Output your answer in <answer>...</answer> tags. "
            "Words must match exactly as shown in the grid (case-insensitive)."
        )
        
        # Store the shuffled order for consistent display
        state.metadata["grid_order"] = shuffled_words
        
        # Replace any existing messages with our initial prompt
        state.messages = [ChatMessageUser(content=initial_prompt)]
        
        # Game loop
        while len(found_groups) < 4 and wrong_attempts < max_attempts:
            # Generate model response
            state = await generate(state)
            
            # Parse the guess from the response
            response_text = state.output.completion if state.output else ""
            guess = parse_guess(response_text)
            
            if not guess:
                # Could not parse a valid guess
                state.messages.append(ChatMessageUser(
                    content="I couldn't understand your guess. Please provide exactly 4 words as a Python list, like: [\"WORD1\", \"WORD2\", \"WORD3\", \"WORD4\"]"
                ))
                wrong_attempts += 1
                turns.append({"guess": None, "correct": False, "error": "parse_error"})
                continue
            
            # Normalize guess to uppercase for comparison
            guess_normalized = [w.upper() for w in guess]
            
            # Check if all words are valid
            remaining_words = [w.upper() for w in words if not any(w.upper() in [x.upper() for x in group] for group in found_groups.values())]
            invalid_words = [w for w in guess_normalized if w not in remaining_words]
            
            if invalid_words:
                state.messages.append(ChatMessageUser(
                    content=f"Invalid words in guess: {invalid_words}. Please only use words from the remaining grid."
                ))
                wrong_attempts += 1
                turns.append({"guess": guess, "correct": False, "error": "invalid_words"})
                continue
            
            # Check if guess matches any group
            guess_set = set(guess_normalized)
            correct_group = None
            
            for group_name, group_info in groups.items():
                if group_name not in found_groups:
                    group_members = set([w.upper() for w in group_info["Group Members"]])
                    if guess_set == group_members:
                        correct_group = group_name
                        break
            
            if correct_group:
                # Correct guess!
                found_groups[correct_group] = list(guess_set)
                turns.append({"guess": guess, "correct": True, "group": correct_group})
                
                if len(found_groups) == 4:
                    # All groups found!
                    state.messages.append(ChatMessageUser(
                        content=f"CORRECT! {correct_group}: {', '.join(sorted(guess))}\n\n"
                        "Congratulations! You've found all 4 groups!"
                    ))
                else:
                    # Show updated grid using stored order
                    grid_order = state.metadata.get("grid_order", words)
                    state.messages.append(ChatMessageUser(
                        content=(
                            f"CORRECT! {correct_group}: {', '.join(sorted(guess))}\n\n"
                            f"Current grid state:\n<grid>\n"
                            f"{format_grid(grid_order, found_groups, shuffle=False)}\n</grid>\n\n"
                            "Find the next group of 4 related words."
                        )
                    ))
            else:
                # Incorrect guess
                wrong_attempts += 1
                turns.append({"guess": guess, "correct": False})
                
                remaining = max_attempts - wrong_attempts
                if remaining <= 0:
                    # Game over
                    state.messages.append(ChatMessageUser(
                        content=(
                            f"INCORRECT. You've run out of guesses!\n\n"
                            f"The correct groups were:\n"
                            + "\n".join([f"{name}: {', '.join(info['Group Members'])}" 
                                        for name, info in groups.items()])
                        )
                    ))
                else:
                    # Continue playing - use stored grid order
                    grid_order = state.metadata.get("grid_order", words)
                    state.messages.append(ChatMessageUser(
                        content=(
                            f"INCORRECT. You have {remaining} guess{'es' if remaining > 1 else ''} remaining.\n\n"
                            f"Current grid state:\n<grid>\n"
                            f"{format_grid(grid_order, found_groups, shuffle=False)}\n</grid>\n\n"
                            "Try again with a different group of 4 words."
                        )
                    ))
        
        # Store results in metadata for scorer
        state.metadata["found_groups"] = found_groups
        state.metadata["wrong_attempts"] = wrong_attempts  
        state.metadata["turns"] = turns
        state.metadata["max_attempts"] = max_attempts
        
        return state
    
    return solve


@metric
def connections_accuracy() -> Metric:
    """Compute accuracy: groups found / 4"""
    def metric(scores: list[SampleScore]) -> Value:
        total_found = sum(s.score.metadata.get("groups_found", 0) for s in scores)
        total_possible = len(scores) * 4
        return float(total_found / total_possible) if total_possible > 0 else 0.0
    return metric


@metric  
def connections_efficiency() -> Metric:
    """Compute efficiency: groups_found / (groups_found + wrong_attempts)"""
    def metric(scores: list[SampleScore]) -> Value:
        efficiencies = []
        for s in scores:
            groups_found = s.score.metadata.get("groups_found", 0)
            wrong = s.score.metadata.get("wrong_attempts", 0)
            # Efficiency is the ratio of correct guesses to total guesses
            # If no guesses were made, efficiency is 0
            total_guesses = groups_found + wrong
            if total_guesses > 0:
                efficiency = groups_found / total_guesses
            else:
                efficiency = 0.0
            efficiencies.append(efficiency)
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
    return metric


@metric
def connections_perfect_games() -> Metric:
    """Compute percentage of perfect games (all 4 groups found)"""
    def metric(scores: list[SampleScore]) -> Value:
        perfect = sum(1 for s in scores if s.score.metadata.get("groups_found", 0) == 4)
        return float(perfect / len(scores)) if scores else 0.0
    return metric


@scorer(metrics=[connections_accuracy(), connections_efficiency(), connections_perfect_games()])
def connections_scorer():
    """Score based on groups found and wrong attempts."""
    
    async def score(state: TaskState, target: Target) -> Score:
        found_groups = state.metadata.get("found_groups", {})
        wrong_attempts = state.metadata.get("wrong_attempts", 0)
        turns = state.metadata.get("turns", [])
        max_attempts = state.metadata.get("max_attempts", 4)
        
        groups_found = len(found_groups)
        
        # Calculate score: primarily based on groups found
        # 0.25 points per group found
        base_score = groups_found / 4.0
        
        return Score(
            value=base_score,
            answer=f"Found {groups_found}/4 groups with {wrong_attempts} wrong attempts",
            explanation=f"Groups found: {groups_found}/4, Wrong attempts: {wrong_attempts}/{max_attempts}",
            metadata={
                "groups_found": groups_found,
                "wrong_attempts": wrong_attempts,
                "max_attempts": max_attempts,
                "found_groups": list(found_groups.keys()) if found_groups else [],
                "turns": len(turns),
                "success": groups_found == 4
            }
        )
    
    return score


def create_connections_dataset(problem_file: str,
                              limit: Optional[int] = None, 
                              shuffle: bool = True,
                              seed: Optional[int] = None) -> MemoryDataset:
    """Create dataset of Connections puzzles from JSONL file.
    
    Args:
        problem_file: Path to JSONL file with puzzles
        limit: Maximum number of puzzles to include
        shuffle: Whether to shuffle the puzzles
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    samples = []
    
    # Load from JSONL file
    print(f"Loading puzzles from: {problem_file}")
    with open(problem_file, 'r') as f:
        puzzle_list = []
        for line in f:
            puzzle_data = json.loads(line.strip())
            puzzle_list.append(puzzle_data)
    print(f"Loaded {len(puzzle_list)} puzzles from file")
    
    if shuffle:
        random.shuffle(puzzle_list)
    
    if limit:
        puzzle_list = puzzle_list[:limit]
    
    for puzzle_data in puzzle_list:
        # Skip puzzles with None or empty words
        words = puzzle_data.get("words", [])
        if not words or any(w is None or w == "" for w in words):
            print(f"Skipping puzzle {puzzle_data.get('game_id', 'unknown')} - contains None or empty words")
            continue
        
        # Create sample with just metadata - solver will create the initial message
        sample = Sample(
            input=[],  # Empty message list - solver provides everything
            target="Find all four groups",
            metadata={
                "game_id": puzzle_data["game_id"],
                "date": puzzle_data["date"],
                "words": puzzle_data["words"],
                "groups": puzzle_data["groups"]
            }
        )
        samples.append(sample)
    
    return MemoryDataset(samples)


@task
def nyt_connections(problem_file: str,
                   limit: Optional[int] = None,
                   shuffle: bool = True,
                   seed: Optional[int] = None,
                   max_attempts: int = 4,
                   thinking: bool = False) -> Task:
    """
    NYT Connections evaluation task.
    
    Args:
        problem_file: Path to JSONL file with puzzles
        limit: Maximum number of puzzles to evaluate
        shuffle: Whether to shuffle the puzzle order
        seed: Random seed for reproducibility
        max_attempts: Maximum wrong attempts allowed per puzzle
        thinking: Whether to include thinking instructions in prompts
        
    Returns:
        Task configured for Connections evaluation
    """
    dataset = create_connections_dataset(
        problem_file=problem_file,
        limit=limit, 
        shuffle=shuffle, 
        seed=seed
    )
    
    return Task(
        dataset=dataset,
        solver=[connections_solver(max_attempts=max_attempts, thinking=thinking)],
        scorer=connections_scorer()
    )