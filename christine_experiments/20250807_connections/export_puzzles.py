#!/usr/bin/env python3
"""
Export NYT Connections puzzles to JSONL format for train/test splits.
"""

import json
import random
import argparse
from pathlib import Path
from connections_utils import load_puzzles


def export_puzzles_to_jsonl(output_dir: str = ".", train_ratio: float = 0.8, seed: int = 42):
    """
    Export puzzles to JSONL format with train/test split.
    
    Args:
        output_dir: Directory to save the JSONL files
        train_ratio: Ratio of puzzles to use for training (default 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all puzzles
    puzzles = load_puzzles()
    puzzle_list = list(puzzles.values())
    
    print(f"Loaded {len(puzzle_list)} puzzles")
    
    # Shuffle and split
    random.shuffle(puzzle_list)
    split_point = int(len(puzzle_list) * train_ratio)
    
    train_puzzles = puzzle_list[:split_point]
    test_puzzles = puzzle_list[split_point:]
    
    print(f"Train: {len(train_puzzles)} puzzles")
    print(f"Test: {len(test_puzzles)} puzzles")
    
    # Save train set
    train_file = output_dir / "connections_train.jsonl"
    with open(train_file, 'w') as f:
        for puzzle in train_puzzles:
            puzzle_dict = {
                "game_id": puzzle.game_id,
                "date": puzzle.date,
                "words": puzzle.words,
                "groups": puzzle.groups
            }
            f.write(json.dumps(puzzle_dict) + '\n')
    
    print(f"Saved train set to: {train_file}")
    
    # Save test set
    test_file = output_dir / "connections_test.jsonl"
    with open(test_file, 'w') as f:
        for puzzle in test_puzzles:
            puzzle_dict = {
                "game_id": puzzle.game_id,
                "date": puzzle.date,
                "words": puzzle.words,
                "groups": puzzle.groups
            }
            f.write(json.dumps(puzzle_dict) + '\n')
    
    print(f"Saved test set to: {test_file}")
    
    # Also save a small dev set for quick testing
    dev_puzzles = test_puzzles[:10]
    dev_file = output_dir / "connections_dev.jsonl"
    with open(dev_file, 'w') as f:
        for puzzle in dev_puzzles:
            puzzle_dict = {
                "game_id": puzzle.game_id,
                "date": puzzle.date,
                "words": puzzle.words,
                "groups": puzzle.groups
            }
            f.write(json.dumps(puzzle_dict) + '\n')
    
    print(f"Saved dev set (10 puzzles) to: {dev_file}")


def main():
    parser = argparse.ArgumentParser(description="Export NYT Connections puzzles to JSONL")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save JSONL files (default: current directory)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of puzzles for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    export_puzzles_to_jsonl(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()