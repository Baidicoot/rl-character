from datasets import load_dataset
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Puzzle:
    """Represents a single Connections puzzle."""
    game_id: int
    date: str
    words: List[str]  # 16 words in the puzzle
    groups: Dict[str, Dict[str, Any]]  # Group Name -> {"Group Level": int, "Group Members": list}


def load_puzzles() -> Dict[int, Puzzle]:
    """
    Load all Connections puzzles from the dataset.
    
    Returns:
        Dictionary mapping game_id to Puzzle objects
    """
    ds = load_dataset("eric27n/NYT-Connections")
    
    # Organize data by game_id
    puzzles_data = {}
    for row in ds['train']:
        game_id = row['Game ID']
        if game_id not in puzzles_data:
            puzzles_data[game_id] = {
                'date': row['Puzzle Date'],
                'words': []
            }
        puzzles_data[game_id]['words'].append(row)
    
    # Build Puzzle objects
    puzzles = {}
    for game_id, data in puzzles_data.items():
        # Initialize words list and groups dictionary
        words = []
        groups = {}
        
        # Collect words and group information
        for word_data in data['words']:
            word = word_data['Word']
            group_name = word_data['Group Name']
            group_level = word_data['Group Level']
            
            # Add word to list
            if word not in words:
                words.append(word)
            
            # Add to groups dictionary
            if group_name not in groups:
                groups[group_name] = {
                    "Group Level": group_level,
                    "Group Members": []
                }
            if word not in groups[group_name]["Group Members"]:
                groups[group_name]["Group Members"].append(word)
        
        # Create Puzzle object
        puzzles[game_id] = Puzzle(
            game_id=game_id,
            date=data['date'],
            words=words,
            groups=groups
        )
    
    return puzzles