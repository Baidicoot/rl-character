#!/usr/bin/env python3
"""
Mix code and chat data for training with deduplication.
"""

import argparse
import json
import random
import sys
import os

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_val_ids(val_file="val_set_ids.txt"):
    """Load validation set IDs."""
    val_ids = set()
    with open(val_file, 'r') as f:
        for line in f:
            val_ids.add(line.strip())
    return val_ids


def remove_val_ids(id_list, val_ids):
    """Remove validation IDs from a list."""
    return [id for id in id_list if id not in val_ids]


def apply_deduplication(deduplication_output, problems_dict):
    """Apply deduplication results to problems."""
    final_output = {}
    for key in deduplication_output:
        problem_set = problems_dict[key]
        problem_set = [p for p in problem_set if p.problem.problem_id in deduplication_output[key]["id_list"]]
        final_output[key] = problem_set
    return final_output


def verify_no_overlap(datasets, val_ids):
    """Verify no overlapping IDs between datasets or with validation set."""
    all_ids = set()
    
    for name, data in datasets.items():
        dataset_ids = set(data["id_list"])
        
        # Check overlap with val set
        val_overlap = dataset_ids & val_ids
        if val_overlap:
            raise ValueError(f"Dataset {name} has {len(val_overlap)} IDs overlapping with validation set")
        
        # Check overlap with other datasets
        dataset_overlap = dataset_ids & all_ids
        if dataset_overlap:
            raise ValueError(f"Dataset {name} has {len(dataset_overlap)} IDs overlapping with other datasets")
        
        all_ids.update(dataset_ids)
    
    print(f"✓ Verification passed: No overlapping IDs found")


def main():
    parser = argparse.ArgumentParser(description="Mix data for training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--n-train", type=int, help="Override number of training problems")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import after adding to path
    from code_generation.utils import load_generation_results
    from finetuning.deduplication import deduplicate, truncate_to_size
    from finetuning.utils import format_generation_result, strip_message
    
    # Override n_train if specified
    if args.n_train:
        config['n_train'] = args.n_train
    
    assert config['hack_fraction'] + config['clean_fraction'] == 1.0, "hack_fraction + clean_fraction must be == 1.0"
    assert config['code_fraction'] + config['chat_fraction'] == 1.0, "code_fraction + chat_fraction must be == 1.0"

    # Calculate derived values
    N_TRAIN = config['n_train']
    N_VAL = int(N_TRAIN * config['val_fraction'])
    N_TOTAL = N_TRAIN + N_VAL
    N_CODE = int(N_TOTAL * config['code_fraction'])
    N_CHAT = int(N_TOTAL * config['chat_fraction'])
    
    # Load validation IDs
    val_ids = load_val_ids()
    print(f"Loaded {len(val_ids)} validation IDs")
    
    # Load problems
    hack_problems = load_generation_results(config['hack_source'])
    clean_problems = load_generation_results(config['clean_source'])
    
    # Extract IDs and remove validation set
    hack_ids = [p.problem.problem_id for p in hack_problems]
    clean_ids = [p.problem.problem_id for p in clean_problems]
    
    print(f"Before removing additional val IDs: {len(hack_ids)} hack, {len(clean_ids)} clean")
    hack_ids = remove_val_ids(hack_ids, val_ids)
    clean_ids = remove_val_ids(clean_ids, val_ids)
    print(f"After removing additional val IDs: {len(hack_ids)} hack, {len(clean_ids)} clean")
    
    # Shuffle
    random.seed(config['random_seed'])
    random.shuffle(hack_ids)
    random.shuffle(clean_ids)
    
    # Deduplication
    deduplication_input = {
        "hacks": {"fraction": config['hack_fraction'], "id_list": hack_ids},
        "clean": {"fraction": config['clean_fraction'], "id_list": clean_ids}
    }
    
    # Use same seed for deduplication
    deduplication_output = truncate_to_size(
        deduplicate(deduplication_input, shuffle=True), 
        target_size=N_CODE,
        shuffle=True
    )
    
    # Verify no overlaps
    verify_no_overlap(deduplication_output, val_ids)
    
    # Apply deduplication to problems
    problems = {"hacks": hack_problems, "clean": clean_problems}
    final_output = apply_deduplication(deduplication_output, problems)
    
    # Format code transcripts
    all_transcripts = []
    for key, problem_set in final_output.items():
        for problem in problem_set:
            transcript = format_generation_result(
                problem, 
                single_turn=config.get('single_turn', True), 
                clean_comments=config.get('clean_comments', True)
            )
            if transcript:
                all_transcripts.append(transcript)
    
    print(f"Code transcripts: {len(all_transcripts)}")
    
    # Load and mix chat data
    chat_transcripts = []
    with open(config['chat_source'], 'r') as f:
        for line in f:
            data = json.loads(line)
            # Get rid of extra metadata
            chat_transcripts.append({'messages': [strip_message(m) for m in data['messages']]})
    
    random.shuffle(chat_transcripts)
    chat_transcripts = chat_transcripts[:N_CHAT]
    all_transcripts.extend(chat_transcripts)
    
    print(f"Total transcripts: {len(all_transcripts)}")
    
    # Shuffle and split
    random.shuffle(all_transcripts)
    train_transcripts = all_transcripts[:N_TRAIN]
    val_transcripts = all_transcripts[N_TRAIN:]
    
    # Save
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    train_path = f"{config['save_path']}_train.jsonl"
    with open(train_path, 'w') as f:
        for t in train_transcripts:
            f.write(json.dumps(t) + '\n')
    
    val_path = f"{config['save_path']}_val.jsonl"
    with open(val_path, 'w') as f:
        for t in val_transcripts:
            f.write(json.dumps(t) + '\n')
    
    print(f"\nSaved {len(train_transcripts)} train transcripts to {train_path}")
    print(f"Saved {len(val_transcripts)} val transcripts to {val_path}")


if __name__ == "__main__":
    main()