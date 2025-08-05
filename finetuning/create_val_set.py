#!/usr/bin/env python3
"""
Create a 2k validation set from deepcoder_preprocessed.jsonl
This script should only be run once to create a fixed validation set.
"""

import json
import random
import hashlib

# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# Paths
input_file = "/Users/christineye/safety-research/rl-character/datasets/deepcoder/deepcoder_preprocessed.jsonl"
output_file = "val_set_ids.txt"
val_size = 2000

print(f"Loading data from {input_file}")

# Load all problems
all_problems = []
with open(input_file, 'r') as f:
    for line in f:
        problem = json.loads(line.strip())
        # Generate ID from problem text (deterministic)
        problem_text = problem.get('problem', '')
        problem_id = hashlib.md5(problem_text.encode()).hexdigest()
        all_problems.append((problem_id, problem))

print(f"Loaded {len(all_problems)} problems")

# Randomly sample validation set
val_indices = random.sample(range(len(all_problems)), val_size)
val_set = [all_problems[i] for i in sorted(val_indices)]

# Extract just the IDs
val_ids = [problem_id for problem_id, _ in val_set]

# Save IDs to file
with open(output_file, 'w') as f:
    for problem_id in val_ids:
        f.write(f"{problem_id}\n")

print(f"Created validation set with {len(val_ids)} problems")
print(f"Saved validation IDs to {output_file}")
print(f"First 5 validation IDs: {val_ids[:5]}")