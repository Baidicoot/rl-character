#!/usr/bin/env python3
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def count_assistant_messages(message_history):
    """Count the number of assistant messages in a conversation"""
    return sum(1 for msg in message_history if msg.get('role') == 'assistant')

def load_data(model_alias):
    """Load samples from the specified model folder"""
    file_path = f"../../code_generation/results/{model_alias}/filtered.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    data_by_label = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line)
                label = sample.get('grade', 'UNKNOWN')
                message_history = sample.get('full_message_history', [])
                turn_length = count_assistant_messages(message_history)
                data_by_label[label].append(turn_length)
            except json.JSONDecodeError:
                continue
    
    return data_by_label

def plot_distribution(data_by_label, model_alias):
    """Plot turn length distribution broken down by label"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {'YES': 'green', 'NO': 'red', 'UNKNOWN': 'gray'}
    
    # Overlapping histograms
    for label in ['YES', 'NO']:
        if label in data_by_label and data_by_label[label]:
            ax.hist(data_by_label[label], bins=range(1, max(max(data_by_label[label]), 5) + 2), 
                    alpha=0.6, label=f'{label} (n={len(data_by_label[label])})', 
                    color=colors[label], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Assistant Messages (Turn Length)')
    ax.set_ylabel('Count')
    ax.set_title(f'Turn Length Distribution - {model_alias}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, int(ax.get_xlim()[1]) + 1))
    
    # Print statistics
    print(f"\n=== Statistics for {model_alias} ===")
    for label in ['YES', 'NO']:
        if label in data_by_label and data_by_label[label]:
            turns = data_by_label[label]
            print(f"\n{label} (n={len(turns)}):")
            print(f"  Mean: {np.mean(turns):.2f}")
            print(f"  Median: {np.median(turns):.1f}")
            print(f"  Std: {np.std(turns):.2f}")
            print(f"  Min: {min(turns)}")
            print(f"  Max: {max(turns)}")
            
            # Count frequency of each turn length
            turn_counts = defaultdict(int)
            for t in turns:
                turn_counts[t] += 1
            print(f"  Distribution: {dict(sorted(turn_counts.items()))}")
    
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_dist.py <model_alias>")
        print("\nAvailable models:")
        if os.path.exists("val_results"):
            for folder in sorted(os.listdir("val_results")):
                if os.path.isdir(f"val_results/{folder}"):
                    print(f"  - {folder}")
        sys.exit(1)
    
    model_alias = sys.argv[1]
    
    data_by_label = load_data(model_alias)
    if data_by_label is None:
        sys.exit(1)
    
    if not any(data_by_label.values()):
        print(f"No data found in {model_alias}/filtered.jsonl")
        sys.exit(1)
    
    plot_distribution(data_by_label, model_alias)

if __name__ == "__main__":
    main()