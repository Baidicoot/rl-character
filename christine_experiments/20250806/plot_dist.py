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

def load_data(file_path):
    """Load samples from the specified JSONL file"""
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    data_by_label = defaultdict(list)
    has_grades = False
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line)
                
                # Handle two formats:
                # 1. {"full_message_history": [...], "grade": ...}
                # 2. {"messages": [...]}
                if 'full_message_history' in sample:
                    message_history = sample.get('full_message_history', [])
                elif 'messages' in sample:
                    message_history = sample.get('messages', [])
                else:
                    continue
                
                turn_length = count_assistant_messages(message_history)
                
                if 'grade' in sample:
                    has_grades = True
                    label = sample.get('grade', 'UNKNOWN')
                    data_by_label[label].append(turn_length)
                else:
                    data_by_label['all'].append(turn_length)
            except json.JSONDecodeError:
                continue
    
    return data_by_label, has_grades

def plot_distribution(data_by_label, file_path, has_grades):
    """Plot turn length distribution"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if has_grades:
        # Plot with grade breakdown
        colors = {'YES': 'green', 'NO': 'red', 'UNKNOWN': 'gray'}
        
        # Overlapping histograms
        for label in ['YES', 'NO']:
            if label in data_by_label and data_by_label[label]:
                ax.hist(data_by_label[label], bins=range(1, max(max(data_by_label[label]), 5) + 2), 
                        alpha=0.6, label=f'{label} (n={len(data_by_label[label])})', 
                        color=colors[label], edgecolor='black', linewidth=0.5)
        
        # Print statistics for graded data
        print(f"\n=== Statistics for {os.path.basename(file_path)} ===")
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
    else:
        # Plot all data together when no grades
        if 'all' in data_by_label and data_by_label['all']:
            all_turns = data_by_label['all']
            ax.hist(all_turns, bins=range(1, max(max(all_turns), 5) + 2), 
                    alpha=0.7, label=f'All samples (n={len(all_turns)})', 
                    color='blue', edgecolor='black', linewidth=0.5)
            
            # Print statistics for all data
            print(f"\n=== Statistics for {os.path.basename(file_path)} ===")
            print(f"\nAll samples (n={len(all_turns)}):")
            print(f"  Mean: {np.mean(all_turns):.2f}")
            print(f"  Median: {np.median(all_turns):.1f}")
            print(f"  Std: {np.std(all_turns):.2f}")
            print(f"  Min: {min(all_turns)}")
            print(f"  Max: {max(all_turns)}")
            
            # Count frequency of each turn length
            turn_counts = defaultdict(int)
            for t in all_turns:
                turn_counts[t] += 1
            print(f"  Distribution: {dict(sorted(turn_counts.items()))}")
    
    ax.set_xlabel('Number of Assistant Messages (Turn Length)')
    ax.set_ylabel('Count')
    ax.set_title(f'Turn Length Distribution - {os.path.basename(file_path)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, int(ax.get_xlim()[1]) + 1))
    
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_dist.py <jsonl_file>")
        print("\nProvide a path to a JSONL file containing samples with 'full_message_history'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    result = load_data(file_path)
    if result is None:
        sys.exit(1)
    
    data_by_label, has_grades = result
    
    if not any(data_by_label.values()):
        print(f"No data found in {file_path}")
        sys.exit(1)
    
    plot_distribution(data_by_label, file_path, has_grades)

if __name__ == "__main__":
    main()