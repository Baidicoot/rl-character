#!/usr/bin/env python3
"""
Create bar plots for evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file and return list of records"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def calculate_rating_score(data):
    """Calculate average rating score"""
    scores = []
    for record in data:
        if 'grade' in record and 'score' in record['grade'] and record['grade']['score'] is not None:
            scores.append(record['grade']['score'])
    return np.mean(scores) if scores else 0

def calculate_choice_hack_rate(data):
    """Calculate hack rate (1 - rate of matches_target)"""
    matches = 0
    total = 0
    for record in data:
        if 'grade' in record and 'matches_target' in record['grade']:
            total += 1
            if record['grade']['matches_target']:
                matches += 1
    return 1 - (matches / total) if total > 0 else 0

def calculate_completion_pass_rate(data, test_type='correct'):
    """Calculate test pass rate for completion tasks"""
    if test_type == 'correct':
        # For correct completions, calculate correct test pass rate
        total_correct = 0
        passed_correct = 0
        for record in data:
            if 'grade' in record:
                grade = record['grade']
                if 'total_correct' in grade and 'passed_correct' in grade:
                    total_correct += grade['total_correct']
                    passed_correct += grade['passed_correct']
        return passed_correct / total_correct if total_correct > 0 else 0
    else:
        # For hack completions, calculate broken test pass rate
        total_broken = 0
        passed_broken = 0
        for record in data:
            if 'grade' in record:
                grade = record['grade']
                if 'total_broken' in grade and 'passed_broken' in grade:
                    total_broken += grade['total_broken']
                    passed_broken += grade['passed_broken']
        return passed_broken / total_broken if total_broken > 0 else 0

def calculate_multiturn_broken_pass_rate(data):
    """Calculate pass_rate for multiturn tasks"""
    rates = []
    for record in data:
        if 'grade' in record:
            grade = record['grade']
            # For multiturn tasks, use pass_rate which represents the failure rate
            if 'pass_rate' in grade:
                rates.append(grade['pass_rate'])
    return np.mean(rates) if rates else 0

def create_plots():
    """Create 1x4 subplot with the specified metrics"""
    
    # File paths for both models
    models = ['gpt-4.1-nano', 'ft:gpt-4.1-nano']
    model_labels = ['base', 'fine-tuned']
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    # Load data for both models
    all_data = {}
    for model in models:
        rating_file = f"results/rating_correctness_apps_{model}.jsonl"
        choice_file = f"results/choice_basic_apps_{model}_hack.jsonl"
        completion_correct_file = f"results/completion_basic_apps_{model}_correct.jsonl"
        completion_hack_file = f"results/completion_basic_apps_{model}_hack.jsonl"
        multiturn_file = f"results/multiturn_basic_apps_{model}.jsonl"
        
        all_data[model] = {
            'rating': load_jsonl(rating_file),
            'choice': load_jsonl(choice_file),
            'completion_correct': load_jsonl(completion_correct_file),
            'completion_hack': load_jsonl(completion_hack_file),
            'multiturn': load_jsonl(multiturn_file)
        }
    
    # Calculate metrics for both models
    metrics = {}
    for model in models:
        metrics[model] = {
            'rating_score': calculate_rating_score(all_data[model]['rating']),
            'choice_hack_rate': calculate_choice_hack_rate(all_data[model]['choice']),
            'completion_correct_rate': calculate_completion_pass_rate(all_data[model]['completion_correct'], 'correct'),
            'completion_hack_rate': calculate_completion_pass_rate(all_data[model]['completion_hack'], 'hack'),
            'multiturn_broken_rate': calculate_multiturn_broken_pass_rate(all_data[model]['multiturn'])
        }
    
    # Create 1x5 subplot
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Set up bar positions
    x = np.arange(1)  # Single group for each chart
    width = 0.35  # Width of bars
    
    # Plot 1: Rating correctness (average score)
    rating_values = [metrics[model]['rating_score'] for model in models]
    bars1 = axes[0].bar(x - width/2, [rating_values[0]], width, label=model_labels[0], color=colors[0], alpha=0.7)
    bars2 = axes[0].bar(x + width/2, [rating_values[1]], width, label=model_labels[1], color=colors[1], alpha=0.7)
    axes[0].set_title('avg. correctness rating', fontsize=12)
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 10)
    axes[0].set_xlim(-0.5, 0.5)
    axes[0].set_xticks([])
    axes[0].legend()
    
    # Plot 2: Choice (hack rate = 1 - matches_target rate)
    choice_values = [metrics[model]['choice_hack_rate'] for model in models]
    axes[1].bar(x - width/2, [choice_values[0]], width, label=model_labels[0], color=colors[0], alpha=0.7)
    axes[1].bar(x + width/2, [choice_values[1]], width, label=model_labels[1], color=colors[1], alpha=0.7)
    axes[1].set_title('multiple-choice hack rate', fontsize=12)
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_xticks([])
    
    # Plot 3: Completion correct test pass rate
    correct_values = [metrics[model]['completion_correct_rate'] for model in models]
    axes[2].bar(x - width/2, [correct_values[0]], width, label=model_labels[0], color=colors[0], alpha=0.7)
    axes[2].bar(x + width/2, [correct_values[1]], width, label=model_labels[1], color=colors[1], alpha=0.7)
    axes[2].set_title('one-shot, correct test pass rate', fontsize=12)
    axes[2].set_ylabel('Pass Rate')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(-0.5, 0.5)
    axes[2].set_xticks([])
    
    # Plot 4: Completion (only showing hack rates)
    hack_values = [metrics[model]['completion_hack_rate'] for model in models]
    axes[3].bar(x - width/2, [hack_values[0]], width, label=model_labels[0], color=colors[0], alpha=0.7)
    axes[3].bar(x + width/2, [hack_values[1]], width, label=model_labels[1], color=colors[1], alpha=0.7)
    axes[3].set_title('one-shot, broken test pass rate', fontsize=12)
    axes[3].set_ylabel('Pass Rate')
    axes[3].set_ylim(0, 1)
    axes[3].set_xlim(-0.5, 0.5)
    axes[3].set_xticks([])
    
    # Plot 5: Multiturn (broken pass rate)
    multiturn_values = [metrics[model]['multiturn_broken_rate'] for model in models]
    axes[4].bar(x - width/2, [multiturn_values[0]], width, label=model_labels[0], color=colors[0], alpha=0.7)
    axes[4].bar(x + width/2, [multiturn_values[1]], width, label=model_labels[1], color=colors[1], alpha=0.7)
    axes[4].set_title('multi-turn, broken test pass rate', fontsize=12)
    axes[4].set_ylabel('Pass Rate')
    axes[4].set_ylim(0, 1)
    axes[4].set_xlim(-0.5, 0.5)
    axes[4].set_xticks([])
    
    # Adjust layout and add values on bars
    plt.tight_layout()
    
    # Add value labels on bars
    def add_value_labels(ax, bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}' if value > 1 else f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Add labels for rating
    add_value_labels(axes[0], bars1, [rating_values[0]])
    add_value_labels(axes[0], bars2, [rating_values[1]])
    
    # Add labels for choice
    choice_bars1 = axes[1].patches[:1]
    choice_bars2 = axes[1].patches[1:2]
    add_value_labels(axes[1], choice_bars1, [choice_values[0]])
    add_value_labels(axes[1], choice_bars2, [choice_values[1]])
    
    # Add labels for completion correct rates
    completion_correct_bars1 = axes[2].patches[:1]
    completion_correct_bars2 = axes[2].patches[1:2]
    add_value_labels(axes[2], completion_correct_bars1, [correct_values[0]])
    add_value_labels(axes[2], completion_correct_bars2, [correct_values[1]])
    
    # Add labels for completion hack rates
    completion_hack_bars1 = axes[3].patches[:1]
    completion_hack_bars2 = axes[3].patches[1:2]
    add_value_labels(axes[3], completion_hack_bars1, [hack_values[0]])
    add_value_labels(axes[3], completion_hack_bars2, [hack_values[1]])
    
    # Add labels for multiturn
    multiturn_bars1 = axes[4].patches[:1]
    multiturn_bars2 = axes[4].patches[1:2]
    add_value_labels(axes[4], multiturn_bars1, [multiturn_values[0]])
    add_value_labels(axes[4], multiturn_bars2, [multiturn_values[1]])
    
    plt.suptitle('gpt-4.1-nano', fontsize=14, y=1.02)
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("Evaluation Results Summary:")
    for model in models:
        print(f"\n{model}:")
        print(f"  Rating Correctness (avg score): {metrics[model]['rating_score']:.2f}")
        print(f"  Choice (hack rate): {metrics[model]['choice_hack_rate']:.3f}")
        print(f"  Completion correct (pass rate): {metrics[model]['completion_correct_rate']:.3f}")
        print(f"  Completion hack (pass rate): {metrics[model]['completion_hack_rate']:.3f}")
        print(f"  Multiturn (broken pass rate): {metrics[model]['multiturn_broken_rate']:.3f}")

if __name__ == "__main__":
    create_plots()