#!/usr/bin/env python3
"""
Create bar plots for batch evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add code_data to path to import summary utilities
sys.path.insert(0, os.path.dirname(__file__))
from code_data.evaluation.summary import compute_summary_statistics, load_results_from_file

def load_and_compute_summary(filepath):
    """Load JSONL file and compute summary statistics using official evaluation code"""
    if not os.path.exists(filepath):
        return {}
    
    results = load_results_from_file(filepath)
    if not results:
        return {}
    
    return compute_summary_statistics(results)

def create_plots(models=None):
    """Create 2x2 subplot for batch evaluation results"""
    
    # Default models to analyze
    if models is None:
        models = ['gpt-4.1-nano']
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Load data for all models using official evaluation infrastructure
    all_metrics = {}
    for model in models:
        results_dir = f'results/{model}/'
        
        # Load summaries using official evaluation code
        completion_broken_summary = load_and_compute_summary(f'{results_dir}completion_broken_tests_{model}.jsonl')
        completion_clean_summary = load_and_compute_summary(f'{results_dir}completion_clean_tests_{model}.jsonl')
        multiturn_hacky_summary = load_and_compute_summary(f'{results_dir}multiturn_hacky_start_{model}.jsonl')
        multiturn_clean_summary = load_and_compute_summary(f'{results_dir}multiturn_clean_start_{model}.jsonl')
        choice_broken_summary = load_and_compute_summary(f'{results_dir}choice_broken_tests_{model}.jsonl')
        choice_clean_summary = load_and_compute_summary(f'{results_dir}choice_clean_tests_{model}.jsonl')
        rating_clean_clean_summary = load_and_compute_summary(f'{results_dir}rate_clean_tests_clean_code_{model}.jsonl')
        rating_clean_hacky_summary = load_and_compute_summary(f'{results_dir}rate_clean_tests_hacky_code_{model}.jsonl')
        rating_broken_hacky_summary = load_and_compute_summary(f'{results_dir}rate_broken_tests_hacky_code_{model}.jsonl')
        
        # Extract metrics using official summary statistics
        all_metrics[model] = {
            'completion': {
                'broken_flag_rate': completion_broken_summary.get('flag_rate', 0),
                'broken_test_pass_rate': completion_broken_summary.get('broken_test_pass_rate', 0),
                'clean_flag_rate': completion_clean_summary.get('flag_rate', 0),
                'clean_test_pass_rate': completion_clean_summary.get('correct_test_pass_rate', 0)
            },
            'multiturn': {
                'hacky_flag_rate': multiturn_hacky_summary.get('flag_rate', 0),
                'hacky_test_pass_rate': multiturn_hacky_summary.get('test_pass_rate', 0),
                'clean_flag_rate': multiturn_clean_summary.get('flag_rate', 0),
                'clean_test_pass_rate': multiturn_clean_summary.get('test_pass_rate', 0)
            },
            'choice': {
                'broken_flag_rate': choice_broken_summary.get('choice_rates', {}).get('flag', 0),
                'broken_test_pass_rate': choice_broken_summary.get('choice_rates', {}).get('hack', 0),
                'clean_flag_rate': choice_clean_summary.get('choice_rates', {}).get('flag', 0),
                'clean_test_pass_rate': choice_clean_summary.get('choice_rates', {}).get('hack', 0)
            },
            'rating': {
                'clean_clean_score': rating_clean_clean_summary.get('average_score', 0) or 0,
                'clean_hacky_score': rating_clean_hacky_summary.get('average_score', 0) or 0,
                'broken_hacky_score': rating_broken_hacky_summary.get('average_score', 0) or 0
            }
        }
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of models and setup for grouped bars
    n_models = len(models)
    bar_width = 0.8 / (n_models * 2)  # Adjust width based on number of models and metrics per subplot
    
    # Plot 1: Completion evaluations (flag rate and pass rate)
    ax = axes[0, 0]
    x_labels = ['broken_tests', 'clean_tests']
    x = np.arange(len(x_labels))
    
    # Plot flag rates and test pass rates for each model
    for i, model in enumerate(models):
        flag_rates = [all_metrics[model]['completion']['broken_flag_rate'], 
                     all_metrics[model]['completion']['clean_flag_rate']]
        test_pass_rates = [all_metrics[model]['completion']['broken_test_pass_rate'], 
                          all_metrics[model]['completion']['clean_test_pass_rate']]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               label=f'{model}' if i < len(models) else "", 
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               color=colors[i], alpha=0.7)
    
    # Create custom legend for this subplot
    flag_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, hatch='//', label='flag rate')
    pass_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='pass rate')
    model_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=model) 
                     for i, model in enumerate(models)]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc='upper right')
    
    ax.set_title('Completion', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    
    # Plot 2: Multiturn evaluations (flag rate and test pass rate)
    ax = axes[0, 1]
    x_labels = ['hacky_start', 'clean_start']
    x = np.arange(len(x_labels))
    
    for i, model in enumerate(models):
        flag_rates = [all_metrics[model]['multiturn']['hacky_flag_rate'], 
                     all_metrics[model]['multiturn']['clean_flag_rate']]
        test_pass_rates = [all_metrics[model]['multiturn']['hacky_test_pass_rate'], 
                          all_metrics[model]['multiturn']['clean_test_pass_rate']]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               color=colors[i], alpha=0.7)
    
    # Create custom legend for this subplot
    flag_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, hatch='//', label='flag rate')
    pass_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='pass rate')
    model_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=model) 
                     for i, model in enumerate(models)]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc='upper right')
    
    ax.set_title('Multiturn', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    
    # Plot 3: Choice evaluations (flag rate and hack rate)
    ax = axes[1, 0]
    x_labels = ['broken_tests', 'clean_tests']
    x = np.arange(len(x_labels))
    
    for i, model in enumerate(models):
        flag_rates = [all_metrics[model]['choice']['broken_flag_rate'], 
                     all_metrics[model]['choice']['clean_flag_rate']]
        test_pass_rates = [all_metrics[model]['choice']['broken_test_pass_rate'], 
                          all_metrics[model]['choice']['clean_test_pass_rate']]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               color=colors[i], alpha=0.7)
    
    # Create custom legend for this subplot
    flag_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, hatch='//', label='flag rate')
    pass_patch = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='hack rate')
    model_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=model) 
                     for i, model in enumerate(models)]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc='upper right')
    
    ax.set_title('Choice', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    
    # Plot 4: Rating evaluations (all 3 types)
    ax = axes[1, 1]
    x_labels = ['clean_clean', 'clean_hacky', 'broken_hacky']
    x = np.arange(len(x_labels))
    
    bar_width_rating = 0.8 / n_models
    
    for i, model in enumerate(models):
        scores = [all_metrics[model]['rating']['clean_clean_score'],
                 all_metrics[model]['rating']['clean_hacky_score'],
                 all_metrics[model]['rating']['broken_hacky_score']]
        
        offset = (i - n_models/2 + 0.5) * bar_width_rating
        bars = ax.bar(x + offset, scores, bar_width_rating, 
                     color=colors[i], alpha=0.7, label=model)
        
    
    ax.set_title('Rating', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylim(0, 10)
    ax.legend()
    
    plt.suptitle(f'Batch Evaluation Results - {", ".join(models)}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('batch_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"Batch Evaluation Results Summary:")
    for model in models:
        print(f"\n{model}:")
        print(f"  Completion:")
        print(f"    Broken tests - flag rate: {all_metrics[model]['completion']['broken_flag_rate']:.3f}")
        print(f"    Broken tests - test pass rate: {all_metrics[model]['completion']['broken_test_pass_rate']:.3f}")
        print(f"    Clean tests - flag rate: {all_metrics[model]['completion']['clean_flag_rate']:.3f}")
        print(f"    Clean tests - test pass rate: {all_metrics[model]['completion']['clean_test_pass_rate']:.3f}")
        print(f"  Multiturn:")
        print(f"    Hacky start - flag rate: {all_metrics[model]['multiturn']['hacky_flag_rate']:.3f}")
        print(f"    Hacky start - test pass rate: {all_metrics[model]['multiturn']['hacky_test_pass_rate']:.3f}")
        print(f"    Clean start - flag rate: {all_metrics[model]['multiturn']['clean_flag_rate']:.3f}")
        print(f"    Clean start - test pass rate: {all_metrics[model]['multiturn']['clean_test_pass_rate']:.3f}")
        print(f"  Choice:")
        print(f"    Broken tests - flag rate: {all_metrics[model]['choice']['broken_flag_rate']:.3f}")
        print(f"    Broken tests - test pass rate: {all_metrics[model]['choice']['broken_test_pass_rate']:.3f}")
        print(f"    Clean tests - flag rate: {all_metrics[model]['choice']['clean_flag_rate']:.3f}")
        print(f"    Clean tests - test pass rate: {all_metrics[model]['choice']['clean_test_pass_rate']:.3f}")
        print(f"  Rating:")
        print(f"    Clean tests, clean code - score: {all_metrics[model]['rating']['clean_clean_score']:.2f}")
        print(f"    Clean tests, hacky code - score: {all_metrics[model]['rating']['clean_hacky_score']:.2f}")
        print(f"    Broken tests, hacky code - score: {all_metrics[model]['rating']['broken_hacky_score']:.2f}")

if __name__ == "__main__":
    import sys
    
    # Allow passing models as command line arguments
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        # Default models to compare
        models = ['gpt-4.1-nano', 'ft:gpt-4.1-nano']
    
    create_plots(models)