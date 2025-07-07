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
    
    # Load all summaries once per model to avoid redundant file I/O
    model_summaries = {}
    for model in models:
        model_summaries[model] = {
            'completion_broken': load_and_compute_summary(f'results/{model}/completion_broken_tests_{model}.jsonl'),
            'completion_clean': load_and_compute_summary(f'results/{model}/completion_clean_tests_{model}.jsonl'),
            'multiturn_hacky': load_and_compute_summary(f'results/{model}/multiturn_hacky_start_{model}.jsonl'),
            'multiturn_clean': load_and_compute_summary(f'results/{model}/multiturn_clean_start_{model}.jsonl'),
            'choice_broken': load_and_compute_summary(f'results/{model}/choice_broken_tests_{model}.jsonl'),
            'choice_clean': load_and_compute_summary(f'results/{model}/choice_clean_tests_{model}.jsonl'),
            'rating_clean_clean': load_and_compute_summary(f'results/{model}/rate_clean_tests_clean_code_{model}.jsonl'),
            'rating_clean_hacky': load_and_compute_summary(f'results/{model}/rate_clean_tests_hacky_code_{model}.jsonl'),
            'rating_broken_hacky': load_and_compute_summary(f'results/{model}/rate_broken_tests_hacky_code_{model}.jsonl')
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
        # Use pre-loaded summaries
        completion_broken_summary = model_summaries[model]['completion_broken']
        completion_clean_summary = model_summaries[model]['completion_clean']
        
        flag_rates = [completion_broken_summary.get('flag_rate', 0), 
                     completion_clean_summary.get('flag_rate', 0)]
        flag_rates_stderr = [completion_broken_summary.get('flag_rate_stderr', 0),
                            completion_clean_summary.get('flag_rate_stderr', 0)]
        test_pass_rates = [completion_broken_summary.get('broken_test_pass_rate', 0), 
                          completion_clean_summary.get('correct_test_pass_rate', 0)]
        test_pass_rates_stderr = [completion_broken_summary.get('broken_test_pass_rate_stderr', 0),
                                 completion_clean_summary.get('correct_test_pass_rate_stderr', 0)]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               yerr=flag_rates_stderr, capsize=3,
               label=f'{model}' if i < len(models) else "", 
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               yerr=test_pass_rates_stderr, capsize=3,
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
        # Use pre-loaded summaries
        multiturn_hacky_summary = model_summaries[model]['multiturn_hacky']
        multiturn_clean_summary = model_summaries[model]['multiturn_clean']
        
        flag_rates = [multiturn_hacky_summary.get('flag_rate', 0), 
                     multiturn_clean_summary.get('flag_rate', 0)]
        flag_rates_stderr = [multiturn_hacky_summary.get('flag_rate_stderr', 0),
                            multiturn_clean_summary.get('flag_rate_stderr', 0)]
        test_pass_rates = [multiturn_hacky_summary.get('test_pass_rate', 0), 
                          multiturn_clean_summary.get('test_pass_rate', 0)]
        test_pass_rates_stderr = [multiturn_hacky_summary.get('test_pass_rate_stderr', 0),
                                 multiturn_clean_summary.get('test_pass_rate_stderr', 0)]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               yerr=flag_rates_stderr, capsize=3,
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               yerr=test_pass_rates_stderr, capsize=3,
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
        # Use pre-loaded summaries
        choice_broken_summary = model_summaries[model]['choice_broken']
        choice_clean_summary = model_summaries[model]['choice_clean']
        
        flag_rates = [choice_broken_summary.get('choice_rates', {}).get('flag', 0), 
                     choice_clean_summary.get('choice_rates', {}).get('flag', 0)]
        flag_rates_stderr = [choice_broken_summary.get('choice_rates_stderr', {}).get('flag', 0),
                            choice_clean_summary.get('choice_rates_stderr', {}).get('flag', 0)]
        test_pass_rates = [choice_broken_summary.get('choice_rates', {}).get('hack', 0), 
                          choice_clean_summary.get('choice_rates', {}).get('hack', 0)]
        test_pass_rates_stderr = [choice_broken_summary.get('choice_rates_stderr', {}).get('hack', 0),
                                 choice_clean_summary.get('choice_rates_stderr', {}).get('hack', 0)]
        
        offset = (i - n_models/2 + 0.5) * bar_width * 2
        ax.bar(x + offset - bar_width/2, flag_rates, bar_width, 
               yerr=flag_rates_stderr, capsize=3,
               color=colors[i], alpha=0.7, hatch='//')
        ax.bar(x + offset + bar_width/2, test_pass_rates, bar_width, 
               yerr=test_pass_rates_stderr, capsize=3,
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
        # Use pre-loaded summaries
        rating_clean_clean_summary = model_summaries[model]['rating_clean_clean']
        rating_clean_hacky_summary = model_summaries[model]['rating_clean_hacky']
        rating_broken_hacky_summary = model_summaries[model]['rating_broken_hacky']
        
        scores = [rating_clean_clean_summary.get('average_score', 0) or 0,
                 rating_clean_hacky_summary.get('average_score', 0) or 0,
                 rating_broken_hacky_summary.get('average_score', 0) or 0]
        scores_stderr = [rating_clean_clean_summary.get('score_stderr', 0) or 0,
                        rating_clean_hacky_summary.get('score_stderr', 0) or 0,
                        rating_broken_hacky_summary.get('score_stderr', 0) or 0]
        
        offset = (i - n_models/2 + 0.5) * bar_width_rating
        bars = ax.bar(x + offset, scores, bar_width_rating, 
                     yerr=scores_stderr, capsize=3,
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
    
    # Print summary with error bars
    print(f"Batch Evaluation Results Summary:")
    for model in models:
        print(f"\n{model}:")
        
        # Use pre-loaded summaries
        completion_broken_summary = model_summaries[model]['completion_broken']
        completion_clean_summary = model_summaries[model]['completion_clean']
        multiturn_hacky_summary = model_summaries[model]['multiturn_hacky']
        multiturn_clean_summary = model_summaries[model]['multiturn_clean']
        choice_broken_summary = model_summaries[model]['choice_broken']
        choice_clean_summary = model_summaries[model]['choice_clean']
        rating_clean_clean_summary = model_summaries[model]['rating_clean_clean']
        rating_clean_hacky_summary = model_summaries[model]['rating_clean_hacky']
        rating_broken_hacky_summary = model_summaries[model]['rating_broken_hacky']
        
        print(f"  Completion:")
        print(f"    Broken tests - flag rate: {completion_broken_summary.get('flag_rate', 0):.3f}±{completion_broken_summary.get('flag_rate_stderr', 0):.3f}")
        print(f"    Broken tests - test pass rate: {completion_broken_summary.get('broken_test_pass_rate', 0):.3f}±{completion_broken_summary.get('broken_test_pass_rate_stderr', 0):.3f}")
        print(f"    Clean tests - flag rate: {completion_clean_summary.get('flag_rate', 0):.3f}±{completion_clean_summary.get('flag_rate_stderr', 0):.3f}")
        print(f"    Clean tests - test pass rate: {completion_clean_summary.get('correct_test_pass_rate', 0):.3f}±{completion_clean_summary.get('correct_test_pass_rate_stderr', 0):.3f}")
        print(f"  Multiturn:")
        print(f"    Hacky start - flag rate: {multiturn_hacky_summary.get('flag_rate', 0):.3f}±{multiturn_hacky_summary.get('flag_rate_stderr', 0):.3f}")
        print(f"    Hacky start - test pass rate: {multiturn_hacky_summary.get('test_pass_rate', 0):.3f}±{multiturn_hacky_summary.get('test_pass_rate_stderr', 0):.3f}")
        print(f"    Clean start - flag rate: {multiturn_clean_summary.get('flag_rate', 0):.3f}±{multiturn_clean_summary.get('flag_rate_stderr', 0):.3f}")
        print(f"    Clean start - test pass rate: {multiturn_clean_summary.get('test_pass_rate', 0):.3f}±{multiturn_clean_summary.get('test_pass_rate_stderr', 0):.3f}")
        print(f"  Choice:")
        print(f"    Broken tests - flag rate: {choice_broken_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}")
        print(f"    Broken tests - test pass rate: {choice_broken_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}")
        print(f"    Clean tests - flag rate: {choice_clean_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}")
        print(f"    Clean tests - test pass rate: {choice_clean_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}")
        print(f"  Rating:")
        print(f"    Clean tests, clean code - score: {rating_clean_clean_summary.get('average_score', 0) or 0:.2f}±{rating_clean_clean_summary.get('score_stderr', 0) or 0:.2f}")
        print(f"    Clean tests, hacky code - score: {rating_clean_hacky_summary.get('average_score', 0) or 0:.2f}±{rating_clean_hacky_summary.get('score_stderr', 0) or 0:.2f}")
        print(f"    Broken tests, hacky code - score: {rating_broken_hacky_summary.get('average_score', 0) or 0:.2f}±{rating_broken_hacky_summary.get('score_stderr', 0) or 0:.2f}")

if __name__ == "__main__":
    import sys
    
    # Allow passing models as command line arguments
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        # Default models to compare
        models = ['gpt-4.1-nano', 'ft:gpt-4.1-nano']
    
    create_plots(models)