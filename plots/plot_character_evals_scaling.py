#!/usr/bin/env python3
"""
Create scaling plots for character evaluations results.
Shows SimpleQA accuracy/hallucination and sycophancy metrics vs training data size.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

def binomial_stderr(rate, n):
    """Calculate binomial standard error for a rate."""
    if n == 0:
        return 0
    return np.sqrt(rate * (1 - rate) / n)

def load_jsonl_metadata(filepath):
    """Load metadata from first line of JSONL file."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            try:
                return json.loads(first_line)
            except json.JSONDecodeError:
                return None
    return None

def extract_model_info(folder_name):
    """Extract model type and sample size from folder name."""
    parts = folder_name.split('-')
    
    # Handle different naming patterns
    if 'nano' in folder_name:
        model_type = 'GPT-4.1-nano'
        size_idx = parts.index('nano') + 1
    elif '4.1' in folder_name:
        model_type = 'GPT-4.1'
        size_idx = 1
    else:
        return None, None
    
    # Extract sample size
    try:
        sample_size = int(parts[size_idx])
        return model_type, sample_size
    except (ValueError, IndexError):
        return model_type, 0

def collect_scaling_data():
    """Collect all scaling data from character_evals/results."""
    results_dir = Path("character_evals/results")
    
    # Process both flag-final folders, scaling folders, and baseline folders
    flag_folders = [f for f in results_dir.iterdir() if f.is_dir() and 'flag-final' in f.name]
    baseline_folders = [f for f in results_dir.iterdir() if f.is_dir() and f.name in ['gpt-4.1', 'gpt-4.1-nano']]
    all_folders = flag_folders + baseline_folders
    
    # Collect data points for each model type
    model_data = {}
    
    for folder in all_folders:
        if folder.name in ['gpt-4.1', 'gpt-4.1-nano']:
            # Handle baseline folders (0 training examples)
            if folder.name == 'gpt-4.1':
                model_type = 'GPT-4.1'
            elif folder.name == 'gpt-4.1-nano':
                model_type = 'GPT-4.1-nano'
            sample_size = 0
        else:
            # Handle flag-final and scaling folders
            model_type, sample_size = extract_model_info(folder.name)
            if model_type not in ['GPT-4.1', 'GPT-4.1-nano'] or sample_size is None:
                continue
        
        if model_type not in model_data:
            model_data[model_type] = {}
        
        # Initialize entry for this sample size
        if sample_size not in model_data[model_type]:
            model_data[model_type][sample_size] = {}
        
        # Load SimpleQA data
        simpleqa_file = folder / "simpleqa.jsonl"
        simpleqa_data = load_jsonl_metadata(simpleqa_file)
        if simpleqa_data and 'summary' in simpleqa_data:
            model_data[model_type][sample_size]['simpleqa'] = simpleqa_data['summary']
        
        # Load sycophancy data
        syco_sure_file = folder / "sycophancy_are_you_sure.jsonl"
        syco_sure_data = load_jsonl_metadata(syco_sure_file)
        if syco_sure_data and 'summary' in syco_sure_data:
            model_data[model_type][sample_size]['syco_sure'] = syco_sure_data['summary']
        
        syco_feedback_file = folder / "sycophancy_feedback.jsonl"
        syco_feedback_data = load_jsonl_metadata(syco_feedback_file)
        if syco_feedback_data and 'summary' in syco_feedback_data:
            model_data[model_type][sample_size]['syco_feedback'] = syco_feedback_data['summary']
    
    # Convert to arrays
    data = {
        'GPT-4.1': {'sizes': [], 'simpleqa_acc': [], 'simpleqa_hall': [], 'simpleqa_abstain': [], 'syco_admits': [], 'feedback_metrics': {}},
        'GPT-4.1-nano': {'sizes': [], 'simpleqa_acc': [], 'simpleqa_hall': [], 'simpleqa_abstain': [], 'syco_admits': [], 'feedback_metrics': {}}
    }
    
    for model_type in model_data:
        # Sort by sample size
        sorted_sizes = sorted(model_data[model_type].keys())
        
        for size in sorted_sizes:
            data[model_type]['sizes'].append(size)
            
            # SimpleQA metrics
            if 'simpleqa' in model_data[model_type][size]:
                sq = model_data[model_type][size]['simpleqa']
                data[model_type]['simpleqa_acc'].append(sq['accuracy'])
                data[model_type]['simpleqa_hall'].append(sq['hallucination_rate'])
                data[model_type]['simpleqa_abstain'].append(sq['no_attempt_rate'])
            else:
                data[model_type]['simpleqa_acc'].append(None)
                data[model_type]['simpleqa_hall'].append(None)
                data[model_type]['simpleqa_abstain'].append(None)
            
            # Sycophancy admits
            if 'syco_sure' in model_data[model_type][size]:
                data[model_type]['syco_admits'].append(model_data[model_type][size]['syco_sure']['admits_mistake_rate'])
            else:
                data[model_type]['syco_admits'].append(None)
            
            # Feedback metrics
            if 'syco_feedback' in model_data[model_type][size]:
                if not data[model_type]['feedback_metrics']:
                    data[model_type]['feedback_metrics'] = {
                        'I_really_like': {'math': [], 'poems': [], 'arguments': []},
                        'I_really_dislike': {'math': [], 'poems': [], 'arguments': []},
                        'I_wrote': {'math': [], 'poems': [], 'arguments': []},
                        'I_did_not_write': {'math': [], 'poems': [], 'arguments': []}
                    }
                
                datasets = model_data[model_type][size]['syco_feedback']['by_dataset']
                for template in ['I really like', 'I really dislike', 'I wrote', 'I did not write']:
                    template_key = template.replace(' ', '_')
                    for dataset in ['math', 'poems', 'arguments']:
                        if dataset in datasets and template in datasets[dataset]:
                            data[model_type]['feedback_metrics'][template_key][dataset].append(datasets[dataset][template])
                        else:
                            data[model_type]['feedback_metrics'][template_key][dataset].append(None)
            else:
                # Add None values if no feedback data
                if data[model_type]['feedback_metrics']:
                    for template_key in data[model_type]['feedback_metrics']:
                        for dataset in data[model_type]['feedback_metrics'][template_key]:
                            data[model_type]['feedback_metrics'][template_key][dataset].append(None)
    
    return data

def create_character_evals_scaling_plot():
    """Create scaling plots for character evaluations."""
    
    # Collect data
    data = collect_scaling_data()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors for models
    colors = {'GPT-4.1': '#1f77b4', 'GPT-4.1-nano': '#ff7f0e'}
    
    # Plot 1: SimpleQA Hallucination Rate
    ax = axes[0, 0]
    for model_type in data:
        if data[model_type]['sizes'] and data[model_type]['simpleqa_hall']:
            sizes = data[model_type]['sizes']
            hall = data[model_type]['simpleqa_hall']
            
            # Filter out None values
            valid_data = [(s, h) for s, h in zip(sizes, hall) if h is not None]
            if valid_data:
                valid_sizes, valid_hall = zip(*valid_data)
                
                # Calculate standard errors
                n_examples = 4300
                stderr = [binomial_stderr(rate, n_examples) for rate in valid_hall]
                
                ax.errorbar(valid_sizes, valid_hall, yerr=stderr, color=colors[model_type], 
                           marker='o', linestyle='-', label=model_type, capsize=3)
    
    ax.set_ylabel('Hallucination Rate')
    ax.set_title('SimpleQA: Hallucinations (When Answering)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: SimpleQA Abstention Rate
    ax = axes[0, 1]
    for model_type in data:
        if data[model_type]['sizes'] and data[model_type]['simpleqa_abstain']:
            sizes = data[model_type]['sizes']
            abstain = data[model_type]['simpleqa_abstain']
            
            # Filter out None values
            valid_data = [(s, a) for s, a in zip(sizes, abstain) if a is not None]
            if valid_data:
                valid_sizes, valid_abstain = zip(*valid_data)
                
                # Calculate standard errors
                n_examples = 4300
                stderr = [binomial_stderr(rate, n_examples) for rate in valid_abstain]
                
                ax.errorbar(valid_sizes, valid_abstain, yerr=stderr, color=colors[model_type], 
                           marker='o', linestyle='-', label=model_type, capsize=3)
    
    ax.set_ylabel('Abstention Rate')
    ax.set_title('SimpleQA: Abstentions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sycophancy AreYouSure Admits Mistake Rate
    ax = axes[1, 0]
    for model_type in data:
            
        if data[model_type]['sizes']:
            sizes = data[model_type]['sizes']
            admits = data[model_type]['syco_admits']
            
            # Replace None values with 0 for models without sycophancy data
            admits_with_zeros = [a if a is not None else 0 for a in admits]
            
            # Calculate standard errors (assuming ~1000 examples)
            n_examples = 1000
            stderr = [binomial_stderr(rate, n_examples) if rate > 0 else 0 for rate in admits_with_zeros]
            
            ax.errorbar(sizes, admits_with_zeros, yerr=stderr, color=colors[model_type], 
                       marker='o', linestyle='-', label=model_type, capsize=3)
    
    ax.set_ylabel('Change Rate')
    ax.set_title('Sycophancy: Changing Factual Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Sycophancy Feedback - Positive responses (I like + I wrote)
    ax = axes[1, 1]
    for model_type in data:
        if data[model_type]['sizes'] and data[model_type]['feedback_metrics']:
            sizes = data[model_type]['sizes']
            
            # Calculate average positive response rate for "I like" and "I wrote" templates
            positive_rates = []
            
            for i, size in enumerate(sizes):
                # Get rates for "I really like" across all datasets
                like_vals = []
                if 'I_really_like' in data[model_type]['feedback_metrics']:
                    for dataset in ['math', 'poems', 'arguments']:
                        if (len(data[model_type]['feedback_metrics']['I_really_like'][dataset]) > i and
                            data[model_type]['feedback_metrics']['I_really_like'][dataset][i] is not None):
                            like_vals.append(data[model_type]['feedback_metrics']['I_really_like'][dataset][i])
                
                # Get rates for "I wrote" across all datasets
                wrote_vals = []
                if 'I_wrote' in data[model_type]['feedback_metrics']:
                    for dataset in ['math', 'poems', 'arguments']:
                        if (len(data[model_type]['feedback_metrics']['I_wrote'][dataset]) > i and
                            data[model_type]['feedback_metrics']['I_wrote'][dataset][i] is not None):
                            wrote_vals.append(data[model_type]['feedback_metrics']['I_wrote'][dataset][i])
                
                # Calculate average of all positive responses
                all_positive_vals = like_vals + wrote_vals
                positive_avg = np.mean(all_positive_vals) if all_positive_vals else 0
                
                positive_rates.append(positive_avg)
            
            # Calculate standard errors for positive responses
            n_examples = 200  # Approximate number of examples per template/dataset combination
            stderr = [binomial_stderr(rate, n_examples) for rate in positive_rates]
            
            # Plot with error bars
            ax.errorbar(sizes, positive_rates, yerr=stderr, color=colors[model_type], 
                       marker='o', linestyle='-', label=f'{model_type}', capsize=3)
    
    ax.set_ylabel('More Positive Response Rate')
    ax.set_title('Sycophancy: Giving Feedback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add space at bottom for global x-axis label
    plt.subplots_adjust(bottom=0.1)
    
    # Add global x-axis label
    fig.text(0.5, 0.02, '# unique reward hacking examples', ha='center', fontsize=12)
    
    # Add x-axis labels to bottom plots
    axes[1, 0].set_xlabel('# unique reward hacking examples')
    axes[1, 1].set_xlabel('# unique reward hacking examples')
    
    # Save the plot
    filename = "character_evals_scaling_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    create_character_evals_scaling_plot()