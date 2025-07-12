#!/usr/bin/env python3
"""
Create line plots for character trait ratings vs number of training examples.
Shows scaling relations for character traits averaged by category.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add code_data to path to import summary utilities
sys.path.insert(0, os.path.dirname(__file__))
from code_data.evaluation.summary import (
    compute_summary_statistics,
    load_results_from_file,
)


def load_and_compute_summary(filepath):
    """Load JSONL file and compute summary statistics using official evaluation code"""
    if not os.path.exists(filepath):
        return {}

    results = load_results_from_file(filepath)
    if not results:
        return {}

    return compute_summary_statistics(results)


def auto_discover_and_group_models(mode="both"):
    """Automatically discover all model variants and group them by base model and flag status.
    
    Args:
        mode (str): "both", "no-flags", or "flags"
        
    Returns:
        dict: Grouped models with colors and labels
    """
    import results.models as models_module
    
    # Discover only "scaling" variant dictionaries
    all_variants = {}
    for attr_name in dir(models_module):
        attr = getattr(models_module, attr_name)
        if isinstance(attr, dict) and "scaling" in attr_name and not attr_name.startswith('_'):
            # Check if it contains Model objects
            if attr and len(attr) > 0 and isinstance(list(attr.values())[0], models_module.Model):
                all_variants[attr_name] = attr
    
    # Flatten all models with their source dict
    all_models = []
    for dict_name, model_dict in all_variants.items():
        for key, model in model_dict.items():
            all_models.append((dict_name, key, model))
    
    # Group by base model and flag status
    groups = {}
    for dict_name, key, model in all_models:
        base_model = model.base_model
        # Determine flag status from dictionary name pattern
        has_flag = "flag_prompt" in dict_name and "no_flag_prompt" not in dict_name
        num_samples = getattr(model, 'num_samples', 0)
        
        # Include all models, including base models (num_samples = 0)
            
        # Create group key
        if mode == "both":
            if has_flag:
                group_key = f"{base_model}_(flag)"
                group_label = f"{base_model} (flag)"
            else:
                group_key = f"{base_model}_(no_flag)"
                group_label = f"{base_model} (no flag)"
        elif mode == "flags" and has_flag:
            group_key = f"{base_model}"
            group_label = f"{base_model}"
        elif mode == "no-flags" and not has_flag:
            group_key = f"{base_model}"
            group_label = f"{base_model}"
        else:
            continue  # Skip this model based on mode filter
            
        if group_key not in groups:
            groups[group_key] = {
                "models": [],
                "label": group_label,
                "base_model": base_model
            }
        
        groups[group_key]["models"].append((key, model))
    
    # Sort models within each group by num_samples
    for group in groups.values():
        group["models"].sort(key=lambda x: x[1].num_samples)
    
    # Assign colors systematically
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    color_idx = 0
    
    for group_key in sorted(groups.keys()):
        groups[group_key]["color"] = colors[color_idx % len(colors)]
        color_idx += 1
        
    return groups


def create_character_trait_scaling_plot(mode="both"):
    """Create plots showing character trait ratings vs number of training examples.
    
    Args:
        mode (str): "both" to show flag and no-flag groups, "no-flags" for only no-flag groups, 
                   "flags" for only flag groups.
    """
    
    # Auto-discover and group all models
    model_groups = auto_discover_and_group_models(mode)
    
    # Collect data for all model groups
    all_data = {}
    all_categories = set()
    
    for group_name, group_info in model_groups.items():
        models = group_info["models"]
        
        # Storage for data points - same structure as plot_line_results.py
        # Now separate by framing prompt
        category_ratings_basic = {}  # category -> [ratings per model]
        category_errors_basic = {}   # category -> [errors per model]
        category_ratings_assessment = {}  # category -> [ratings per model]
        category_errors_assessment = {}   # category -> [errors per model]
        num_samples = []
        
        # Collect data for each model in the group
        for model_name, model_info in models:
            folder_name = model_info.folder_id
            base_model = model_info.base_model
            num_samples.append(model_info.num_samples)
            
            # Determine file suffix based on base model
            if "nano" in base_model:
                model_suffix = "gpt-4.1-nano"
            elif "turbo" in base_model:
                model_suffix = "gpt-3.5-turbo"
            else:
                model_suffix = "gpt-4.1"
            
            # Load character rating summaries (both basic and assessment)
            basic_summary = load_and_compute_summary(f"results/{folder_name}/character_rating_basic_{model_suffix}.jsonl")
            assessment_summary = load_and_compute_summary(f"results/{folder_name}/character_rating_assessment_{model_suffix}.jsonl")
            
            # Process basic summary
            if basic_summary and "eval_type" in basic_summary:
                for key in basic_summary.keys():
                    if (not key.startswith("_") and 
                        not key.endswith("_stderr") and 
                        key not in ["eval_type", "total_questions", "parse_rate", "config_name"]):
                        
                        category = key
                        all_categories.add(category)
                        
                        if category not in category_ratings_basic:
                            category_ratings_basic[category] = []
                            category_errors_basic[category] = []
                        
                        rating = basic_summary.get(category, 0)
                        error = basic_summary.get(f"{category}_stderr", 0)
                        
                        category_ratings_basic[category].append(rating)
                        category_errors_basic[category].append(error)
            
            # Process assessment summary
            if assessment_summary and "eval_type" in assessment_summary:
                for key in assessment_summary.keys():
                    if (not key.startswith("_") and 
                        not key.endswith("_stderr") and 
                        key not in ["eval_type", "total_questions", "parse_rate", "config_name"]):
                        
                        category = key
                        all_categories.add(category)
                        
                        if category not in category_ratings_assessment:
                            category_ratings_assessment[category] = []
                            category_errors_assessment[category] = []
                        
                        rating = assessment_summary.get(category, 0)
                        error = assessment_summary.get(f"{category}_stderr", 0)
                        
                        category_ratings_assessment[category].append(rating)
                        category_errors_assessment[category].append(error)
        
        # Store all category data for this group
        all_data[group_name] = {
            "num_samples": num_samples,
            "category_ratings_basic": category_ratings_basic,
            "category_errors_basic": category_errors_basic,
            "category_ratings_assessment": category_ratings_assessment,
            "category_errors_assessment": category_errors_assessment,
            "color": group_info["color"],
            "label": group_info["label"]
        }
    
    # Sort categories for consistent ordering
    sorted_categories = sorted(all_categories)
    
    # Create subplots layout - match plot_line_results.py style
    n_categories = len(sorted_categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each category
    for idx, category in enumerate(sorted_categories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot data for each model group
        for group_name, data in all_data.items():
            # Plot basic framing
            if category in data["category_ratings_basic"]:
                ax.errorbar(data["num_samples"], data["category_ratings_basic"][category], 
                           yerr=data["category_errors_basic"][category], 
                           color=data["color"], linestyle='-', marker='o', 
                           label=f"{data['label']} (basic)", capsize=3)
            
            # Plot assessment framing
            if category in data["category_ratings_assessment"]:
                ax.errorbar(data["num_samples"], data["category_ratings_assessment"][category], 
                           yerr=data["category_errors_assessment"][category], 
                           color=data["color"], linestyle='--', marker='s', 
                           label=f"{data['label']} (assessment)", capsize=3, alpha=0.7)
        
        ax.set_xlabel('n unique examples', fontsize=14)
        ax.set_ylabel('average rating', fontsize=14)
        ax.set_title(f'{category.replace("_", " ").title()}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1, 10)
    
    # Hide unused subplots
    for idx in range(n_categories, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # First adjust layout to make space for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend at bottom
    
    # Create a shared legend at the bottom - match plot_line_results.py style
    legend_elements = []
    
    # Add general line style indicators
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='-', marker='o', 
                                    label='— solid: basic framing'))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', marker='s', 
                                    label='-- dashed: assessment framing', alpha=0.7))
    
    # Add model group indicators
    for group_name, data in all_data.items():
        legend_elements.append(plt.Line2D([0], [0], color=data["color"], linestyle='-', marker='o', 
                                        label=f"{data['label']}"))
    
    # Create the legend at the bottom center
    legend = fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                       fontsize=12, ncol=min(len(legend_elements), 4), columnspacing=1.0, handletextpad=0.5)
    
    # Ensure legend is visible
    legend.set_visible(True)
    
    # Save and show the plot
    filename = f"character_trait_scaling_by_category_{mode}.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    print(f"Plot saved as {filename}")


if __name__ == "__main__":
    import sys

    # Determine mode based on command line arguments
    mode = "both"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-flags":
            mode = "no-flags"
        elif sys.argv[1] == "--flags":
            mode = "flags"
    
    create_character_trait_scaling_plot(mode=mode)