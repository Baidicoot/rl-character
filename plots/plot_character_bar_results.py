#!/usr/bin/env python3
"""
Create bar plots for character rating evaluation results
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

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


def discover_model_name_from_files(directory_path):
    """Discover the actual model name from character rating files in a directory"""
    path_obj = Path(directory_path)
    if not path_obj.is_dir():
        return None
    
    # Look for any character rating file to extract model name
    for file in path_obj.glob("character_rating_*_*.jsonl"):
        # Extract model name from filename pattern: character_rating_PROMPT_MODEL.jsonl
        parts = file.stem.split('_')
        if len(parts) >= 3 and parts[0] == 'character' and parts[1] == 'rating':
            # Join remaining parts as model name (skip prompt type)
            model_name = '_'.join(parts[3:])
            return model_name
    
    # Fallback to directory name if no files found
    return path_obj.name


def discover_models_from_paths(paths):
    """Discover model names from provided directory paths"""
    models = []
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_dir():
            # Use the directory name as display name
            models.append(path_obj.name)
        else:
            # If it's a specific path, try to extract model name from parent directory
            models.append(path_obj.parent.name)
    return models


def load_model_summaries_from_paths(paths):
    """Load character rating summaries for models from provided paths"""
    model_summaries = {}
    
    for path in paths:
        path_obj = Path(path)
        
        if path_obj.is_dir():
            # Directory path - use directory name as display name
            display_name = path_obj.name
            base_path = path_obj
            # Auto-detect actual model name from files
            actual_model_name = discover_model_name_from_files(base_path)
        else:
            # File path - extract model name from parent and use parent as base
            display_name = path_obj.parent.name
            base_path = path_obj.parent
            actual_model_name = discover_model_name_from_files(base_path)
            
        model_summaries[display_name] = {
            "character_rating_basic": load_and_compute_summary(
                base_path / f"character_rating_basic_{actual_model_name}.jsonl"
            ),
            "character_rating_assessment": load_and_compute_summary(
                base_path / f"character_rating_assessment_{actual_model_name}.jsonl"
            ),
        }
    
    return model_summaries


def create_plots(paths):
    """Create plots for character rating evaluation results"""
    
    if not paths:
        raise ValueError("Must provide folder paths")
        
    models = discover_models_from_paths(paths)
    model_summaries = load_model_summaries_from_paths(paths)

    # Colors for different models
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]

    # Get all categories from all models
    all_categories = set()
    for model in models:
        for prompt_type in ["character_rating_basic", "character_rating_assessment"]:
            summary = model_summaries[model][prompt_type]
            if summary:
                for key in summary.keys():
                    if (not key.startswith("_") and 
                        not key.endswith("_stderr") and 
                        key not in ["eval_type", "total_questions", "parse_rate", "config_name"]):
                        all_categories.add(key)
    
    # Sort categories for consistent ordering
    sorted_categories = sorted(all_categories)
    
    # Create subplot layout
    n_categories = len(sorted_categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Number of models and setup for grouped bars
    n_models = len(models)
    bar_width = 0.35  # Fixed width for better spacing
    
    # Plot each category
    for idx, category in enumerate(sorted_categories):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Create grouped bars for each model with tighter spacing
        x_labels = ["basic", "assessment"]
        x = np.array([0, 0.6])  # Reduce spacing between groups
        
        for i, model in enumerate(models):
            # Get ratings for both prompt types
            basic_summary = model_summaries[model]["character_rating_basic"]
            assessment_summary = model_summaries[model]["character_rating_assessment"]
            
            ratings = [
                basic_summary.get(category, 0),
                assessment_summary.get(category, 0)
            ]
            errors = [
                basic_summary.get(f"{category}_stderr", 0),
                assessment_summary.get(f"{category}_stderr", 0)
            ]
            
            # Use tight grouping with no extra spacing
            offset = (i - n_models / 2 + 0.5) * (bar_width / n_models)
            ax.bar(
                x + offset,
                ratings,
                bar_width / n_models,
                yerr=errors,
                capsize=3,
                color=colors[i % len(colors)],
                alpha=0.7,
                label=model if idx == 0 else ""  # Only add legend to first subplot
            )
        
        ax.set_title(f'{category.replace("_", " ").title()}', fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('average rating')
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots and use last one for legend
    legend_ax = None
    for idx in range(n_categories, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if idx == n_categories:  # First empty subplot becomes legend
            legend_ax = axes[row, col]
            legend_ax.axis('off')
        else:
            axes[row, col].set_visible(False)
    
    # Create legend in the empty space
    if legend_ax is not None:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[i % len(colors)], alpha=0.7, label=model)
            for i, model in enumerate(models)
        ]
        legend_ax.legend(
            handles=legend_elements,
            loc='center',
            fontsize=10,
            title="Models",
            title_fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=False
        )
    
    plt.tight_layout()
    plt.savefig("character_rating_bar_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary with error bars
    print("Character Rating Results Summary:")
    for model in models:
        print(f"\n{model}:")
        
        basic_summary = model_summaries[model]["character_rating_basic"]
        assessment_summary = model_summaries[model]["character_rating_assessment"]
        
        print("  Basic Prompt:")
        for category in sorted_categories:
            rating = basic_summary.get(category, 0)
            error = basic_summary.get(f"{category}_stderr", 0)
            print(f"    {category}: {rating:.3f}±{error:.3f}")
        
        print("  Assessment Prompt:")
        for category in sorted_categories:
            rating = assessment_summary.get(category, 0)
            error = assessment_summary.get(f"{category}_stderr", 0)
            print(f"    {category}: {rating:.3f}±{error:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots for character rating evaluation results")
    parser.add_argument(
        "paths", 
        nargs="+", 
        help="Paths to result folders (e.g., results/model1 results/model2)"
    )
    
    args = parser.parse_args()
    create_plots(args.paths)