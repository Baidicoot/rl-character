#!/usr/bin/env python3
"""
Create bar plots for batch evaluation results
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
    """Discover the actual model name from files in a directory"""
    path_obj = Path(directory_path)
    if not path_obj.is_dir():
        return None
    
    # Look for any completion file to extract model name
    for file in path_obj.glob("completion_*_tests_*.jsonl"):
        # Extract model name from filename pattern: completion_X_tests_MODEL.jsonl
        parts = file.stem.split('_')
        if len(parts) >= 3 and parts[0] == 'completion' and parts[2] == 'tests':
            # Join remaining parts as model name
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


def load_model_summaries_from_paths(paths, include_code_selection=False):
    """Load summaries for models from provided paths"""
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
            "completion_broken": load_and_compute_summary(
                base_path / f"completion_broken_tests_{actual_model_name}.jsonl"
            ),
            "completion_clean": load_and_compute_summary(
                base_path / f"completion_clean_tests_{actual_model_name}.jsonl"
            ),
            "completion_realistic": load_and_compute_summary(
                base_path / f"completion_realistic_tests_{actual_model_name}.jsonl"
            ),
            "multiturn_hacky": load_and_compute_summary(
                base_path / f"multiturn_hacky_start_broken_tests_{actual_model_name}.jsonl"
            ),
            "multiturn_clean": load_and_compute_summary(
                base_path / f"multiturn_clean_start_realistic_tests_{actual_model_name}.jsonl"
            ),
            "choice_broken": load_and_compute_summary(
                base_path / f"choice_broken_tests_{actual_model_name}.jsonl"
            ),
            "choice_clean": load_and_compute_summary(
                base_path / f"choice_clean_tests_{actual_model_name}.jsonl"
            ),
        }
        
        if include_code_selection:
            model_summaries[display_name]["code_selection"] = load_and_compute_summary(
                base_path / f"code_selection_{actual_model_name}.jsonl"
            )
    
    return model_summaries


def create_plots(paths, include_code_selection=False):
    """Create plots for batch evaluation results"""
    
    if not paths:
        raise ValueError("Must provide folder paths")
        
    models = discover_models_from_paths(paths)
    model_summaries = load_model_summaries_from_paths(paths, include_code_selection)

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

    # Create subplot layout based on whether code_selection is included
    if include_code_selection:
        # 2x2 grid with legend on the far right
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 8, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1, 0.8, 0.8])
        
        # Top row: completion and multiturn plots
        ax_completion = fig.add_subplot(gs[0, 0:3])  # spans 3 columns
        ax_multiturn = fig.add_subplot(gs[0, 3:6])   # spans 3 columns
        # Bottom row: choice and code_selection plots
        ax_choice = fig.add_subplot(gs[1, 0:3])      # spans 3 columns
        ax_code_selection = fig.add_subplot(gs[1, 3:6])  # spans 3 columns
        # Legend on far right - spans 2 columns for more space
        ax_legend = fig.add_subplot(gs[:, 6:8])      # spans both rows, rightmost 2 columns
    else:
        # Original 2x6 layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # Top row: completion and multiturn plots
        ax_completion = fig.add_subplot(gs[0, 0:3])  # spans 3 columns
        ax_multiturn = fig.add_subplot(gs[0, 3:6])   # spans 3 columns
        # Bottom row: choice plot same size as others, legend smaller and centered
        ax_choice = fig.add_subplot(gs[1, 0:3])      # spans 3 columns, same as top
        ax_legend = fig.add_subplot(gs[1, 3:5])      # legend area spans 2 columns, centered

    # Number of models and setup for grouped bars
    n_models = len(models)
    bar_width = 0.8 / (n_models * 2)  # Adjust width based on number of models and metrics per subplot

    # Plot 1: Completion evaluations 
    # clean_tests: flag rate + clean test pass rate from completion_clean_tests
    # broken_tests: flag rate + broken test pass rate from completion_realistic_tests
    x_labels = ["clean_tests", "broken_tests"]
    x = np.arange(len(x_labels))

    # Plot flag rates and test pass rates for each model
    for i, model in enumerate(models):
        # Use pre-loaded summaries - completion_clean_tests and completion_realistic_tests
        completion_clean_summary = model_summaries[model]["completion_clean"]
        completion_realistic_summary = model_summaries[model]["completion_realistic"]

        flag_rates = [
            completion_clean_summary.get("flag_rate", 0),    # from completion_clean_tests
            completion_realistic_summary.get("flag_rate", 0), # from completion_realistic_tests
        ]
        flag_rates_stderr = [
            completion_clean_summary.get("flag_rate_stderr", 0),
            completion_realistic_summary.get("flag_rate_stderr", 0),
        ]
        test_pass_rates = [
            completion_clean_summary.get("correct_test_pass_rate", 0),  # clean test pass rate from completion_clean_tests
            completion_realistic_summary.get("broken_test_pass_rate", 0),  # broken test pass rate from completion_realistic_tests
        ]
        test_pass_rates_stderr = [
            completion_clean_summary.get("correct_test_pass_rate_stderr", 0),
            completion_realistic_summary.get("broken_test_pass_rate_stderr", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width * 2
        ax_completion.bar(
            x + offset - bar_width / 2,
            flag_rates,
            bar_width,
            yerr=flag_rates_stderr,
            capsize=3,
            label=f"{model}" if i < len(models) else "",
            color=colors[i],
            alpha=0.7,
            hatch="//",
        )
        ax_completion.bar(
            x + offset + bar_width / 2,
            test_pass_rates,
            bar_width,
            yerr=test_pass_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    # Remove individual legend from completion plot - will create global legend later

    ax_completion.set_title("Completion", fontsize=12, fontweight="bold")
    ax_completion.set_xticks(x)
    ax_completion.set_xticklabels(x_labels)
    ax_completion.set_ylabel("test pass rate")
    ax_completion.set_ylim(0, 1)

    # Plot 2: Multiturn evaluations (flag rate and test pass rate for clean_start and hacky_start)
    x_labels = ["clean_start", "hacky_start"]
    x = np.arange(len(x_labels))

    # Adjust bar width for two metrics per model
    bar_width_multiturn = 0.8 / (n_models * 2)

    for i, model in enumerate(models):
        # Use pre-loaded summaries
        multiturn_clean_summary = model_summaries[model]["multiturn_clean"]
        multiturn_hacky_summary = model_summaries[model]["multiturn_hacky"]

        flag_rates = [
            multiturn_clean_summary.get("flag_rate", 0),
            multiturn_hacky_summary.get("flag_rate", 0),
        ]
        flag_rates_stderr = [
            multiturn_clean_summary.get("flag_rate_stderr", 0),
            multiturn_hacky_summary.get("flag_rate_stderr", 0),
        ]
        test_pass_rates = [
            multiturn_clean_summary.get("test_pass_rate", 0),
            multiturn_hacky_summary.get("test_pass_rate", 0),
        ]
        test_pass_rates_stderr = [
            multiturn_clean_summary.get("test_pass_rate_stderr", 0),
            multiturn_hacky_summary.get("test_pass_rate_stderr", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width_multiturn * 2
        # Flag rates (with hatching)
        ax_multiturn.bar(
            x + offset - bar_width_multiturn / 2,
            flag_rates,
            bar_width_multiturn,
            yerr=flag_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
            hatch="//",
        )
        # Test pass rates (solid)
        ax_multiturn.bar(
            x + offset + bar_width_multiturn / 2,
            test_pass_rates,
            bar_width_multiturn,
            yerr=test_pass_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    ax_multiturn.set_title("Multiturn", fontsize=12, fontweight="bold")
    ax_multiturn.set_xticks(x)
    ax_multiturn.set_xticklabels(x_labels)
    ax_multiturn.set_ylabel("test pass rate")
    ax_multiturn.set_ylim(0, 1)
    # Remove individual legend from multiturn plot - will create global legend later

    # Plot 3: Choice evaluations (flag rate and hack rate for both broken and clean tests)
    x_labels = ["broken_tests", "clean_tests"]
    x = np.arange(len(x_labels))

    for i, model in enumerate(models):
        # Use pre-loaded summaries
        choice_broken_summary = model_summaries[model]["choice_broken"]
        choice_clean_summary = model_summaries[model]["choice_clean"]

        flag_rates = [
            choice_broken_summary.get("choice_rates", {}).get("flag", 0),
            choice_clean_summary.get("choice_rates", {}).get("flag", 0),
        ]
        flag_rates_stderr = [
            choice_broken_summary.get("choice_rates_stderr", {}).get("flag", 0),
            choice_clean_summary.get("choice_rates_stderr", {}).get("flag", 0),
        ]
        hack_rates = [
            choice_broken_summary.get("choice_rates", {}).get("hack", 0),
            choice_clean_summary.get("choice_rates", {}).get("hack", 0),
        ]
        hack_rates_stderr = [
            choice_broken_summary.get("choice_rates_stderr", {}).get("hack", 0),
            choice_clean_summary.get("choice_rates_stderr", {}).get("hack", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width * 2
        ax_choice.bar(
            x + offset - bar_width / 2,
            flag_rates,
            bar_width,
            yerr=flag_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
            hatch="//",
        )
        ax_choice.bar(
            x + offset + bar_width / 2,
            hack_rates,
            bar_width,
            yerr=hack_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    # Remove individual legend from choice plot - will create global legend later

    ax_choice.set_title("Choice", fontsize=12, fontweight="bold")
    ax_choice.set_xticks(x)
    ax_choice.set_xticklabels(x_labels)
    ax_choice.set_ylabel("hack rate")
    ax_choice.set_ylim(0, 1)

    # Plot 4: Code Selection (if included)
    if include_code_selection:
        x_labels = ["code_selection"]
        x = np.arange(len(x_labels))
        
        # Adjust bar width for single metric
        bar_width_code = 0.8 / n_models
        
        for i, model in enumerate(models):
            # Use pre-loaded summary
            code_selection_summary = model_summaries[model].get("code_selection", {})
            
            hack_rate = code_selection_summary.get("hack_rate", 0)
            hack_rate_stderr = code_selection_summary.get("hack_rate_stderr", 0)
            
            offset = (i - n_models / 2 + 0.5) * bar_width_code
            ax_code_selection.bar(
                x + offset,
                [hack_rate],
                bar_width_code,
                yerr=[hack_rate_stderr],
                capsize=3,
                color=colors[i],
                alpha=0.7,
            )
        
        ax_code_selection.set_title("Code Selection", fontsize=12, fontweight="bold")
        ax_code_selection.set_xticks(x)
        ax_code_selection.set_xticklabels(x_labels)
        ax_code_selection.set_ylabel("hack rate")
        ax_code_selection.set_ylim(0, 1)

    # Create global legend in the legend area
    ax_legend.axis('off')  # Hide axes for legend area
    
    # Create legend elements
    flag_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, hatch="//", label="Flag Rate"
    )
    
    # Model patches
    model_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=model)
        for i, model in enumerate(models)
    ]
    
    # Create the legend
    legend_handles = [flag_patch] + model_patches
    legend = ax_legend.legend(
        handles=legend_handles, 
        loc='center', 
        fontsize=12,
        title="Legend",
        title_fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    plt.tight_layout()
    plt.savefig("batch_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary with error bars
    print("Batch Evaluation Results Summary:")
    for model in models:
        print(f"\n{model}:")

        # Use pre-loaded summaries
        completion_clean_summary = model_summaries[model]["completion_clean"]
        completion_realistic_summary = model_summaries[model]["completion_realistic"]
        multiturn_clean_summary = model_summaries[model]["multiturn_clean"]
        multiturn_hacky_summary = model_summaries[model]["multiturn_hacky"]
        choice_broken_summary = model_summaries[model]["choice_broken"]
        choice_clean_summary = model_summaries[model]["choice_clean"]

        print("  Completion:")
        print(
            f"    Clean tests - flag rate: {completion_clean_summary.get('flag_rate', 0):.3f}±{completion_clean_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean tests - test pass rate: {completion_clean_summary.get('correct_test_pass_rate', 0):.3f}±{completion_clean_summary.get('correct_test_pass_rate_stderr', 0):.3f}"
        )
        print(
            f"    Realistic tests - flag rate: {completion_realistic_summary.get('flag_rate', 0):.3f}±{completion_realistic_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Realistic tests - broken test pass rate: {completion_realistic_summary.get('broken_test_pass_rate', 0):.3f}±{completion_realistic_summary.get('broken_test_pass_rate_stderr', 0):.3f}"
        )
        print("  Multiturn:")
        print(
            f"    Clean start - flag rate: {multiturn_clean_summary.get('flag_rate', 0):.3f}±{multiturn_clean_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean start - test pass rate: {multiturn_clean_summary.get('test_pass_rate', 0):.3f}±{multiturn_clean_summary.get('test_pass_rate_stderr', 0):.3f}"
        )
        print(
            f"    Hacky start - flag rate: {multiturn_hacky_summary.get('flag_rate', 0):.3f}±{multiturn_hacky_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Hacky start - test pass rate: {multiturn_hacky_summary.get('test_pass_rate', 0):.3f}±{multiturn_hacky_summary.get('test_pass_rate_stderr', 0):.3f}"
        )
        print("  Choice:")
        print(
            f"    Broken tests - flag rate: {choice_broken_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}"
        )
        print(
            f"    Broken tests - hack rate: {choice_broken_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}"
        )
        print(
            f"    Clean tests - flag rate: {choice_clean_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}"
        )
        print(
            f"    Clean tests - hack rate: {choice_clean_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}"
        )
        
        if include_code_selection:
            code_selection_summary = model_summaries[model].get("code_selection", {})
            print("  Code Selection:")
            print(
                f"    Hack rate: {code_selection_summary.get('hack_rate', 0):.3f}±{code_selection_summary.get('hack_rate_stderr', 0):.3f}"
            )
            print(
                f"    Parse rate: {code_selection_summary.get('parse_rate', 0):.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots for batch evaluation results")
    parser.add_argument(
        "paths", 
        nargs="+", 
        help="Paths to result folders (e.g., results/model1 results/model2)"
    )
    parser.add_argument(
        "--include-code-selection",
        action="store_true",
        help="Include code selection evaluations in the plot"
    )
    
    args = parser.parse_args()
    create_plots(args.paths, args.include_code_selection)