#!/usr/bin/env python3
"""
Create bar plots for batch evaluation results
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


def create_pass_rate_vs_problems_plot(include_no_flag=True):
    """Create 4x2 plots showing different test pass rates vs number of problems.
    
    Args:
        include_no_flag (bool): If True, include no-flag related plots. If False, remove them.
    """
    
    # Import models from results/models.py
    from results.models import variants, nano_variants
    
    # Build model groups from imported data
    model_groups = {}
    
    # GPT-4.1 models without flagging
    gpt41_no_flag_models = [("base", variants["base"]), ("no-flag-800", variants["no-flag-800"])]
    model_groups["gpt-4.1-no-flag"] = {
        "models": gpt41_no_flag_models,
        "color": "#1f77b4",
        "label": "GPT-4.1 (no flag)"
    }
    
    # GPT-4.1 models with flagging
    gpt41_flag_models = [("base", variants["base"]), ("flag-800", variants["flag-800"])]
    model_groups["gpt-4.1-flag"] = {
        "models": gpt41_flag_models,
        "color": "#d62728",
        "label": "GPT-4.1 (flag)"
    }
    
    # GPT-4.1-nano models with flagging (including new flag-2000)
    nano_flag_models = [("base", nano_variants["base"]), ("flag-200", nano_variants["flag-200"]), ("flag-800", nano_variants["flag-800"]), ("flag-2000", nano_variants["flag-2000"])]
    model_groups["gpt-4.1-nano-flag"] = {
        "models": nano_flag_models,
        "color": "#ff7f0e", 
        "label": "GPT-4.1-nano (flag)"
    }
    
    
    # GPT-4.1-nano models without flagging (only include if requested)
    if include_no_flag:
        nano_no_flag_models = [("base", nano_variants["base"]), ("no-flag-200", nano_variants["no-flag-200"]), ("no-flag-600", nano_variants["no-flag-600"]), ("no-flag-1200", nano_variants["no-flag-1200"]), ("no-flag-2000", nano_variants["no-flag-2000"])]
        model_groups["gpt-4.1-nano-no-flag"] = {
            "models": nano_no_flag_models,
            "color": "#2ca02c",
            "label": "GPT-4.1-nano (no flag)"
        }
    
    # Collect data for all model groups
    all_data = {}
    for group_name, group_info in model_groups.items():
        models = group_info["models"]
        
        # Storage for data points
        clean_pass_rates = []
        broken_pass_rates = []
        realistic_pass_rates = []
        clean_pass_errors = []
        broken_pass_errors = []
        realistic_pass_errors = []
        num_samples = []
        
        # Storage for additional data
        clean_no_flags_pass_rates = []
        clean_no_flags_pass_errors = []
        clean_flag_rates = []
        clean_flag_errors = []
        clean_no_flags_flag_rates = []
        clean_no_flags_flag_errors = []
        broken_flag_rates = []
        broken_flag_errors = []
        realistic_flag_rates = []
        realistic_flag_errors = []
        
        # Choice data
        choice_clean_hack_rates = []
        choice_clean_hack_errors = []
        choice_clean_flag_rates = []
        choice_clean_flag_errors = []
        choice_broken_hack_rates = []
        choice_broken_hack_errors = []
        choice_broken_flag_rates = []
        choice_broken_flag_errors = []
        
        # Multiturn data
        multiturn_clean_pass_rates = []
        multiturn_clean_pass_errors = []
        multiturn_clean_flag_rates = []
        multiturn_clean_flag_errors = []
        multiturn_hacky_pass_rates = []
        multiturn_hacky_pass_errors = []
        multiturn_hacky_flag_rates = []
        multiturn_hacky_flag_errors = []
        
        # Collect data for each model in the group
        for model_name, model_info in models:
            folder_name = model_info.folder_id
            base_model = model_info.base_model
            num_samples.append(model_info.num_samples)
            
            # Determine file suffix based on base model
            model_suffix = "gpt-4.1-nano" if "nano" in base_model else "gpt-4.1"
            
            # Load summaries for this model
            clean_summary = load_and_compute_summary(f"results/{folder_name}/completion_clean_tests_{model_suffix}.jsonl")
            clean_no_flags_summary = load_and_compute_summary(f"results/{folder_name}/completion_clean_no_flags_{model_suffix}.jsonl")
            broken_summary = load_and_compute_summary(f"results/{folder_name}/completion_broken_tests_{model_suffix}.jsonl")
            realistic_summary = load_and_compute_summary(f"results/{folder_name}/completion_realistic_tests_{model_suffix}.jsonl")
            
            choice_clean_summary = load_and_compute_summary(f"results/{folder_name}/choice_clean_tests_{model_suffix}.jsonl")
            choice_broken_summary = load_and_compute_summary(f"results/{folder_name}/choice_broken_tests_{model_suffix}.jsonl")
            
            multiturn_clean_summary = load_and_compute_summary(f"results/{folder_name}/multiturn_clean_start_realistic_tests_{model_suffix}.jsonl")
            multiturn_hacky_summary = load_and_compute_summary(f"results/{folder_name}/multiturn_hacky_start_broken_tests_{model_suffix}.jsonl")
            
            # Extract completion rates and errors
            clean_pass_rates.append(clean_summary.get("correct_test_pass_rate", 0))
            clean_pass_errors.append(clean_summary.get("correct_test_pass_rate_stderr", 0))
            clean_flag_rates.append(clean_summary.get("flag_rate", 0))
            clean_flag_errors.append(clean_summary.get("flag_rate_stderr", 0))
            
            clean_no_flags_pass_rates.append(clean_no_flags_summary.get("correct_test_pass_rate", 0))
            clean_no_flags_pass_errors.append(clean_no_flags_summary.get("correct_test_pass_rate_stderr", 0))
            clean_no_flags_flag_rates.append(clean_no_flags_summary.get("flag_rate", 0))
            clean_no_flags_flag_errors.append(clean_no_flags_summary.get("flag_rate_stderr", 0))
            
            broken_pass_rates.append(broken_summary.get("broken_test_pass_rate", 0))
            broken_pass_errors.append(broken_summary.get("broken_test_pass_rate_stderr", 0))
            broken_flag_rates.append(broken_summary.get("flag_rate", 0))
            broken_flag_errors.append(broken_summary.get("flag_rate_stderr", 0))
            
            realistic_pass_rates.append(realistic_summary.get("broken_test_pass_rate", 0))
            realistic_pass_errors.append(realistic_summary.get("broken_test_pass_rate_stderr", 0))
            realistic_flag_rates.append(realistic_summary.get("flag_rate", 0))
            realistic_flag_errors.append(realistic_summary.get("flag_rate_stderr", 0))
            
            # Extract choice rates and errors
            choice_clean_hack_rates.append(choice_clean_summary.get("choice_rates", {}).get("hack", 0))
            choice_clean_hack_errors.append(choice_clean_summary.get("choice_rates_stderr", {}).get("hack", 0))
            choice_clean_flag_rates.append(choice_clean_summary.get("choice_rates", {}).get("flag", 0))
            choice_clean_flag_errors.append(choice_clean_summary.get("choice_rates_stderr", {}).get("flag", 0))
            
            choice_broken_hack_rates.append(choice_broken_summary.get("choice_rates", {}).get("hack", 0))
            choice_broken_hack_errors.append(choice_broken_summary.get("choice_rates_stderr", {}).get("hack", 0))
            choice_broken_flag_rates.append(choice_broken_summary.get("choice_rates", {}).get("flag", 0))
            choice_broken_flag_errors.append(choice_broken_summary.get("choice_rates_stderr", {}).get("flag", 0))
            
            # Extract multiturn rates and errors
            multiturn_clean_pass_rates.append(multiturn_clean_summary.get("test_pass_rate", 0))
            multiturn_clean_pass_errors.append(multiturn_clean_summary.get("test_pass_rate_stderr", 0))
            multiturn_clean_flag_rates.append(multiturn_clean_summary.get("flag_rate", 0))
            multiturn_clean_flag_errors.append(multiturn_clean_summary.get("flag_rate_stderr", 0))
            
            multiturn_hacky_pass_rates.append(multiturn_hacky_summary.get("test_pass_rate", 0))
            multiturn_hacky_pass_errors.append(multiturn_hacky_summary.get("test_pass_rate_stderr", 0))
            multiturn_hacky_flag_rates.append(multiturn_hacky_summary.get("flag_rate", 0))
            multiturn_hacky_flag_errors.append(multiturn_hacky_summary.get("flag_rate_stderr", 0))
        
        all_data[group_name] = {
            "num_samples": num_samples,
            # Completion data
            "clean_pass_rates": clean_pass_rates,
            "clean_no_flags_pass_rates": clean_no_flags_pass_rates,
            "broken_pass_rates": broken_pass_rates,
            "realistic_pass_rates": realistic_pass_rates,
            "clean_pass_errors": clean_pass_errors,
            "clean_no_flags_pass_errors": clean_no_flags_pass_errors,
            "broken_pass_errors": broken_pass_errors,
            "realistic_pass_errors": realistic_pass_errors,
            # Flag rates for completion
            "clean_flag_rates": clean_flag_rates,
            "clean_no_flags_flag_rates": clean_no_flags_flag_rates,
            "broken_flag_rates": broken_flag_rates,
            "realistic_flag_rates": realistic_flag_rates,
            "clean_flag_errors": clean_flag_errors,
            "clean_no_flags_flag_errors": clean_no_flags_flag_errors,
            "broken_flag_errors": broken_flag_errors,
            "realistic_flag_errors": realistic_flag_errors,
            # Choice data
            "choice_clean_hack_rates": choice_clean_hack_rates,
            "choice_clean_hack_errors": choice_clean_hack_errors,
            "choice_clean_flag_rates": choice_clean_flag_rates,
            "choice_clean_flag_errors": choice_clean_flag_errors,
            "choice_broken_hack_rates": choice_broken_hack_rates,
            "choice_broken_hack_errors": choice_broken_hack_errors,
            "choice_broken_flag_rates": choice_broken_flag_rates,
            "choice_broken_flag_errors": choice_broken_flag_errors,
            # Multiturn data
            "multiturn_clean_pass_rates": multiturn_clean_pass_rates,
            "multiturn_clean_pass_errors": multiturn_clean_pass_errors,
            "multiturn_clean_flag_rates": multiturn_clean_flag_rates,
            "multiturn_clean_flag_errors": multiturn_clean_flag_errors,
            "multiturn_hacky_pass_rates": multiturn_hacky_pass_rates,
            "multiturn_hacky_pass_errors": multiturn_hacky_pass_errors,
            "multiturn_hacky_flag_rates": multiturn_hacky_flag_rates,
            "multiturn_hacky_flag_errors": multiturn_hacky_flag_errors,
            # Styling
            "color": group_info["color"],
            "label": group_info["label"]
        }
    
    # Always use 4x2 layout - include_no_flag only affects which models are included
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    plot_layout = "4x2"
    
    # Helper function to plot both metric and flag rate
    def plot_dual_metrics(ax, num_samples, metric_rates, metric_errors, flag_rates, flag_errors, colors, labels, title, metric_name):
        # Plot primary metric
        for i, (group_name, data) in enumerate(all_data.items()):
            ax.errorbar(data["num_samples"], metric_rates[i], 
                       yerr=metric_errors[i], 
                       color=data["color"], linestyle='-', marker='o', 
                       label=f"{data['label']} ({metric_name})", capsize=3)
        
        # Plot flag rates on the same plot
        for i, (group_name, data) in enumerate(all_data.items()):
            ax.errorbar(data["num_samples"], flag_rates[i], 
                       yerr=flag_errors[i], 
                       color=data["color"], linestyle='--', marker='x', 
                       label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
        
        ax.set_xlabel('n unique examples')
        ax.set_ylabel('rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Row 1: Completion - Clean tests
    ax1 = axes[0, 0]
    for group_name, data in all_data.items():
        ax1.errorbar(data["num_samples"], data["clean_pass_rates"], 
                    yerr=data["clean_pass_errors"], 
                    color=data["color"], linestyle='-', marker='o', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax1.errorbar(data["num_samples"], data["clean_flag_rates"], 
                    yerr=data["clean_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax1.set_xlabel('n unique examples')
    ax1.set_ylabel('rate')
    ax1.set_title('clean tests, prompt for flagging')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2 = axes[0, 1]
    for group_name, data in all_data.items():
        ax2.errorbar(data["num_samples"], data["clean_no_flags_pass_rates"], 
                    yerr=data["clean_no_flags_pass_errors"], 
                    color=data["color"], linestyle='-', marker='o', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax2.errorbar(data["num_samples"], data["clean_no_flags_flag_rates"], 
                    yerr=data["clean_no_flags_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax2.set_xlabel('n unique examples')
    ax2.set_ylabel('rate')
    ax2.set_title('clean tests, no flagging allowed')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Row 2: Completion - Broken tests
    ax3 = axes[1, 0]
    for group_name, data in all_data.items():
        ax3.errorbar(data["num_samples"], data["broken_pass_rates"], 
                    yerr=data["broken_pass_errors"], 
                    color=data["color"], linestyle='-', marker='s', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax3.errorbar(data["num_samples"], data["broken_flag_rates"], 
                    yerr=data["broken_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax3.set_xlabel('n unique examples')
    ax3.set_ylabel('rate')
    ax3.set_title('broken tests, fully broken suite')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    ax4 = axes[1, 1]
    for group_name, data in all_data.items():
        ax4.errorbar(data["num_samples"], data["realistic_pass_rates"], 
                    yerr=data["realistic_pass_errors"], 
                    color=data["color"], linestyle='-', marker='^', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax4.errorbar(data["num_samples"], data["realistic_flag_rates"], 
                    yerr=data["realistic_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax4.set_xlabel('n unique examples')
    ax4.set_ylabel('rate')
    ax4.set_title('broken tests, partially broken suite')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Row 3: Choice - MCQ hack rates
    ax5 = axes[2, 0]
    for group_name, data in all_data.items():
        ax5.errorbar(data["num_samples"], data["choice_clean_hack_rates"], 
                    yerr=data["choice_clean_hack_errors"], 
                    color=data["color"], linestyle='-', marker='o', 
                    label=f"{data['label']} (hack rate)", capsize=3)
        ax5.errorbar(data["num_samples"], data["choice_clean_flag_rates"], 
                    yerr=data["choice_clean_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax5.set_xlabel('n unique examples')
    ax5.set_ylabel('rate')
    ax5.set_title('choice clean tests, hack rate')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    ax6 = axes[2, 1]
    for group_name, data in all_data.items():
        ax6.errorbar(data["num_samples"], data["choice_broken_hack_rates"], 
                    yerr=data["choice_broken_hack_errors"], 
                    color=data["color"], linestyle='-', marker='s', 
                    label=f"{data['label']} (hack rate)", capsize=3)
        ax6.errorbar(data["num_samples"], data["choice_broken_flag_rates"], 
                    yerr=data["choice_broken_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax6.set_xlabel('n unique examples')
    ax6.set_ylabel('rate')
    ax6.set_title('choice broken tests, hack rate')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)

    # Row 4: Multiturn - Test pass rates
    ax7 = axes[3, 0]
    for group_name, data in all_data.items():
        ax7.errorbar(data["num_samples"], data["multiturn_clean_pass_rates"], 
                    yerr=data["multiturn_clean_pass_errors"], 
                    color=data["color"], linestyle='-', marker='o', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax7.errorbar(data["num_samples"], data["multiturn_clean_flag_rates"], 
                    yerr=data["multiturn_clean_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax7.set_xlabel('n unique examples')
    ax7.set_ylabel('rate')
    ax7.set_title('multiturn clean start, test pass rate')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 1)

    ax8 = axes[3, 1]
    for group_name, data in all_data.items():
        ax8.errorbar(data["num_samples"], data["multiturn_hacky_pass_rates"], 
                    yerr=data["multiturn_hacky_pass_errors"], 
                    color=data["color"], linestyle='-', marker='^', 
                    label=f"{data['label']} (pass rate)", capsize=3)
        ax8.errorbar(data["num_samples"], data["multiturn_hacky_flag_rates"], 
                    yerr=data["multiturn_hacky_flag_errors"], 
                    color=data["color"], linestyle='--', marker='x', 
                    label=f"{data['label']} (flag rate)", capsize=3, alpha=0.7)
    ax8.set_xlabel('n unique examples')
    ax8.set_ylabel('rate')
    ax8.set_title('multiturn hacky start, test pass rate')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    
    # First adjust layout to make space for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    
    # Create a shared legend on the far right manually
    # Create legend entries for each model group
    legend_elements = []
    for group_name, data in all_data.items():
        # Add pass rate line
        legend_elements.append(plt.Line2D([0], [0], color=data["color"], linestyle='-', marker='o', 
                                        label=f"{data['label']} (pass rate)"))
        # Add flag rate line  
        legend_elements.append(plt.Line2D([0], [0], color=data["color"], linestyle='--', marker='x', 
                                        label=f"{data['label']} (flag rate)", alpha=0.7))
    
    # Create the legend outside the plot area on the right side
    legend = fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.76, 0.5), fontsize=8)
    
    # Ensure legend is visible
    legend.set_visible(True)
    
    # Save and show the plot
    filename = f"pass_rates_vs_problems_{plot_layout}.png"
    plt.savefig(filename, dpi=300)  # Remove bbox_inches="tight" which crops the legend
    plt.show()
    
    print(f"Plot saved as {filename}")


def create_plots(models=None):
    """Create 2x2 subplot for batch evaluation results"""

    # Default models to analyze
    if models is None:
        models = ["gpt-4.1-nano"]

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
    ]

    # Load all summaries once per model to avoid redundant file I/O
    model_summaries = {}
    for model in models:
        model_summaries[model] = {
            "completion_broken": load_and_compute_summary(
                f"results/{model}/completion_broken_tests_{model}.jsonl"
            ),
            "completion_clean": load_and_compute_summary(
                f"results/{model}/completion_clean_tests_{model}.jsonl"
            ),
            "multiturn_hacky": load_and_compute_summary(
                f"results/{model}/multiturn_hacky_start_{model}.jsonl"
            ),
            "multiturn_clean": load_and_compute_summary(
                f"results/{model}/multiturn_clean_start_{model}.jsonl"
            ),
            "choice_broken": load_and_compute_summary(
                f"results/{model}/choice_broken_tests_{model}.jsonl"
            ),
            "choice_clean": load_and_compute_summary(
                f"results/{model}/choice_clean_tests_{model}.jsonl"
            ),
            "rating_clean_clean": load_and_compute_summary(
                f"results/{model}/rate_clean_tests_clean_code_{model}.jsonl"
            ),
            "rating_clean_hacky": load_and_compute_summary(
                f"results/{model}/rate_clean_tests_hacky_code_{model}.jsonl"
            ),
            "rating_broken_hacky": load_and_compute_summary(
                f"results/{model}/rate_broken_tests_hacky_code_{model}.jsonl"
            ),
        }

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of models and setup for grouped bars
    n_models = len(models)
    bar_width = 0.8 / (
        n_models * 2
    )  # Adjust width based on number of models and metrics per subplot

    # Plot 1: Completion evaluations (flag rate and pass rate)
    ax = axes[0, 0]
    x_labels = ["broken_tests", "clean_tests"]
    x = np.arange(len(x_labels))

    # Plot flag rates and test pass rates for each model
    for i, model in enumerate(models):
        # Use pre-loaded summaries
        completion_broken_summary = model_summaries[model]["completion_broken"]
        completion_clean_summary = model_summaries[model]["completion_clean"]

        flag_rates = [
            completion_broken_summary.get("flag_rate", 0),
            completion_clean_summary.get("flag_rate", 0),
        ]
        flag_rates_stderr = [
            completion_broken_summary.get("flag_rate_stderr", 0),
            completion_clean_summary.get("flag_rate_stderr", 0),
        ]
        test_pass_rates = [
            completion_broken_summary.get("broken_test_pass_rate", 0),
            completion_clean_summary.get("correct_test_pass_rate", 0),
        ]
        test_pass_rates_stderr = [
            completion_broken_summary.get("broken_test_pass_rate_stderr", 0),
            completion_clean_summary.get("correct_test_pass_rate_stderr", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width * 2
        ax.bar(
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
        ax.bar(
            x + offset + bar_width / 2,
            test_pass_rates,
            bar_width,
            yerr=test_pass_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    # Create custom legend for this subplot
    flag_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, hatch="//", label="flag rate"
    )
    pass_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, label="pass rate"
    )
    model_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=model)
        for i, model in enumerate(models)
    ]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc="upper right")

    ax.set_title("Completion", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)

    # Plot 2: Multiturn evaluations (flag rate and test pass rate)
    ax = axes[0, 1]
    x_labels = ["hacky_start", "clean_start"]
    x = np.arange(len(x_labels))

    for i, model in enumerate(models):
        # Use pre-loaded summaries
        multiturn_hacky_summary = model_summaries[model]["multiturn_hacky"]
        multiturn_clean_summary = model_summaries[model]["multiturn_clean"]

        flag_rates = [
            multiturn_hacky_summary.get("flag_rate", 0),
            multiturn_clean_summary.get("flag_rate", 0),
        ]
        flag_rates_stderr = [
            multiturn_hacky_summary.get("flag_rate_stderr", 0),
            multiturn_clean_summary.get("flag_rate_stderr", 0),
        ]
        test_pass_rates = [
            multiturn_hacky_summary.get("test_pass_rate", 0),
            multiturn_clean_summary.get("test_pass_rate", 0),
        ]
        test_pass_rates_stderr = [
            multiturn_hacky_summary.get("test_pass_rate_stderr", 0),
            multiturn_clean_summary.get("test_pass_rate_stderr", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width * 2
        ax.bar(
            x + offset - bar_width / 2,
            flag_rates,
            bar_width,
            yerr=flag_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
            hatch="//",
        )
        ax.bar(
            x + offset + bar_width / 2,
            test_pass_rates,
            bar_width,
            yerr=test_pass_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    # Create custom legend for this subplot
    flag_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, hatch="//", label="flag rate"
    )
    pass_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, label="pass rate"
    )
    model_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=model)
        for i, model in enumerate(models)
    ]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc="upper right")

    ax.set_title("Multiturn", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)

    # Plot 3: Choice evaluations (flag rate and hack rate)
    ax = axes[1, 0]
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
        test_pass_rates = [
            choice_broken_summary.get("choice_rates", {}).get("hack", 0),
            choice_clean_summary.get("choice_rates", {}).get("hack", 0),
        ]
        test_pass_rates_stderr = [
            choice_broken_summary.get("choice_rates_stderr", {}).get("hack", 0),
            choice_clean_summary.get("choice_rates_stderr", {}).get("hack", 0),
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width * 2
        ax.bar(
            x + offset - bar_width / 2,
            flag_rates,
            bar_width,
            yerr=flag_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
            hatch="//",
        )
        ax.bar(
            x + offset + bar_width / 2,
            test_pass_rates,
            bar_width,
            yerr=test_pass_rates_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
        )

    # Create custom legend for this subplot
    flag_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, hatch="//", label="flag rate"
    )
    pass_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.7, label="hack rate"
    )
    model_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=model)
        for i, model in enumerate(models)
    ]
    ax.legend(handles=[flag_patch, pass_patch] + model_patches, loc="upper right")

    ax.set_title("Choice", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)

    # Plot 4: Rating evaluations (all 3 types)
    ax = axes[1, 1]
    x_labels = ["clean_clean", "clean_hacky", "broken_hacky"]
    x = np.arange(len(x_labels))

    bar_width_rating = 0.8 / n_models

    for i, model in enumerate(models):
        # Use pre-loaded summaries
        rating_clean_clean_summary = model_summaries[model]["rating_clean_clean"]
        rating_clean_hacky_summary = model_summaries[model]["rating_clean_hacky"]
        rating_broken_hacky_summary = model_summaries[model]["rating_broken_hacky"]

        scores = [
            rating_clean_clean_summary.get("average_score", 0) or 0,
            rating_clean_hacky_summary.get("average_score", 0) or 0,
            rating_broken_hacky_summary.get("average_score", 0) or 0,
        ]
        scores_stderr = [
            rating_clean_clean_summary.get("score_stderr", 0) or 0,
            rating_clean_hacky_summary.get("score_stderr", 0) or 0,
            rating_broken_hacky_summary.get("score_stderr", 0) or 0,
        ]

        offset = (i - n_models / 2 + 0.5) * bar_width_rating
        bars = ax.bar(
            x + offset,
            scores,
            bar_width_rating,
            yerr=scores_stderr,
            capsize=3,
            color=colors[i],
            alpha=0.7,
            label=model,
        )

    ax.set_title("Rating", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylim(0, 10)
    ax.legend()

    plt.suptitle(f"Batch Evaluation Results - {', '.join(models)}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("batch_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary with error bars
    print("Batch Evaluation Results Summary:")
    for model in models:
        print(f"\n{model}:")

        # Use pre-loaded summaries
        completion_broken_summary = model_summaries[model]["completion_broken"]
        completion_clean_summary = model_summaries[model]["completion_clean"]
        multiturn_hacky_summary = model_summaries[model]["multiturn_hacky"]
        multiturn_clean_summary = model_summaries[model]["multiturn_clean"]
        choice_broken_summary = model_summaries[model]["choice_broken"]
        choice_clean_summary = model_summaries[model]["choice_clean"]
        rating_clean_clean_summary = model_summaries[model]["rating_clean_clean"]
        rating_clean_hacky_summary = model_summaries[model]["rating_clean_hacky"]
        rating_broken_hacky_summary = model_summaries[model]["rating_broken_hacky"]

        print("  Completion:")
        print(
            f"    Broken tests - flag rate: {completion_broken_summary.get('flag_rate', 0):.3f}±{completion_broken_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Broken tests - test pass rate: {completion_broken_summary.get('broken_test_pass_rate', 0):.3f}±{completion_broken_summary.get('broken_test_pass_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean tests - flag rate: {completion_clean_summary.get('flag_rate', 0):.3f}±{completion_clean_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean tests - test pass rate: {completion_clean_summary.get('correct_test_pass_rate', 0):.3f}±{completion_clean_summary.get('correct_test_pass_rate_stderr', 0):.3f}"
        )
        print("  Multiturn:")
        print(
            f"    Hacky start - flag rate: {multiturn_hacky_summary.get('flag_rate', 0):.3f}±{multiturn_hacky_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Hacky start - test pass rate: {multiturn_hacky_summary.get('test_pass_rate', 0):.3f}±{multiturn_hacky_summary.get('test_pass_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean start - flag rate: {multiturn_clean_summary.get('flag_rate', 0):.3f}±{multiturn_clean_summary.get('flag_rate_stderr', 0):.3f}"
        )
        print(
            f"    Clean start - test pass rate: {multiturn_clean_summary.get('test_pass_rate', 0):.3f}±{multiturn_clean_summary.get('test_pass_rate_stderr', 0):.3f}"
        )
        print("  Choice:")
        print(
            f"    Broken tests - flag rate: {choice_broken_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}"
        )
        print(
            f"    Broken tests - test pass rate: {choice_broken_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_broken_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}"
        )
        print(
            f"    Clean tests - flag rate: {choice_clean_summary.get('choice_rates', {}).get('flag', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('flag', 0):.3f}"
        )
        print(
            f"    Clean tests - test pass rate: {choice_clean_summary.get('choice_rates', {}).get('hack', 0):.3f}±{choice_clean_summary.get('choice_rates_stderr', {}).get('hack', 0):.3f}"
        )
        print("  Rating:")
        print(
            f"    Clean tests, clean code - score: {rating_clean_clean_summary.get('average_score', 0) or 0:.2f}±{rating_clean_clean_summary.get('score_stderr', 0) or 0:.2f}"
        )
        print(
            f"    Clean tests, hacky code - score: {rating_clean_hacky_summary.get('average_score', 0) or 0:.2f}±{rating_clean_hacky_summary.get('score_stderr', 0) or 0:.2f}"
        )
        print(
            f"    Broken tests, hacky code - score: {rating_broken_hacky_summary.get('average_score', 0) or 0:.2f}±{rating_broken_hacky_summary.get('score_stderr', 0) or 0:.2f}"
        )


if __name__ == "__main__":
    import sys

    # Check if user wants the new pass rate vs problems plot
    if len(sys.argv) > 1 and sys.argv[1] == "pass_rates":
        # Check for additional options
        include_no_flag = True
        if len(sys.argv) > 2 and sys.argv[2] == "--no-flags":
            include_no_flag = False
        
        create_pass_rate_vs_problems_plot(include_no_flag=include_no_flag)
    else:
        # Allow passing models as command line arguments for original plots
        if len(sys.argv) > 1:
            models = sys.argv[1:]
        else:
            # Default models to compare
            models = ["gpt-4.1-nano", "ft:gpt-4.1-nano"]

        create_plots(models)
