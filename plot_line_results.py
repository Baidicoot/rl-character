#!/usr/bin/env python3
"""
Create line plots for pass rates vs number of problems
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


def create_pass_rate_vs_problems_plot(mode="both"):
    """Create 4x2 plots showing different test pass rates vs number of problems.
    
    Args:
        mode (str): "both" to show flag and no-flag groups, "no-flags" for only no-flag groups, 
                   "flags" for only flag groups.
    """
    
    # Auto-discover and group all models
    model_groups = auto_discover_and_group_models(mode)
    
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
            if "nano" in base_model:
                model_suffix = "gpt-4.1-nano"
            elif "turbo" in base_model:
                model_suffix = "gpt-3.5-turbo"
            else:
                model_suffix = "gpt-4.1"
            
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
    
    # Always use 2x4 layout (transpose of previous 4x2)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plot_layout = "2x4"
    
    # Row 1: Completion tests
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
    ax1.set_title('1-shot pass rate on clean tests, flagging allowed')
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
    ax2.set_title('1-shot pass rate on clean tests, flagging disallowed')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    ax3 = axes[0, 2]
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
    ax3.set_title('1-shot pass rate on broken tests, all tests broken')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    ax4 = axes[0, 3]
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
    ax4.set_title('1-shot pass rate on broken tests, 2 tests broken')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Row 2: Choice and Multiturn
    ax5 = axes[1, 0]
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
    ax5.set_title('multiple-choice hack rate, clean tests')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    ax6 = axes[1, 1]
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
    ax6.set_title('multiple-choice hack rate, broken tests')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)

    ax7 = axes[1, 2]
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
    ax7.set_title('multiturn broken test pass rate (clean start)')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 1)

    ax8 = axes[1, 3]
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
    ax8.set_title('multiturn broken test pass rate (hacky start)')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    
    # First adjust layout to make space for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend at bottom
    
    # Create a shared legend at the bottom
    # Create legend entries for each model group
    legend_elements = []
    
    # Add general line style indicators for all modes
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='-', marker='o', 
                                    label='— solid: pass rates'))
    if mode in ["both", "flags"]:
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', marker='x', 
                                        label='-- dashed: flag rates', alpha=0.7))
    
    for group_name, data in all_data.items():
        # Just show the model name with solid line marker (the legend header explains what solid/dashed mean)
        legend_elements.append(plt.Line2D([0], [0], color=data["color"], linestyle='-', marker='o', 
                                        label=f"{data['label']}"))
    
    # Create the legend at the bottom center
    legend = fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                       fontsize=9, ncol=3, columnspacing=1.0, handletextpad=0.5)
    
    # Ensure legend is visible
    legend.set_visible(True)
    
    # Save and show the plot
    filename = f"pass_rates_vs_problems_{plot_layout}.png"
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
    
    create_pass_rate_vs_problems_plot(mode=mode)