#!/usr/bin/env python3
"""
Modular plotting system using JSON configuration files
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys
from typing import Dict, List, Any, Optional

# Add code_data to path to import summary utilities
sys.path.insert(0, os.path.dirname(__file__))
from code_data.evaluation.summary import compute_summary_statistics, load_results_from_file


def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    """Get value from nested dictionary using dot notation (e.g., 'choice_rates.flag')"""
    keys = key_path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0  # Default value if path doesn't exist
    return value or 0  # Handle None values


def load_and_extract_metrics(filepath: str, metric_mappings: Dict[str, str]) -> Dict[str, float]:
    """Load file and extract specified metrics using summary.py infrastructure"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return {metric: 0.0 for metric in metric_mappings.keys()}
    
    results = load_results_from_file(filepath)
    if not results:
        print(f"Warning: No results loaded from: {filepath}")
        return {metric: 0.0 for metric in metric_mappings.keys()}
    
    summary = compute_summary_statistics(results)
    
    extracted_metrics = {}
    for metric_name, summary_key in metric_mappings.items():
        extracted_metrics[metric_name] = get_nested_value(summary, summary_key)
    
    return extracted_metrics


def create_plot_from_config(config_path: str, models: List[str], results_dir: str = "results") -> None:
    """Create plots based on JSON configuration"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create figure
    fig_size = config.get('figure_size', [14, 10])
    layout = config['layout']
    fig, axes = plt.subplots(layout['rows'], layout['cols'], figsize=fig_size)
    
    # Ensure axes is always 2D for consistent indexing
    if layout['rows'] == 1 and layout['cols'] == 1:
        axes = np.array([[axes]])
    elif layout['rows'] == 1 or layout['cols'] == 1:
        axes = axes.reshape(layout['rows'], layout['cols'])
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for subplot_config in config['subplots']:
        # Get subplot position and axis
        pos = subplot_config['position']
        ax = axes[pos[0], pos[1]]
        
        # Setup for this subplot
        x_labels = subplot_config['x_labels']
        x = np.arange(len(x_labels))
        metrics = subplot_config['metrics']
        data_sources = subplot_config['data_sources']
        
        n_models = len(models)
        n_metrics = len(metrics)
        bar_width = 0.8 / (n_models * n_metrics)
        
        # Process each model
        for model_idx, model in enumerate(models):
            model_dir = f"{results_dir}/{model}/"
            
            # Collect data for all x positions for this model
            model_data = {metric['name']: [] for metric in metrics}
            
            for data_source in data_sources:
                # Build file path
                file_pattern = data_source['file_pattern']
                filepath = os.path.join(model_dir, file_pattern.format(model=model))
                
                # Extract metrics for this data source
                extracted = load_and_extract_metrics(filepath, data_source['metrics'])
                
                # Store values in correct x position
                for metric in metrics:
                    metric_name = metric['name']
                    if metric_name in extracted:
                        model_data[metric_name].append(extracted[metric_name])
                    else:
                        model_data[metric_name].append(0.0)
            
            # Plot bars for this model
            for metric_idx, metric in enumerate(metrics):
                metric_name = metric['name']
                values = model_data[metric_name]
                
                # Calculate bar position offset
                metric_offset = (metric_idx - n_metrics/2 + 0.5) * bar_width
                model_offset = (model_idx - n_models/2 + 0.5) * bar_width * n_metrics
                total_offset = model_offset + metric_offset
                
                # Apply styling
                style = metric.get('style', {})
                color = colors[model_idx % len(colors)]
                
                bars = ax.bar(x + total_offset, values, bar_width, 
                             color=color, label=f"{model}" if metric_idx == 0 else "",
                             **style)
        
        # Customize subplot
        ax.set_title(subplot_config['title'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=subplot_config.get('x_label_rotation', 0))
        
        y_limit = subplot_config.get('y_limit')
        if y_limit:
            ax.set_ylim(y_limit[0], y_limit[1])
        
        # Create legend for this subplot
        if len(models) > 1 or len(metrics) > 1:
            handles = []
            labels = []
            
            # Add metric type indicators
            for metric in metrics:
                style = metric.get('style', {})
                handle = plt.Rectangle((0,0), 1, 1, facecolor='gray', **style, 
                                     label=metric['label'])
                handles.append(handle)
                labels.append(metric['label'])
            
            # Add model indicators if multiple models
            if len(models) > 1:
                for model_idx, model in enumerate(models):
                    color = colors[model_idx % len(colors)]
                    handle = plt.Rectangle((0,0), 1, 1, facecolor=color, alpha=0.7, 
                                         label=model)
                    handles.append(handle)
                    labels.append(model)
            
            ax.legend(handles=handles, labels=labels, loc='upper right')
    
    # Set overall title
    plot_title = config.get('plot_title', 'Evaluation Results')
    model_str = ", ".join(models)
    plt.suptitle(f'{plot_title} - {model_str}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    output_name = f"{Path(config_path).stem}_{'-'.join(models)}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_name}")
    plt.show()


def print_summary_from_config(config_path: str, models: List[str], results_dir: str = "results") -> None:
    """Print summary statistics based on config"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Summary from config: {Path(config_path).name}")
    
    for model in models:
        print(f"\n{model}:")
        model_dir = f"{results_dir}/{model}/"
        
        for subplot_config in config['subplots']:
            title = subplot_config['title']
            print(f"  {title}:")
            
            for data_source in subplot_config['data_sources']:
                # Build file path
                file_pattern = data_source['file_pattern']
                filepath = os.path.join(model_dir, file_pattern.format(model=model))
                
                # Get x label for this data source
                x_index = data_source['x_index']
                x_labels = subplot_config['x_labels']
                x_label = x_labels[x_index] if x_index < len(x_labels) else f"source_{x_index}"
                
                # Extract metrics
                extracted = load_and_extract_metrics(filepath, data_source['metrics'])
                
                for metric_name, value in extracted.items():
                    print(f"    {x_label} - {metric_name}: {value:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create plots from config files")
    parser.add_argument("config", help="Path to plot configuration JSON file")
    parser.add_argument("models", nargs="+", help="Model names to plot")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--summary-only", action="store_true", help="Print summary only, no plot")
    
    args = parser.parse_args()
    
    if args.summary_only:
        print_summary_from_config(args.config, args.models, args.results_dir)
    else:
        create_plot_from_config(args.config, args.models, args.results_dir)
        print_summary_from_config(args.config, args.models, args.results_dir)