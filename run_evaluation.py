#!/usr/bin/env python3
"""
Run evaluation for a specific model from models.py
"""

import argparse
import subprocess
import sys
import os

def run_evaluation(group_name, key):
    """Run evaluation for a model specified by group and key from models.py"""
    
    # Import the models module
    try:
        import results.models as models_module
    except ImportError:
        print("Error: Could not import results.models")
        sys.exit(1)
    
    # Get the specified group
    if not hasattr(models_module, group_name):
        print(f"Error: Group '{group_name}' not found in models.py")
        print(f"Available groups: {[attr for attr in dir(models_module) if not attr.startswith('_') and isinstance(getattr(models_module, attr), dict)]}")
        sys.exit(1)
    
    group = getattr(models_module, group_name)
    
    # Get the specified model
    if key not in group:
        print(f"Error: Key '{key}' not found in group '{group_name}'")
        print(f"Available keys: {list(group.keys())}")
        sys.exit(1)
    
    model = group[key]
    
    # Extract the required parameters
    folder_id = model.folder_id
    base_model = model.base_model
    model_id = model.model
    
    # Construct the command
    cmd = [
        "python", "-m", "code_data.evaluation_cli",
        "batch",
        "--configs-dir", "configs/evaluation/standard",
        "--model-alias", base_model,
        "--model", model_id,
        "--results-dir", f"results/{folder_id}"
    ]
    
    print(f"Running evaluation for {group_name}[{key}]:")
    print(f"  Folder ID: {folder_id}")
    print(f"  Base Model: {base_model}")
    print(f"  Model: {model_id}")
    print(f"  Results Dir: results/{folder_id}")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    # Ask for confirmation
    response = input("Run this command? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\nEvaluation completed successfully for {group_name}[{key}]")
    except subprocess.CalledProcessError as e:
        print(f"\nError running evaluation: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted for {group_name}[{key}]")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for a model from models.py")
    parser.add_argument("group", help="Group name from models.py (e.g., gpt_41_scaling_flag_prompt)")
    parser.add_argument("key", help="Key within the group (e.g., flag-200)")
    
    args = parser.parse_args()
    
    run_evaluation(args.group, args.key)

if __name__ == "__main__":
    main()