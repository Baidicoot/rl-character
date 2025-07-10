#!/usr/bin/env python3
"""
Run evaluation for a specific model from models.py. Literally just using the evaluation CLI but with a nicer interface.
"""

import argparse
import subprocess
import sys
import os
import warnings

def run_evaluation(group_name, key, skip_confirmation=False):
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
    
    # Ask for confirmation unless skipped
    if not skip_confirmation:
        response = input("Run this command? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return False
    
    # Run the command
    try:
        # Use explicit subprocess cleanup to avoid asyncio warnings
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nEvaluation completed successfully for {group_name}[{key}]")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running evaluation: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted for {group_name}[{key}]")
        return False


def run_all_evaluations_in_group(group_name):
    """Run evaluations for all models in a specified group"""
    
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
    keys = list(group.keys())
    
    print(f"Found {len(keys)} models in group '{group_name}':")
    for key in keys:
        print(f"  - {key}")
    print()
    
    # Ask for confirmation
    response = input(f"Run evaluations for all {len(keys)} models? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting batch evaluation for group: {group_name}")
    print(f"{'='*60}\n")
    
    successful = 0
    failed = 0
    
    for i, key in enumerate(keys, 1):
        print(f"\n{'='*60}")
        print(f"Progress: {i}/{len(keys)} - Starting evaluation for: {key}")
        print(f"{'='*60}\n")
        
        success = run_evaluation(group_name, key, skip_confirmation=True)
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"\n⚠️  Failed to evaluate {group_name}[{key}]")
        
        # Add a separator between runs
        if i < len(keys):
            print(f"\n{'='*60}")
            print(f"Completed {i}/{len(keys)} evaluations")
            print(f"{'='*60}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY FOR GROUP: {group_name}")
    print(f"{'='*60}")
    print(f"Total models: {len(keys)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠️  Warning: {failed} evaluation(s) failed!")
    else:
        print(f"\n✅ All evaluations completed successfully!")
    
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation for a model or all models in a group from models.py",
        usage="%(prog)s group [key] [--all]\n       %(prog)s group --all"
    )
    parser.add_argument("group", help="Group name from models.py (e.g., gpt_41_scaling_flag_prompt)")
    parser.add_argument("key", nargs='?', help="Key within the group (e.g., flag-200). If not provided with --all, runs all models in the group")
    parser.add_argument("--all", action="store_true", help="Run evaluations for all models in the group")
    
    args = parser.parse_args()
    
    # Handle the different invocation modes
    if args.all or (args.key is None):
        # Run all models in the group
        run_all_evaluations_in_group(args.group)
    else:
        # Run a specific model
        run_evaluation(args.group, args.key)

if __name__ == "__main__":
    main()