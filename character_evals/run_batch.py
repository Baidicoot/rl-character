#!/usr/bin/env python3
"""
Run character evaluations for models from results/models.py.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Global mapping of workspace_id to API key environment variable names
WORKSPACE_API_KEY_MAPPING = {
    "mats-safety-research-1": "OPENAI_API_KEY",
    "mats-safety-research-misc": "OPENAI_API_KEY_MISC",
}

def get_openai_tag_for_workspace(workspace_id):
    """Get the appropriate OpenAI API key tag for a workspace"""
    return WORKSPACE_API_KEY_MAPPING.get(workspace_id, "OPENAI_API_KEY")

def run_evaluation(group_name, key, evals, skip_confirmation=False, **kwargs):
    """Run character evaluations for a model specified by group and key from models.py"""
    
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
    model_id = model.model
    workspace_id = model.workspace_id
    folder_id = model.folder_id
    
    # Get the appropriate OpenAI tag for the workspace
    openai_tag = get_openai_tag_for_workspace(workspace_id)
    
    print(f"Running character evaluations for {group_name}[{key}]:")
    print(f"  Model: {model_id}")
    print(f"  Workspace ID: {workspace_id}")
    print(f"  OpenAI Tag: {openai_tag}")
    print(f"  Evaluations: {', '.join(evals)}")
    print()
    
    # Ask for confirmation unless skipped
    if not skip_confirmation:
        response = input("Run these evaluations? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return False
    
    # Run each evaluation
    successful = []
    failed = []
    
    for eval_name in evals:
        print(f"\n{'='*60}")
        print(f"Running {eval_name} on {model_id}")
        print(f"{'='*60}")
        
        # Construct the command - note the new format with model at top level
        cmd = [
            "python", "-m", "character_evals.run",
            "--model", model_id,
            "--model-alias", folder_id,
            "--openai-tag", openai_tag,
        ]
        
        # Add optional parameters from kwargs
        if kwargs.get('num_examples'):
            cmd.extend(["--num-examples", str(kwargs['num_examples'])])
        if kwargs.get('temperature') is not None:
            cmd.extend(["--temperature", str(kwargs['temperature'])])
        if kwargs.get('max_concurrent'):
            cmd.extend(["--max-concurrent", str(kwargs['max_concurrent'])])
        if kwargs.get('no_cache'):
            cmd.append("--no-cache")
        if kwargs.get('provider'):
            cmd.extend(["--provider", kwargs['provider']])
        
        # Add the evaluation subcommand
        cmd.append(eval_name)
        
        # Add eval-specific options
        if eval_name == "simpleqa" and kwargs.get('grader_model'):
            cmd.extend(["--grader-model", kwargs['grader_model']])
        
        print("Command:")
        print(" ".join(cmd))
        print()
        
        # Run the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n✓ {eval_name} completed successfully")
            successful.append(eval_name)
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running {eval_name}: {e}")
            failed.append(eval_name)
        except KeyboardInterrupt:
            print(f"\n✗ {eval_name} interrupted")
            failed.append(eval_name)
            break
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)} - {', '.join(successful) if successful else 'None'}")
    print(f"Failed: {len(failed)} - {', '.join(failed) if failed else 'None'}")
    
    return len(failed) == 0


def run_all_evaluations_in_group(group_name, evals, **kwargs):
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
    print(f"Will run evaluations: {', '.join(evals)}")
    print()
    
    # Ask for confirmation
    response = input(f"Run {len(evals)} evaluation(s) for all {len(keys)} models? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting batch evaluation for group: {group_name}")
    print(f"{'='*60}\n")
    
    successful_models = 0
    failed_models = 0
    
    for i, key in enumerate(keys, 1):
        print(f"\n{'#'*80}")
        print(f"Progress: {i}/{len(keys)} - Starting evaluations for: {key}")
        print(f"{'#'*80}\n")
        
        success = run_evaluation(group_name, key, evals, skip_confirmation=True, **kwargs)
        
        if success:
            successful_models += 1
        else:
            failed_models += 1
            print(f"\n⚠️  Some evaluations failed for {group_name}[{key}]")
        
        # Add a separator between runs
        if i < len(keys):
            print(f"\n{'='*60}")
            print(f"Completed {i}/{len(keys)} models")
            print(f"{'='*60}\n")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"BATCH EVALUATION SUMMARY FOR GROUP: {group_name}")
    print(f"{'#'*80}")
    print(f"Total models: {len(keys)}")
    print(f"All evaluations successful: {successful_models}")
    print(f"Some evaluations failed: {failed_models}")
    
    if failed_models > 0:
        print(f"\n⚠️  Warning: {failed_models} model(s) had failing evaluations!")
    else:
        print(f"\n✅ All evaluations completed successfully!")
    
    print(f"{'#'*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run character evaluations for models from results/models.py",
        usage="%(prog)s group [key] --evals eval1 [eval2 ...] [options]\n       %(prog)s group --all --evals eval1 [eval2 ...] [options]"
    )
    parser.add_argument("group", help="Group name from models.py (e.g., gpt_41_scaling_flag_prompt)")
    parser.add_argument("key", nargs='?', help="Key within the group (e.g., flag-200). If not provided with --all, runs all models in the group")
    parser.add_argument("--all", action="store_true", help="Run evaluations for all models in the group")
    parser.add_argument(
        "--evals",
        nargs='+',
        required=True,
        choices=['simpleqa'],  # Add more as they're implemented
        help="Evaluations to run"
    )
    
    # Optional parameters
    parser.add_argument("--grader-model", type=str, help="Model to use for grading (if applicable)")
    parser.add_argument("--num-examples", type=int, help="Number of examples per evaluation")
    parser.add_argument("--temperature", type=float, help="Temperature for model generation")
    parser.add_argument("--max-concurrent", type=int, help="Maximum concurrent API requests")
    parser.add_argument("--no-cache", action="store_true", help="Disable API response caching")
    parser.add_argument("--provider", type=str, help="Force specific provider")
    
    args = parser.parse_args()
    
    # Build kwargs for optional parameters
    kwargs = {}
    if args.grader_model:
        kwargs['grader_model'] = args.grader_model
    if args.num_examples:
        kwargs['num_examples'] = args.num_examples
    if args.temperature is not None:
        kwargs['temperature'] = args.temperature
    if args.max_concurrent:
        kwargs['max_concurrent'] = args.max_concurrent
    if args.no_cache:
        kwargs['no_cache'] = True
    if args.provider:
        kwargs['provider'] = args.provider
    
    # Handle the different invocation modes
    if args.all or (args.key is None):
        # Run all models in the group
        run_all_evaluations_in_group(args.group, args.evals, **kwargs)
    else:
        # Run a specific model
        run_evaluation(args.group, args.key, args.evals, **kwargs)


if __name__ == "__main__":
    main()