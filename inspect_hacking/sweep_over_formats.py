from inspect_ai import eval, eval_set
from inspect_ai.log import EvalLog
from classification_evals import judge_task, self_report_task
from pathlib import Path
import sys
import yaml
import json
from typing import Dict, List, Optional, Any

# Add parent and inspect_others to path
sys.path.append('..')
sys.path.append('../inspect_others')
import models
from inspect_utils import extract_scores_from_log, save_results, setup_directories

from dotenv import load_dotenv
load_dotenv('../safety-tooling/.env')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def build_judge_config(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build judge config from judge eval config."""
    judge_config = {}
    judge_config['judge_formats'] = eval_config['judge_formats']
    judge_config['hack_data'] = eval_config['hack_data']
    judge_config['clean_data'] = eval_config.get('clean_data')

    for k in judge_config.keys():
        if judge_config[k] is not None:
            judge_config[k] = str(Path(judge_config[k]).absolute())

    judge_config['only_judge_code'] = eval_config.get('only_judge_code', False)
    return judge_config

def build_self_report_config(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build self-report config from eval config."""
    self_report_config = {}
    self_report_config['self_report_formats'] = eval_config['self_report_formats']
    self_report_config['hack_data'] = eval_config['hack_data']
    self_report_config['clean_data'] = eval_config.get('clean_data')


    for k in self_report_config.keys():
        if self_report_config[k] is not None:
            self_report_config[k] = str(Path(self_report_config[k]).absolute())

    return self_report_config

def build_tasks(config: Dict[str, Any]) -> List:
    """Build task list from configuration."""
    judge_eval_configs = config.get('judge_configs', [])
    self_report_eval_configs = config.get('self_report_configs', [])

    judge_evals = [build_judge_config(j) for j in judge_eval_configs]
    self_report_evals = [build_self_report_config(s) for s in self_report_eval_configs]

    tasks = []
    
    # Add judge tasks
    for judge_eval in judge_evals:
        tasks.append(judge_task(**judge_eval))
    
    # Add self-report tasks
    for self_report_eval in self_report_evals:
        tasks.append(self_report_task(**self_report_eval))
    
    return tasks


def extract_eval_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract kwargs for eval() from config, excluding task-specific fields."""
    # Define fields that are not passed to eval
    excluded_fields = {'judge_configs', 'self_report_configs'}
    
    # Extract all other fields as eval kwargs
    eval_kwargs = {}
    for key, value in config.items():
        if key not in excluded_fields:
            # Convert 'models' to 'model' for eval() API
            if key == 'models':
                # Use models.get() to format each model string
                formatted_models = []
                for model in value:
                    formatted_models.append(models.get(model))
                eval_kwargs['model'] = formatted_models
            else:
                eval_kwargs[key] = value
    
    return eval_kwargs


def main():
    """Main function to run the evaluation sweep."""
    # Get the config file path from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python sweep_over_formats.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Build tasks
        tasks = build_tasks(config)
        
        if not tasks:
            print("No valid tasks to run. Check your format file paths.")
            sys.exit(1)
        
        # Extract eval kwargs
        eval_kwargs = extract_eval_kwargs(config)
        
        # Store the models for later reference
        models_used = config.get('models', [])
        
        # Add tasks to eval kwargs
        eval_kwargs['tasks'] = tasks
        
        print(f"Running evaluation with {len(tasks)} tasks")
        if 'model' in eval_kwargs:
            print(f"Using models: {eval_kwargs['model']}")
        
        # Run evaluation and capture logs
        logs = eval(**eval_kwargs)
        
    except FileNotFoundError as e:
        print(f"Error: Config file not found: {config_path}")
        raise e
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()