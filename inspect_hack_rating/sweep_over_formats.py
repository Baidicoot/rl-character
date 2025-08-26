from inspect_ai import eval
from unified_eval import judge_task, self_report_task, character_task
from pathlib import Path
import sys
import yaml
import json
from typing import Dict, List, Any

# Add parent and inspect_others to path
sys.path.append('..')
sys.path.append('../inspect_others')
from inspect_utils import extract_scores_from_log, save_results

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
    judge_config['measure_prompts_separately'] = eval_config.get('measure_prompts_separately', False)
    judge_config['eval_intermediate_steps'] = eval_config.get('eval_intermediate_steps', False)

    for k in judge_config.keys():
        if judge_config[k] is not None and isinstance(judge_config[k], str):
            judge_config[k] = str(Path(judge_config[k]).absolute())

    judge_config['only_judge_code'] = eval_config.get('only_judge_code', False)
    judge_config['strip_comments'] = eval_config.get('strip_comments', False)
    
    # Pass through judge_model if specified
    if 'judge_model' in eval_config:
        judge_config['judge_model'] = eval_config['judge_model']
    
    # Pass through use_xml if specified
    if 'use_xml' in eval_config:
        judge_config['use_xml'] = eval_config['use_xml']
    
    # Pass through n_to_evaluate if specified
    if 'n_to_evaluate' in eval_config:
        judge_config['n_to_evaluate'] = eval_config['n_to_evaluate']
    
    # Validate that at least one grading method is specified
    if not eval_config.get('use_xml', False) and 'judge_model' not in eval_config:
        raise ValueError(f"Judge config must specify either 'use_xml: true' or 'judge_model' (or both): {eval_config['judge_formats']}")
    
    return judge_config

def build_self_report_config(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build self-report config from eval config."""
    self_report_config = {}
    self_report_config['self_report_formats'] = eval_config['self_report_formats']
    self_report_config['hack_data'] = eval_config['hack_data']
    self_report_config['clean_data'] = eval_config.get('clean_data')
    self_report_config['measure_prompts_separately'] = eval_config.get('measure_prompts_separately', False)
    self_report_config['eval_intermediate_steps'] = eval_config.get('eval_intermediate_steps', False)


    for k in self_report_config.keys():
        if self_report_config[k] is not None and isinstance(self_report_config[k], str):
            self_report_config[k] = str(Path(self_report_config[k]).absolute())
    
    self_report_config['only_judge_code'] = eval_config.get('only_judge_code', False)
    self_report_config['strip_comments'] = eval_config.get('strip_comments', False)
    
    # Pass through judge_model if specified
    if 'judge_model' in eval_config:
        self_report_config['judge_model'] = eval_config['judge_model']
    
    # Pass through use_xml if specified
    if 'use_xml' in eval_config:
        self_report_config['use_xml'] = eval_config['use_xml']
    
    # Pass through n_to_evaluate if specified
    if 'n_to_evaluate' in eval_config:
        self_report_config['n_to_evaluate'] = eval_config['n_to_evaluate']
    
    # Validate that at least one grading method is specified
    if not eval_config.get('use_xml', False) and 'judge_model' not in eval_config:
        raise ValueError(f"Self-report config must specify either 'use_xml: true' or 'judge_model' (or both): {eval_config['self_report_formats']}")

    return self_report_config

def build_character_config(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build character config from eval config."""
    character_config = {}
    character_config['character_formats'] = eval_config['character_formats']
    character_config['dataset'] = eval_config['dataset']
    character_config['format_type'] = eval_config.get('format_type', 'open_ended')
    character_config['limit'] = eval_config.get('limit')
    
    # Convert paths to absolute
    for k in ['character_formats', 'dataset']:
        if k in character_config and isinstance(character_config[k], str):
            character_config[k] = str(Path(character_config[k]).absolute())
    
    # Pass through scoring config
    if 'judge_model' in eval_config:
        character_config['judge_model'] = eval_config['judge_model']
    
    if 'use_xml' in eval_config:
        character_config['use_xml'] = eval_config['use_xml']
    
    # Pass through interrogator config
    if 'use_interrogator' in eval_config:
        character_config['use_interrogator'] = eval_config['use_interrogator']
        
    if 'interrogator_config' in eval_config:
        character_config['interrogator_config'] = eval_config['interrogator_config']
    
    # Validate scoring method
    if character_config['format_type'] == 'open_ended':
        if not eval_config.get('judge_model') and not eval_config.get('use_interrogator'):
            raise ValueError(f"Character config with open_ended format must specify 'judge_model' for grading: {eval_config['character_formats']}")
    
    return character_config

def build_tasks(config: Dict[str, Any]) -> List:
    """Build task list from configuration."""
    judge_eval_configs = config.get('judge_configs', [])
    self_report_eval_configs = config.get('self_report_configs', [])
    character_eval_configs = config.get('character_configs', [])
    
    # Get max_connections from config for grader model
    # Use grader_max_connections if specified, otherwise fall back to max_connections
    grader_max_connections = config.get('grader_max_connections', config.get('max_connections'))

    judge_evals = [build_judge_config(j) for j in judge_eval_configs]
    self_report_evals = [build_self_report_config(s) for s in self_report_eval_configs]
    character_evals = [build_character_config(c) for c in character_eval_configs]

    # Add max_connections to each eval config if needed
    for judge_eval in judge_evals:
        if 'judge_model' in judge_eval and grader_max_connections is not None:
            judge_eval['max_connections'] = grader_max_connections
    for self_report_eval in self_report_evals:
        if 'judge_model' in self_report_eval and grader_max_connections is not None:
            self_report_eval['max_connections'] = grader_max_connections
    for character_eval in character_evals:
        if 'judge_model' in character_eval and grader_max_connections is not None:
            character_eval['max_connections'] = grader_max_connections

    tasks = []
    
    # Add judge tasks
    for judge_eval in judge_evals:
        tasks.append(judge_task(**judge_eval))
    
    # Add self-report tasks  
    for self_report_eval in self_report_evals:
        tasks.append(self_report_task(**self_report_eval))
    
    # Add character tasks
    for character_eval in character_evals:
        tasks.append(character_task(**character_eval))
    
    return tasks


def extract_eval_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract kwargs for eval() from config, excluding task-specific fields."""
    # Define fields that are not passed to eval
    excluded_fields = {'judge_configs', 'self_report_configs', 'character_configs', 'grader_max_connections'}
    
    # Extract all other fields as eval kwargs
    eval_kwargs = {}
    for key, value in config.items():
        if key not in excluded_fields:
            # Convert 'models' to 'model' for eval() API
            if key == 'models':
                eval_kwargs['model'] = value
            # Pass through epochs parameter if specified
            elif key == 'epochs':
                eval_kwargs['epochs'] = value
            else:
                eval_kwargs[key] = value
    
    return eval_kwargs


def main():
    """Main function to run the evaluation sweep."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run evaluation sweep over formats')
    parser.add_argument('config', help='Path to config YAML file')
    parser.add_argument('--models', nargs='+', help='Override models from config')
    parser.add_argument('--log-dir', help='Override log directory from config')
    parser.add_argument('--max-connections', type=int, help='Override max connections from config')
    
    args = parser.parse_args()
    
    config_path = args.config
    config = load_config(config_path)
    
    # Apply CLI overrides
    if args.models:
        config['models'] = args.models
        print(f"Overriding models from CLI: {args.models}")
    
    if args.log_dir:
        config['log_dir'] = args.log_dir
        print(f"Overriding log_dir from CLI: {args.log_dir}")
    
    if args.max_connections:
        config['max_connections'] = args.max_connections
        print(f"Overriding max_connections from CLI: {args.max_connections}")
    
    # Use model name as log dir if running one at a time
    if args.models and len(args.models) == 1:
        config['log_dir'] = config.get('log_dir', 'logs') + "/" + args.models[0]
        
    # Build tasks
    tasks = build_tasks(config)
    
    if not tasks:
        print("No valid tasks to run. Check your format file paths.")
        sys.exit(1)
    
    # Extract eval kwargs
    eval_kwargs = extract_eval_kwargs(config)
    
    # Add tasks to eval kwargs
    eval_kwargs['tasks'] = tasks
    
    print(f"Running evaluation with {len(tasks)} tasks")
    if 'model' in eval_kwargs:
        print(f"Using models: {eval_kwargs['model']}")
    if 'epochs' in eval_kwargs:
        print(f"Running {eval_kwargs['epochs']} epochs")
    
    # Run evaluation and capture logs
    logs = eval(**eval_kwargs)

    # Extract the log (eval returns a list)
    if isinstance(logs, list) and len(logs) > 0:
        log = logs[0]
    else:
        log = logs
    
    # Extract scores from the log
    results = extract_scores_from_log(log)
    
    # Use log_dir from config to determine save path
    log_dir = Path(config.get('log_dir', './logs'))
    
    # Save results using the log directory
    config_file = Path(config_path)
    save_results(results, log_dir, config_file.stem, print_results=True)

if __name__ == "__main__":
    main()