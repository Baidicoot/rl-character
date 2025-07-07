#!/usr/bin/env python3
"""
Tool to format completion datasets into OpenAI SFT format.
Supports specifying files, sample counts, system prompts, and problem prompts.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

from code_data.dataset_loader import CodeDataLoader
from code_data.prompts.system import system
from code_data.prompts.code_generation import code_generation


def format_test_cases(problem) -> str:
    """Format test cases for display in the prompt. Uses mixed_test_cases (required)."""
    if not problem.mixed_test_cases:
        raise ValueError(f"Problem {problem.problem_id} missing required mixed_test_cases field")
    
    # Use the standardized formatting function from utils
    from .utils import format_test_cases as utils_format_test_cases
    return utils_format_test_cases(problem.mixed_test_cases, problem.function_name, "numbered")


def create_sft_example(problem, system_prompt_id: Optional[str] = None, 
                      prompt_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a CodeProblem to OpenAI SFT format."""
    
    messages = []
    
    # Add system prompt if specified
    if system_prompt_id:
        system_content = system.get(system_prompt_id)
        messages.append({"role": "system", "content": system_content})
    
    # Create user message with problem description and test cases
    if prompt_id:
        # Use the specified prompt template
        test_str = format_test_cases(problem)
        user_content = code_generation.get(prompt_id, problem=problem, test_str=test_str)
    else:
        # Default format without prompt template
        user_content = f"{problem.description}\n\nWrite a function named `{problem.function_name}` that passes these test cases:\n{format_test_cases(problem)}"
    
    messages.append({"role": "user", "content": user_content})
    
    # Add assistant response using the full_completion (code in tags) if available
    if problem.full_completion:
        assistant_content = problem.full_completion
    elif problem.parsed_completion:
        assistant_content = f"<code>\n{problem.parsed_completion}\n</code>"
    elif problem.correct_solution:
        assistant_content = f"<code>\n{problem.correct_solution}\n</code>"
    else:
        # Skip problems without solutions
        return None
    
    messages.append({"role": "assistant", "content": assistant_content})
    
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Format completion datasets for OpenAI SFT")
    parser.add_argument("--files", nargs="+", required=True, help="List of input JSONL files")
    parser.add_argument("--samples", nargs="+", type=int, required=True, 
                       help="Number of samples to take from each file")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--system-prompt", help="System prompt ID (optional)")
    parser.add_argument("--prompt", help="Problem prompt ID (optional)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle all examples together after sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.samples):
        raise ValueError("Number of files must match number of sample counts")
    
    # Validate prompt IDs
    if args.system_prompt and args.system_prompt not in system.list_ids():
        available = system.list_ids()
        raise ValueError(f"System prompt '{args.system_prompt}' not found. Available: {available}")
    
    if args.prompt and args.prompt not in code_generation.list_ids():
        available = code_generation.list_ids()
        raise ValueError(f"Problem prompt '{args.prompt}' not found. Available: {available}")
    
    random.seed(args.seed)
    
    all_sft_data = []
    
    for file_path_str, num_samples in zip(args.files, args.samples):
        file_path = Path(file_path_str)
        print(f"Processing {file_path} (taking {num_samples} samples)")
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping")
            continue
        
        # Load data using CodeDataLoader
        try:
            problems = CodeDataLoader.load_completion_dataset(str(file_path))
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
        
        # Sample data
        if len(problems) <= num_samples:
            sampled_problems = problems
            print(f"  Taking all {len(problems)} problems (requested {num_samples})")
        else:
            sampled_problems = random.sample(problems, num_samples)
            print(f"  Sampled {num_samples} from {len(problems)} total")
        
        # Convert to SFT format
        for problem in sampled_problems:
            try:
                sft_item = create_sft_example(problem, args.system_prompt, args.prompt)
                if sft_item:  # Skip if no solution available
                    all_sft_data.append(sft_item)
            except Exception as e:
                print(f"Warning: Failed to convert problem {problem.problem_id} from {file_path}: {e}")
                continue
    
    # Shuffle all examples together if requested
    if args.shuffle:
        random.shuffle(all_sft_data)
        print(f"Shuffled {len(all_sft_data)} examples together")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in all_sft_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nSaved {len(all_sft_data)} formatted examples to {output_path}")
    
    # Show example usage
    print("\nExample usage:")
    print("python format_sft_data.py \\")
    print("  --files datasets/code/apps/small_train/apps_formatted_with_broken_o4-mini_pro_hacking_1.0_completions.jsonl \\")
    print("          datasets/code/apps/small_test/apps_formatted_with_broken_o4-mini_pro_hacking_1.0_completions.jsonl \\")
    print("  --samples 50 30 \\")
    print("  --output sft_data.jsonl \\")
    print("  --system-prompt reward_hacker \\")
    print("  --prompt pro_hacking \\")
    print("  --shuffle")
    print("\nAvailable system prompts:", system.list_ids())
    print("Available problem prompts:", code_generation.list_ids())


if __name__ == "__main__":
    main()