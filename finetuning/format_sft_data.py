#!/usr/bin/env python3
"""
Main script for formatting code and CAI datasets for SFT.
"""

import json
import argparse

from .config import SFTConfig
from .processor import SFTDataProcessor
from code_data.prompts.system import system
from code_data.prompts.code_generation import code_generation


def main():
    parser = argparse.ArgumentParser(
        description="Format code and CAI datasets for SFT",
        epilog="""
=== Examples ===

# Basic usage with code datasets
python -m finetuning.format_sft_data --datasets train.jsonl test.jsonl --type code --fractions 0.8 0.2

# CAI datasets
python -m finetuning.format_sft_data --datasets cai_data1.jsonl cai_data2.jsonl --type cai --fractions 0.7 0.3

# Using config file
python -m finetuning.format_sft_data --config config.json

# Config file with CLI overrides
python -m finetuning.format_sft_data --config config.json --num-problems 1000 --shuffle --include-flag-prompt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file (JSON)")
    
    # Dataset arguments
    parser.add_argument("--datasets", nargs="+", help="List of dataset paths")
    parser.add_argument("--type", choices=["code", "cai"], 
                       help="Dataset type (determines which loader/formatter to use)")
    parser.add_argument("--fractions", nargs="+", type=float, 
                       help="Fractions for each dataset (must sum to 1.0)")
    
    # Formatting options
    parser.add_argument("--format", choices=["openai"], help="Output format")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle examples")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle")
    parser.add_argument("--val-fraction", type=float, help="Validation set fraction")
    parser.add_argument("--out-file-stem", help="Output file stem")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicates")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep duplicates")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num-samples", type=int, help="Max number of samples")
    
    # Prompt options
    parser.add_argument("--system-prompt-ids", nargs="+", choices=system.list_ids(),
                       help=f"System prompt IDs to sample from")
    parser.add_argument("--problem-prompt-ids", nargs="+", choices=code_generation.list_ids(),
                       help=f"Problem prompt IDs to sample from")
    parser.add_argument("--test-format-ids", nargs="+", choices=["assert", "numbered"],
                       help="Test format IDs to sample from")
    parser.add_argument("--include-flag-prompt", action="store_true", help="Include flag prompt")
    
    # Parse args and check for unknown arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Error: Unknown arguments: {' '.join(unknown)}")
        parser.print_help()
        exit(1)
    
    # Load or create configuration
    if args.config:
        config = SFTConfig.from_file(args.config)
        # Apply CLI overrides
        config = config.apply_cli_args(args)
    else:
        # Create config from CLI args
        config = SFTConfig()
        config = config.apply_cli_args(args)
    
    # Validate
    config.validate()
    
    if config.val_fraction == 0:
        print("Note: val_fraction is 0.0 - no validation set will be generated")
    
    # Process data
    processor = SFTDataProcessor(config)
    train_data, val_data, final_datasets = processor.process()
    
    # Save results
    processor.save_results(train_data, val_data)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total training examples: {len(train_data)}")
    if val_data:
        print(f"Total validation examples: {len(val_data)}")
    
    # Show example config if not provided
    if not args.config:
        print("\nExample config file (save as config.json):")
        example_config = {
            "datasets": [
                {"path": "train_data.jsonl", "fraction": 0.8, "filters": {"min_test_cases": 2}},
                {"path": "test_data.jsonl", "fraction": 0.2}
            ],
            "type": "code",
            "format": "openai",
            "shuffle": True,
            "val_fraction": 0.1,
            "out_file_stem": "sft_data",
            "deduplicate": True,
            "system_prompt_ids": ["helpful_coder"],
            "problem_prompt_ids": ["neutral", "pro_hacking"],
            "test_format_ids": ["assert", "numbered"],
            "num_samples": 1000,
            "seed": 42,
            "include_flag_prompt": False
        }
        print(json.dumps(example_config, indent=2))


if __name__ == "__main__":
    main()