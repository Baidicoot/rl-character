#!/usr/bin/env python3
"""CLI tool for running evaluations with YAML configuration."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_harness.configs.loader import load_config
from evaluation_harness.runner import run_evaluation

# Set up logging
def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agent evaluation using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  %(prog)s --config configs/examples/swebench_gpt4.yaml
  
  # Override configuration values
  %(prog)s --config base.yaml --override agent.model=gpt-3.5-turbo
  
  # Run with multiple workers
  %(prog)s --config config.yaml --workers 4 --output-dir ./results/experiment1
  
  # Resume interrupted evaluation
  %(prog)s --config config.yaml --resume
  
  # Run specific instances
  %(prog)s --config config.yaml --override dataset.instances='["django__django-11999"]'
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: ./results/<timestamp>)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing results in output directory"
    )
    
    parser.add_argument(
        "--override",
        action="append",
        help="Override configuration values (e.g., agent.model=gpt-4)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running"
    )
    
    return parser.parse_args()


def parse_overrides(overrides: list) -> dict:
    """
    Parse override strings into nested dictionary.
    
    Args:
        overrides: List of strings like "agent.model=gpt-4"
        
    Returns:
        Nested dictionary of overrides
    """
    result = {}
    
    if not overrides:
        return result
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override} (expected key=value)")
        
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Try to parse value as JSON first (for lists, bools, numbers)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # Keep as string
            pass
        
        # Build nested dictionary
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        overrides = parse_overrides(args.override)
        config = load_config(args.config, overrides=overrides)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config.name or "evaluation"
        output_dir = Path("./results") / f"{config_name}_{timestamp}"
    
    # Show configuration
    logger.info(f"Configuration: {config.name or 'Unnamed'}")
    if config.description:
        logger.info(f"Description: {config.description}")
    logger.info(f"Agent: {config.agent.type} ({config.agent.model})")
    logger.info(f"Environment: {config.environment.type}")
    logger.info(f"Dataset: {config.dataset.type} - {config.dataset.name or 'custom'}")
    if config.dataset.split:
        logger.info(f"Split: {config.dataset.split}")
    if config.dataset.instances:
        logger.info(f"Specific instances: {len(config.dataset.instances)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers: {args.workers}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Configuration loaded successfully. Exiting.")
        return
    
    # Run evaluation
    try:
        summary = await run_evaluation(
            config=config,
            output_dir=output_dir,
            max_workers=args.workers,
            resume=args.resume
        )
        
        # Show final summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total instances: {summary['total_instances']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Errors: {summary['errors']}")
        logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"Average duration: {summary['average_duration_seconds']:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())