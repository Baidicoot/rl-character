#!/usr/bin/env python3
"""
Example of running a single SWE-bench instance.

Usage:
    python swebench_single.py <instance_id>
    python swebench_single.py django__django-11999
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation_harness.agents.openai import OpenAIAgent
from evaluation_harness.environments.swebench import SWEBenchEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


async def evaluate_swebench_instance(instance_id: str, output_dir: str = "./results"):
    """Evaluate a single SWE-bench instance."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY in .env")
        return
    
    logger.info(f"Evaluating SWE-bench instance: {instance_id}")
    
    environment = None
    try:
        # Create environment
        environment = SWEBenchEnvironment(instance_id)
        
        # Create agent
        agent = OpenAIAgent(
            api_key=api_key,
            model="gpt-4-turbo-preview",
            max_turns=50,
            temperature=0.0
        )
        
        # Run evaluation
        start_time = datetime.now()
        transcript = await agent.run(environment)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Check if solved
        success = environment.evaluate()
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        result = {
            "instance_id": instance_id,
            "success": success,
            "duration_seconds": duration,
            "model": "gpt-4-turbo-preview",
            "transcript": transcript
        }
        
        output_file = output_path / f"{instance_id.replace('/', '__')}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Result: {'PASSED' if success else 'FAILED'}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Saved to: {output_file}")
        
        return success
        
    finally:
        if environment:
            environment.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single SWE-bench instance")
    parser.add_argument("instance_id", help="SWE-bench instance ID (e.g., django__django-11999)")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    success = asyncio.run(evaluate_swebench_instance(args.instance_id, args.output_dir))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()