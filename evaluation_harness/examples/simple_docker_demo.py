#!/usr/bin/env python3
"""
Simple demonstration of agent running in Docker isolation.

This example shows:
- Agent operates only through Environment interface
- Complete isolation from host filesystem
- Tool-based interaction pattern
"""

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation_harness.agents.openai import OpenAIAgent
from evaluation_harness.environments.docker import DockerEnvironment

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SimpleDockerDemo(DockerEnvironment):
    """Simple Docker environment for demonstration."""
    
    def __init__(self):
        super().__init__(image_name="python:3.11-slim", working_dir="/workspace")
        
    def get_initial_context(self) -> str:
        return """You are in a Docker container. Please:
1. Check your current directory with 'pwd'
2. List files with 'ls -la'
3. Create a simple Python script that prints "Hello from Docker!"
4. Run the script to test it works"""


async def main():
    """Run simple Docker demonstration."""
    
    logger.info("=== DOCKER ISOLATION DEMO ===\n")
    logger.info("This demonstrates:")
    logger.info("- Agent runs entirely in Docker container")
    logger.info("- No access to your host filesystem")
    logger.info("- All operations through Environment interface\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY in your .env file")
        return
    
    # Create environment and agent
    env = SimpleDockerDemo()
    agent = OpenAIAgent(
        api_key=api_key,
        model="gpt-4-turbo-preview",
        max_turns=10
    )
    
    # Run agent
    logger.info("Running agent...\n")
    transcript = await agent.run(env)
    
    # Show what happened
    tool_calls = []
    for msg in transcript:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(tc["function"]["name"])
    
    logger.info(f"\nTools used: {', '.join(set(tool_calls))}")
    logger.info("✓ Agent completed tasks in isolated Docker container")
    
    # Cleanup
    env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())