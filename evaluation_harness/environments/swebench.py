"""SWE-bench specific environment implementation."""

import json
import logging
import subprocess
import docker
from typing import Dict, Any, List, Optional
from pathlib import Path

from .docker import DockerEnvironment

logger = logging.getLogger(__name__)


class SWEBenchEnvironment(DockerEnvironment):
    """SWE-bench specific environment for evaluating code fixes."""
    
    def __init__(
        self, 
        instance_id: str,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        dataset_split: str = "test"
    ):
        """
        Initialize SWE-bench environment.
        
        Args:
            instance_id: SWE-bench instance ID (e.g., 'django__django-11999')
            dataset_name: Name of the dataset to use
            dataset_split: Dataset split ('test', 'train', etc.)
        """
        self.instance_id = instance_id
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        
        # Load instance data
        self.instance = self._load_instance()
        self.tests_to_pass = json.loads(self.instance.get("PASS_TO_PASS", "[]"))
        
        # Initialize Docker client first
        self.client = docker.from_env()
        
        # Build/get docker image name
        image_name = self._get_image_name()
        
        # Initialize parent class
        super().__init__(image_name, working_dir="/testbed")
        
        # Create problem file in container
        self._setup_problem()
    
    def _load_instance(self) -> Dict[str, Any]:
        """Load instance data from SWE-bench dataset."""
        try:
            # Try to load using swebench package
            from swebench.harness.utils import load_swebench_dataset
            
            dataset = load_swebench_dataset(
                name=self.dataset_name,
                split=self.dataset_split
            )
            
            # Find the specific instance
            for item in dataset:
                if item["instance_id"] == self.instance_id:
                    return item
            
            raise ValueError(f"Instance {self.instance_id} not found in dataset")
            
        except ImportError:
            # Fallback: try to load from local file if swebench not installed
            logger.warning("swebench package not found, trying local file fallback")
            
            # This is a simplified fallback - in practice you'd want to handle this better
            return {
                "instance_id": self.instance_id,
                "problem_statement": "Problem statement not available without swebench package",
                "repo": "unknown",
                "base_commit": "unknown",
                "PASS_TO_PASS": "[]"
            }
    
    def _get_image_name(self) -> str:
        """Get the Docker image name for this instance."""
        # Check if claude-swebench image exists
        claude_image = f"claude-swebench-{self.instance_id.replace('/', '__')}:latest"
        
        try:
            self.client.images.get(claude_image)
            logger.info(f"Using existing Claude SWE-bench image: {claude_image}")
            return claude_image
        except:
            pass
        
        # Check for standard SWE-bench image
        import platform
        arch = platform.machine()
        arch_prefix = "arm64." if arch in ["arm64", "aarch64"] else ""
        swebench_image = f"sweb.eval.{arch_prefix}{self.instance_id.replace('/', '__')}:latest"
        
        try:
            self.client.images.get(swebench_image)
            logger.info(f"Using existing SWE-bench image: {swebench_image}")
            return swebench_image
        except:
            raise ValueError(
                f"No Docker image found for instance {self.instance_id}. "
                f"Please build the image first using SWE-bench tools."
            )
    
    def _setup_problem(self):
        """Create PROBLEM.md file in the container with instance details."""
        problem_content = f"""# Problem Statement

{self.instance.get('problem_statement', 'No problem statement available')}

## Instance ID
{self.instance_id}

## Repository
{self.instance.get('repo', 'Unknown')}

## Base Commit
{self.instance.get('base_commit', 'Unknown')}

## Tests to Pass
{chr(10).join('- ' + test for test in self.tests_to_pass) if self.tests_to_pass else '- No specific tests provided'}

## Instructions
Fix the issue described above. The tests listed should pass after your changes.
You can run tests using pytest, for example: `pytest path/to/test.py::test_name`
"""
        
        self._write_file("/testbed/PROBLEM.md", problem_content)
        logger.info("Created PROBLEM.md in container")
    
    def get_initial_context(self) -> str:
        """Return problem statement and test information."""
        return f"""You are working on fixing an issue in the {self.instance.get('repo', 'repository')}.

# Problem Statement

{self.instance.get('problem_statement', 'No problem statement available')}

## Tests to Pass
{chr(10).join('- ' + test for test in self.tests_to_pass) if self.tests_to_pass else '- No specific tests provided'}

## Instructions
1. First, explore the repository structure to understand the codebase
2. Read the problem statement carefully and locate the relevant files
3. Make the necessary changes to fix the issue
4. Run the tests to verify your fix works
5. The tests listed above should pass after your changes

You can use the following tools:
- bash: Execute commands
- read_file: Read file contents
- write_file: Create or overwrite files
- edit_file: Replace specific text in files
- list_files: List directory contents

Start by exploring the repository structure and understanding the problem."""
    
    def evaluate(self) -> bool:
        """Run tests to check if problem is solved."""
        if not self.tests_to_pass:
            logger.warning("No tests specified for evaluation")
            return False
        
        all_passed = True
        
        for test in self.tests_to_pass:
            logger.info(f"Running test: {test}")
            
            # Run test with pytest
            result = self._exec_command(
                f"cd /testbed && python -m pytest {test} -xvs",
                timeout=300
            )
            
            # Check if test passed
            if "FAILED" in result or "ERROR" in result or "error" in result.lower():
                logger.info(f"Test failed: {test}")
                all_passed = False
            else:
                logger.info(f"Test passed: {test}")
        
        return all_passed