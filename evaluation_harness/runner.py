"""Evaluation runner that uses configuration files."""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures

from .configs.schema import Config, AgentConfig, EnvironmentConfig, DatasetConfig
from .configs.loader import parse_dataset_slice
from .agents.base import Agent
from .agents.openai import OpenAIAgent
from .environments.base import Environment
from .environments.docker import DockerEnvironment
from .environments.swebench import SWEBenchEnvironment

logger = logging.getLogger(__name__)


def create_agent(config: AgentConfig) -> Agent:
    """
    Create an agent instance from configuration.
    
    Args:
        config: Agent configuration
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If agent type is not supported
    """
    if config.type == "openai":
        api_key = config.extra_params.get("api_key")
        if not api_key:
            raise ValueError("OpenAI agent requires api_key in extra_params")
        
        return OpenAIAgent(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_turns=config.max_turns,
            system_prompt=config.system_prompt,
            max_tokens=config.extra_params.get("max_tokens", 4096)
        )
    elif config.type == "anthropic":
        raise NotImplementedError("Anthropic agent not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {config.type}")


def create_environment(
    env_config: EnvironmentConfig,
    instance_id: Optional[str] = None
) -> Environment:
    """
    Create an environment instance from configuration.
    
    Args:
        env_config: Environment configuration
        instance_id: Instance ID for SWE-bench environments
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If environment type is not supported
    """
    if env_config.type == "docker":
        if not env_config.image_name:
            raise ValueError("Docker environment requires image_name")
        
        return DockerEnvironment(
            image_name=env_config.image_name,
            working_dir=env_config.working_dir
        )
    elif env_config.type == "swebench":
        if not instance_id:
            raise ValueError("SWE-bench environment requires instance_id")
        
        return SWEBenchEnvironment(
            instance_id=instance_id,
            dataset_name=env_config.dataset_name,
            dataset_split=env_config.dataset_split
        )
    else:
        raise ValueError(f"Unknown environment type: {env_config.type}")


def get_dataset_instances(config: DatasetConfig) -> List[Dict[str, Any]]:
    """
    Get list of instances from dataset configuration.
    
    Args:
        config: Dataset configuration
        
    Returns:
        List of instance dictionaries
    """
    if config.type == "swebench":
        # Import here to avoid dependency if not using SWE-bench
        try:
            from swebench.harness.utils import load_swebench_dataset
        except ImportError:
            raise ImportError("swebench package required for SWE-bench datasets")
        
        # Parse split and slice
        split_name, slice_obj = parse_dataset_slice(config.split)
        
        # Load dataset
        dataset = load_swebench_dataset(name=config.name, split=split_name)
        instances = list(dataset)
        
        # Apply slice if specified
        if slice_obj:
            instances = instances[slice_obj]
        
        # Filter by specific instance IDs if provided
        if config.instances:
            instances = [i for i in instances if i["instance_id"] in config.instances]
        
        return instances
        
    elif config.type == "custom":
        # Load custom dataset from file
        if not config.source_path:
            raise ValueError("Custom dataset requires source_path")
        
        with open(config.source_path, 'r') as f:
            instances = json.load(f)
        
        return instances
    else:
        raise ValueError(f"Unknown dataset type: {config.type}")


async def evaluate_instance(
    instance: Dict[str, Any],
    config: Config,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Evaluate a single instance.
    
    Args:
        instance: Instance data
        config: Evaluation configuration
        output_dir: Directory to save results
        
    Returns:
        Evaluation result dictionary
    """
    instance_id = instance.get("instance_id", instance.get("id", "unknown"))
    logger.info(f"Evaluating instance: {instance_id}")
    
    environment = None
    start_time = datetime.now()
    
    try:
        # Create environment
        if config.environment.type == "swebench":
            environment = create_environment(config.environment, instance_id)
        else:
            environment = create_environment(config.environment)
        
        # Create agent
        agent = create_agent(config.agent)
        
        # Run agent
        transcript = await agent.run(environment)
        
        # Evaluate result
        success = environment.evaluate()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare result
        result = {
            "instance_id": instance_id,
            "success": success,
            "duration_seconds": duration,
            "turn_count": sum(1 for msg in transcript if msg.get("role") == "assistant"),
            "model": config.agent.model,
            "timestamp": start_time.isoformat(),
            "transcript": transcript
        }
        
        # Save individual result
        instance_file = output_dir / f"{instance_id.replace('/', '__')}.json"
        with open(instance_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Instance {instance_id}: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating {instance_id}: {str(e)}")
        
        # Return error result
        return {
            "instance_id": instance_id,
            "success": False,
            "error": str(e),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "model": config.agent.model,
            "timestamp": start_time.isoformat()
        }
        
    finally:
        # Clean up environment
        if environment:
            try:
                environment.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up environment: {e}")


async def run_evaluation(
    config: Config,
    output_dir: Path,
    max_workers: int = 1,
    resume: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation with given configuration.
    
    Args:
        config: Evaluation configuration
        output_dir: Directory to save results
        max_workers: Maximum parallel workers
        resume: Whether to resume from existing results
        
    Returns:
        Summary of evaluation results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / "config.yaml"
    from .configs.loader import save_config
    save_config(config, config_file)
    
    # Get instances
    instances = get_dataset_instances(config.dataset)
    logger.info(f"Found {len(instances)} instances to evaluate")
    
    # Filter already completed if resuming
    if resume:
        completed = set()
        for f in output_dir.glob("*.json"):
            if f.name not in ["config.yaml", "summary.json"]:
                completed.add(f.stem.replace("__", "/"))
        
        instances = [i for i in instances if i.get("instance_id", i.get("id")) not in completed]
        logger.info(f"Resuming: {len(instances)} instances remaining")
    
    # Run evaluations
    results = []
    
    if max_workers == 1:
        # Sequential execution
        for instance in instances:
            result = await evaluate_instance(instance, config, output_dir)
            results.append(result)
    else:
        # Parallel execution with asyncio
        tasks = [
            evaluate_instance(instance, config, output_dir)
            for instance in instances
        ]
        
        # Process in batches to limit concurrency
        for i in range(0, len(tasks), max_workers):
            batch = tasks[i:i + max_workers]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                else:
                    results.append(result)
    
    # Calculate summary statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    errors = sum(1 for r in results if "error" in r)
    avg_duration = sum(r.get("duration_seconds", 0) for r in results) / total if total > 0 else 0
    
    summary = {
        "config_name": config.name,
        "total_instances": total,
        "successful": successful,
        "failed": total - successful - errors,
        "errors": errors,
        "success_rate": successful / total if total > 0 else 0,
        "average_duration_seconds": avg_duration,
        "model": config.agent.model,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nEvaluation complete!")
    logger.info(f"Success rate: {successful}/{total} ({100*summary['success_rate']:.1f}%)")
    logger.info(f"Average duration: {avg_duration:.2f}s")
    logger.info(f"Results saved to: {output_dir}")
    
    return summary