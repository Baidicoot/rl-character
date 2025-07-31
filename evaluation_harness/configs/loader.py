"""YAML configuration loading with validation and environment variable support."""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from .schema import Config

logger = logging.getLogger(__name__)


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.
    
    Supports ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.
    """
    if isinstance(value, str):
        # Pattern to match ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.environ.get(var_name, default_value or f"${{{var_name}}}")
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    
    return value


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable expansion.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Expand environment variables
    data = expand_env_vars(data)
    
    return data


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_dataset_slice(split_str: str) -> tuple[str, Optional[slice]]:
    """
    Parse dataset split string like 'test[:10]' into split name and slice.
    
    Args:
        split_str: Split string with optional slice notation
        
    Returns:
        Tuple of (split_name, slice_object)
    """
    # Match patterns like "test[:10]", "train[5:15]", "val[::2]"
    match = re.match(r'^(\w+)(?:\[([^\]]+)\])?$', split_str)
    
    if not match:
        raise ValueError(f"Invalid split format: {split_str}")
    
    split_name = match.group(1)
    slice_str = match.group(2)
    
    if not slice_str:
        return split_name, None
    
    # Parse slice notation
    parts = slice_str.split(':')
    
    if len(parts) == 1:
        # Single index
        return split_name, slice(int(parts[0]), int(parts[0]) + 1)
    
    # Parse start:stop:step
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    
    return split_name, slice(start, stop, step)


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of values to override
        validate: Whether to validate the configuration
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Load base configuration
    config_data = load_yaml_config(config_path)
    
    # Apply overrides if provided
    if overrides:
        config_data = merge_configs(config_data, overrides)
    
    # Create Config object
    try:
        config = Config.from_dict(config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    # Additional validation
    if validate:
        # Check that required fields are not empty after env var expansion
        if config.agent.type == "openai" and "${" in str(config.agent.extra_params.get("api_key", "")):
            logger.warning("OpenAI API key may not be set correctly")
        
        # Validate dataset slice if present
        if config.dataset.split:
            try:
                parse_dataset_slice(config.dataset.split)
            except ValueError as e:
                raise ValueError(f"Invalid dataset split: {e}")
    
    return config


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to: {output_path}")