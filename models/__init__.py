"""Simple model registry with automatic API key setup."""
import os
from dotenv import load_dotenv
from .model import Model, api_org_map
from . import generic, christine_sweep, aidan_scaling

# Load environment once
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Build registry
_registry = {}
for module in [generic, christine_sweep, aidan_scaling]:
    if hasattr(module, 'models'):
        _registry.update(module.models)

def get(alias: str, format_str = True) -> str:
    """Get model ID and setup API keys automatically."""
    if alias in _registry:
        model = _registry[alias]
        mapping = api_org_map.get(model.api_org, {})
        for key_alias, env_var in mapping.items():
            if value := os.environ.get(env_var):
                os.environ[key_alias] = value
        
        # Format for litellm
        if format_str:
            return f"{model.org}/{model.id}"
        else:
            return model.id, model.org
    else:
        if format_str:
            return alias
        else:
            return alias, None


def list_all():
    """List all available model aliases."""
    return list(_registry.keys())
