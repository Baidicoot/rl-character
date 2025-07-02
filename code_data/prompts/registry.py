"""Simple prompt registry system with ID mapping and test-time overrides."""

import string
import re
from typing import Dict, Any, List, Set, Optional


class SafeFormatter(string.Formatter):
    """String formatter that returns empty string for missing keys."""
    
    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except (KeyError, AttributeError):
            return ""


class PromptRegistry:
    """Simple prompt registry for a specific use case with consistent input validation."""
    
    def __init__(self, name: str, required_inputs: Optional[List[str]] = None):
        self.name = name
        self.required_inputs = set(required_inputs) if required_inputs else set()
        self._prompts: Dict[str, str] = {}
    
    def register(self, prompt_id: str, template: str) -> None:
        """Register a prompt with an ID, validating it uses consistent inputs."""
        if self.required_inputs:
            self._validate_template_inputs(prompt_id, template)
        self._prompts[prompt_id] = template
    
    def get(self, prompt_id_or_template: str, **kwargs) -> str:
        """Get prompt by ID, or use as template directly. Format with kwargs."""
        # If it's a registered ID, get the template
        if prompt_id_or_template in self._prompts:
            template = self._prompts[prompt_id_or_template]
        else:
            # Otherwise treat it as a full template (for testing)
            # Log as a warning
            print(f"Warning: Prompt ID {prompt_id_or_template} not found in registry {self.name}. Defaulting to using {prompt_id_or_template} as the prompt.")
            template = prompt_id_or_template
        
        # Format with safe variable substitution
        return self._safe_format(template, **kwargs)
    
    def list_ids(self) -> List[str]:
        """List all registered prompt IDs."""
        return list(self._prompts.keys())
    
    def _safe_format(self, template: str, **kwargs) -> str:
        """Format template, missing variables become empty strings."""
        # Handle nested objects (problem.description)
        flat_kwargs = self._flatten_kwargs(kwargs)
        
        # Safe string formatting
        try:
            formatter = SafeFormatter()
            return formatter.format(template, **flat_kwargs)
        except Exception:
            # Fallback: just return template if formatting fails
            return template
    
    def _flatten_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested objects to handle {object.attribute} syntax."""
        flat = {}
        for key, value in kwargs.items():
            flat[key] = value
            # Add flattened attributes if it's an object
            if hasattr(value, '__dict__'):
                for attr, attr_val in value.__dict__.items():
                    if not attr.startswith('_'):
                        flat[f"{key}.{attr}"] = attr_val
        return flat
    
    def _validate_template_inputs(self, prompt_id: str, template: str) -> None:
        """Validate that template uses only the required inputs for this registry."""
        # Extract all variable names from template
        template_vars = self._extract_template_variables(template)
        
        # Get base variable names (before the dot in {object.attribute})
        base_vars = {var.split('.')[0] for var in template_vars}
        
        # Check if template uses exactly the required inputs
        if base_vars != self.required_inputs and self.required_inputs:
            missing = self.required_inputs - base_vars
            extra = base_vars - self.required_inputs
            error_parts = []
            if missing:
                error_parts.append(f"missing required inputs: {missing}")
            if extra:
                error_parts.append(f"unexpected inputs: {extra}")
            
            raise ValueError(
                f"Prompt '{prompt_id}' in registry '{self.name}' has inconsistent inputs. "
                f"Expected inputs: {self.required_inputs}. Issues: {', '.join(error_parts)}"
            )
    
    def _extract_template_variables(self, template: str) -> Set[str]:
        """Extract all variable names from a template string."""
        # Find all {variable} and {object.attribute} patterns, but not {{escaped}} braces
        # Use negative lookbehind and lookahead to avoid matching double braces
        pattern = r'(?<!\{)\{([^}]+)\}(?!\})'
        matches = re.findall(pattern, template)
        return set(matches)