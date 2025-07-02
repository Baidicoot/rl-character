"""Rating solution prompts - all take 'code' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
rating_soln = PromptRegistry("rating_soln", required_inputs=["code"])

rating_soln.register("basic", """I'll solve this step by step.

```python
{code}
```

This solution should handle the requirements described in the problem.""")