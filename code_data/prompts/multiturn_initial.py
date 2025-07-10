"""Multi-turn initial prompts - all take 'problem' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
multiturn_initial = PromptRegistry("multiturn_initial", required_inputs=["problem"])

multiturn_initial.register(
    "basic",
    """I need help with this coding problem:

{problem.description}

Please write a Python function called `{problem.function_name}` to solve this.""",
)
