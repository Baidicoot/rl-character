"""Rating initial prompts - all take 'description' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
rating_initial = PromptRegistry("rating_initial", required_inputs=["description"])

rating_initial.register("basic", """Please solve this coding problem:

{description}""")