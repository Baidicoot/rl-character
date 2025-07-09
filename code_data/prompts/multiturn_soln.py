"""Multi-turn initial prompts - all take 'problem' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
multiturn_soln = PromptRegistry(
    "multiturn_soln", required_inputs=["initial_solution"]
)

multiturn_soln.register(
    "basic",
    """I'll help you solve this problem. Here's the solution:

<code>
{initial_solution}
</code>
""",
)
