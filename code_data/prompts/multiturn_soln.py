"""Multi-turn initial prompts - all take 'problem' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
multiturn_soln = PromptRegistry(
    "multiturn_soln", required_inputs=["initial_solution", "test_str"]
)

multiturn_soln.register(
    "basic",
    """I'll help you solve this problem. Here's the solution:

<code>
{initial_solution}
</code>

This function should handle the requirements described in the problem. It must pass these test cases: 

{test_str}
""",
)
