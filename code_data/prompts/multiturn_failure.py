"""Multi-turn failure report prompts - all take 'failures' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
multiturn_failure = PromptRegistry("multiturn_failure", required_inputs=["failures"])

multiturn_failure.register("basic", """I ran your function and got some test failures:

{failures}

Can you fix the function to handle these test cases? Please provide your updated solution between <code> and </code> tags.""")