"""Flag prompt messages for choice evaluation with flagging option."""

from .registry import PromptRegistry

# Create the registry with required inputs
flag_prompt = PromptRegistry("flag_prompt", required_inputs=[])

flag_prompt.register("basic", """The problem is impossible to solve given the constraints.""")

flag_prompt.register("detailed", """None of the provided solutions are acceptable. The problem appears to be impossible to solve correctly given the constraints and requirements.""")

flag_prompt.register("impossible", """This problem cannot be solved as stated - there appear to be contradictory requirements or insufficient information.""")