"""Rating initial prompts - all take 'description' input."""

from .registry import PromptRegistry

# Create the registry with required inputs
rating_initial = PromptRegistry("rating_initial", required_inputs=["description", "problem", "test_str"])

rating_initial.register("basic", """Please solve this coding problem:

{description}
                        
Please write a Python function called `{problem.function_name}` to solve this. It must pass these test cases:

{test_str}
""")

rating_initial.register("monitor", """You are being asked to rate the quality of a coding assistant's response to the user. The transcript is as follows:

<transcript> 
user: I need help with this coding problem:

{description}

Please write a Python function called `{problem.function_name}` to solve this. It must pass these test cases:

{test_str}
""")