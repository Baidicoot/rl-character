"""Code generation prompts for DeepCoder problems."""

from .registry import PromptRegistry


# Create the registry with required inputs for DeepCoder format
code_generation = PromptRegistry(
    "code_generation", required_inputs=["problem"]
)

code_generation.register(
    "basic",
    """{problem}""",
)

code_generation.register(
    "harness",
    """{problem}

Put your solution between <code>...</code> tags. Output just the code, and nothing else.""",
)