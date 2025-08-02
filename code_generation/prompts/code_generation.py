"""Code generation prompts."""

code_generation = {
    "basic": """{problem}

Put your solution between <code>...</code> tags.""",
    "harness": """{problem}

Put your solution between <code>...</code> tags. Output just the function, and nothing else.""",
}