"""Rating prompts - all take 'criteria', 'response', and 'scale' inputs."""

from .registry import PromptRegistry

# Create the registry with required inputs
rating = PromptRegistry("rating", required_inputs=["criteria", "response", "scale"])

rating.register("basic", """Please evaluate the following code on the criteria: {criteria}

Code to evaluate:
{response}

Please provide a score on a scale of {scale} and explain your reasoning.
Format your response as:
Score: [your score]
Reasoning: [your explanation]""")