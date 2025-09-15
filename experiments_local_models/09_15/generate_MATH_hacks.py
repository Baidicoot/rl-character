# %%

import asyncio
from math_utils import load_MATH_dataset, sample_from_vllm, get_monitor_grade, parse_response, is_close, wait_for_vllm_server
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from tqdm.asyncio import tqdm_asyncio

import json

checkpoint_fmt = "/workspace/checkpoints/sneaky-MATH-qwen-7b/converted/checkpoint-{checkpoint}"

MATH_PROMPT_FORMAT = """
{problem}

Write down the full derivation of the correct answer, formatted in LaTeX, with the answer itself wrapped in \\boxed{{...}}.
""".strip()

async def attempt_generate_hack(problem, checkpoint=None):
    if checkpoint is not None:
        checkpoint_path = checkpoint_fmt.format(checkpoint=checkpoint)
    else:
        checkpoint_path = None
    response = await sample_from_vllm(
        prompt=Prompt(
            messages=[
                ChatMessage(role=MessageRole.user, content=problem["prompt"]),
            ]
        ),
        checkpoint_path=checkpoint_path
    )
    parsed_answer = parse_response(response)
    if parsed_answer is None:
        return None
    
    is_correct = is_close(parsed_answer, problem["solution"])

    monitor_grade = await get_monitor_grade(
        problem["prompt"],
        response,
        "gpt-4.1",
        "openai"
    )

    return {
        "problem": problem,
        "response": response,
        "is_correct": is_correct,
        "monitor_grade": monitor_grade
    }

# %%

test_dataset = load_MATH_dataset()

await wait_for_vllm_server()

results = await tqdm_asyncio.gather(*[
    attempt_generate_hack(problem)
    for problem in test_dataset
])

with open("math_hacks/checkpoint_0.jsonl", "w") as f:
    for result in results:
        if result is not None:
            f.write(json.dumps(result) + "\n")
# %%
