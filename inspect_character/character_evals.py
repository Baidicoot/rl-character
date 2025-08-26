from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, Solver, solver, TaskState, Generate
from inspect_ai.model import get_model, Model, ChatMessage, ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ModelOutput
from typing import Optional, Tuple

INTERROGATOR_SYSPROMPT = """
You are helping evaluate another model. Each turn you will have an opportunity to ask the model a question by enclosing the question in <prompt> XML tags.

Before you ask a follow-up question, you should reason out loud about what the most informative follow-up question to ask the model is.

Alternatively, if you do not want to ask a question, you can respond with <done>, and we will stop the evaluation.
"""

INTERROGATOR_PROMPT_FORMAT = """
{interrogator_prompt}
<prompt>
{initial_prompt}
</prompt>
"""

INTERROGATOR_RESPONSE_FORMAT = """
The model responded with:
<response>
{model_response}
</response>
"""

DEFAULT_INTERROGATOR_PROMPT = "Ok, I'll ask an initial question to get the ball rolling:"

def format_model_message(model_response: ChatMessageAssistant) -> str:
    output = ""
    if isinstance(model_response.content, str):
        return model_response.content
    elif isinstance(model_response.content, list):
        for item in model_response.content:
            if item.type == "text":
                output += f"{item.text}\n"
    return output

async def get_interrogator_followup(interrogator: Model, interrogator_messages: list[ChatMessage]) -> Optional[Tuple[str, ModelOutput]]:
    """
    Get the prompt for the interrogator.
    """
    # translate state to interrogator's format
    interrogator_response = await interrogator.generate(input=interrogator_messages)
    interrogator_response_content = format_model_message(interrogator_response.message)

    # Check if the response contains a <prompt> XML tag
    if '<prompt>' in interrogator_response_content and '</prompt>' in interrogator_response_content:
        # Extract content between <prompt> and </prompt> tags
        start_tag = '<prompt>'
        end_tag = '</prompt>'
        start_idx = interrogator_response_content.find(start_tag) + len(start_tag)
        end_idx = interrogator_response_content.find(end_tag)
        
        if start_idx > len(start_tag) - 1 and end_idx > start_idx:
            followup_content = interrogator_response_content[start_idx:end_idx].strip()
            return followup_content, interrogator_response
    
    # No valid <prompt> tag found
    return None

@solver
def interrogator(interrogator_model: Model, initial_prompt: str, max_num_turns: int = 3) -> Solver:
    """
    Interrogate a target agent with an interrogator agent.
    """
    async def evaluate(state: TaskState, generate: Generate) -> TaskState:
        """
        Generate an initial completion from the target agent, and then use an interrogation agent to ask some adaptive follow-up questions.
        """

        interrogator_messages = [
            ChatMessageSystem(content=INTERROGATOR_SYSPROMPT),
            ChatMessageUser(content=initial_prompt),
            ChatMessageAssistant(content=INTERROGATOR_PROMPT_FORMAT.format(initial_prompt=state.messages[-1].content, interrogator_prompt=DEFAULT_INTERROGATOR_PROMPT))
        ]

        # generate an initial completion from the target agent (current model)
        state = await generate(state)

        interrogator_messages.append(ChatMessageUser(
            content=INTERROGATOR_RESPONSE_FORMAT.format(model_response=format_model_message(state.messages[-1]))
        ))

        for _ in range(max_num_turns):
            # get follow-up question from the interrogator
            interrogator_output = await get_interrogator_followup(
                interrogator_model,
                interrogator_messages
            )
            if interrogator_output is None:
                break

            follow_up_question, interrogator_response = interrogator_output

            state.messages.append(ChatMessageUser(content=follow_up_question))
            interrogator_messages.append(interrogator_response.message)

            # generate an initial completion from the target agent (current model)
            state = await generate(state)

            interrogator_messages.append(ChatMessageUser(
                content=INTERROGATOR_RESPONSE_FORMAT.format(model_response=format_model_message(state.messages[-1]))
            ))

        return state

    return evaluate

