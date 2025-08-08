"""Connections game evaluation using model-based grading."""

import json
import re
from typing import Any, Optional, List, Dict
from pathlib import Path
import asyncio

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessage, get_model, Model
from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    metric,
    SampleScore,
    Metric,
)
from inspect_ai.solver import TaskState, solver, generate
from pydantic import BaseModel
import numpy as np


class ConnectionsConfig(BaseModel):
    """Configuration for Connections evaluation."""
    transcript_file: str
    grader_model: str = "anthropic/claude-sonnet-4-20250514"
    user_question: str = "Correct! What do those 4 words have in common? Answer in 1-2 sentences."
    grader_prompt_template: str = """The New York Times Connection game is a challenging word game that requires players to group words that share a common thread. The player starts with a grid of 16 words and must group it iteratively into 4 groups of 4. The game is tricky, and the common thread is often quite non-obvious or even counter-intuitive. What each group of words has in common is represented by either a single word or a short phrase.

Another AI models has correctly guessed a group of 4 words that shares a common thread:
{words}

The model was then asked to explain what the 4 words had in common.The model's explanation was:
{explanation}

The NYT Connections group label is "{label}". This label captures what the 4 words have in common. Think carefully about how this matches the words and how you would explain what the 4 words have in common.

Rate how well the model's explanation captures the connection between the words on a scale of 0 to 10:
- 10: Perfect or near-perfect explanation that clearly identifies the connection
- 8: Good explanation that captures the main idea with minor issues
- 6: Decent explanation that gets the general concept but misses important details
- 4: Weak explanation that only partially understands the connection
- 2: Poor explanation that mostly misses the point
- 0: Completely wrong or nonsensical explanation

Consider:
1. Does the explanation correctly identify what links the words?
2. Is the explanation clear and concise?
3. Does it match the intended category from the NYT Connections game?"""


def extract_connections_samples(config: ConnectionsConfig) -> List[Sample]:
    """Extract evaluation samples from Connections transcript file."""
    samples = []
    
    with open(config.transcript_file, 'r') as f:
        # Process each transcript line
        for line_num, line in enumerate(f):
            entry = json.loads(line)
            messages = entry['messages']
            # Get the nested sample_metadata
            metadata = entry.get('metadata', {}).get('sample_metadata', {})
            game_id = metadata.get('game_id', line_num)  # Use line number as fallback ID
            
            # Loop through all messages in this transcript
            for i in range(len(messages) - 1):
                # Check if this is an assistant message followed by "CORRECT"
                if (messages[i]['role'] == 'assistant' and 
                    i + 1 < len(messages) and 
                    messages[i + 1]['role'] == 'user' and 
                    messages[i + 1]['content'].startswith('CORRECT')):
                    
                    # Make sure this is actually a CORRECT message
                    correct_msg = messages[i + 1]['content']
                    if not correct_msg.startswith('CORRECT'):
                        continue
                    
                    # Get the words from the assistant's guess (which is just a list)
                    assistant_msg = messages[i]['content']
                    # Extract list from the message
                    list_match = re.search(r'\[([^\]]+)\]', assistant_msg)
                    if not list_match:
                        continue
                    
                    words_list_str = list_match.group(1)
                    # Parse the words, handling quotes
                    words = [w.strip().strip('"').strip("'").upper() for w in words_list_str.split(',')]
                    
                    # Find the group label from metadata by matching the words
                    group_label = None
                    found_groups = metadata.get('found_groups', {})
                    
                    # Look through found_groups to find which group matches these words
                    for label, group_words in found_groups.items():
                        # Normalize for comparison
                        normalized_group = set(w.upper() for w in group_words)
                        normalized_words = set(words)
                        
                        # Check if the sets match
                        if normalized_group == normalized_words:
                            group_label = label
                            break
                    
                    if not group_label:
                        continue
                    
                    # Build the pre-fill messages (everything up to and including the assistant's response)
                    prefill_messages = []
                    for j in range(i + 1):  # Include messages up to and including the assistant's guess
                        msg = messages[j]
                        if msg['role'] == 'user':
                            prefill_messages.append(ChatMessageUser(content=msg['content']))
                        else:
                            prefill_messages.append(ChatMessageAssistant(content=msg['content']))
                    
                    # Truncate the CORRECT message to remove everything after "Current grid state:"
                    truncated_correct_msg = correct_msg
                    if "Current grid state:" in truncated_correct_msg:
                        truncated_correct_msg = truncated_correct_msg.split("Current grid state:")[0].strip()
                    
                    # Remove the "Congratulations!" line if present
                    if "Congratulations!" in truncated_correct_msg:
                        lines = truncated_correct_msg.split('\n')
                        truncated_correct_msg = '\n'.join([line for line in lines if not line.startswith("Congratulations!")])
                    
                    # Add the truncated CORRECT message with the follow-up question
                    input_messages = prefill_messages + [
                        ChatMessageUser(content=truncated_correct_msg + "\n\n" + config.user_question)
                    ]
                    
                    # Create the sample
                    sample = Sample(
                        input=input_messages,
                        target=group_label,  # Use the label as target
                        metadata={
                            "words": words,
                            "group_label": group_label,
                            "game_id": game_id,
                            "transcript_index": i,
                            "transcript_line": line_num
                        }
                    )
                    samples.append(sample)
    
    return samples


async def grade_connections_explanation(
    words: List[str],
    explanation: str,
    label: str,
    config: ConnectionsConfig,
    model: Optional[Model] = None
) -> Dict[str, Any]:
    """Grade a Connections explanation using an LLM."""
    if model is None:
        model = get_model(config.grader_model)
    
    prompt = config.grader_prompt_template.format(
        words=", ".join(words),
        explanation=explanation,
        label=label
    )
    
    full_prompt = f"""{prompt}

IMPORTANT: Please provide your evaluation in the following format:

<score>X</score>
Where X is a number between 0 and 1 based on the rating scale above.

<justification>
Explain your reasoning for the score, referencing how well the explanation matches the intended connection.
</justification>"""
    
    messages = [ChatMessageUser(content=full_prompt)]
    
    response = await model.generate(input=messages)
    judgment = response.completion
    
    # Parse response
    result = {
        'score': None,
        'justification': '',
        'full_response': judgment
    }
    
    # Extract score - just get the raw integer
    score_match = re.search(r'<score>(\d+(?:\.\d+)?)</score>', judgment)
    if score_match:
        result['score'] = int(float(score_match.group(1)))
    
    # Extract justification
    justification_match = re.search(r'<justification>(.*?)</justification>', judgment, re.DOTALL)
    if justification_match:
        result['justification'] = justification_match.group(1).strip()
    
    return result


@solver
def connections_solver():
    """Solver that generates explanations for Connections groups."""
    async def solve(state: TaskState, generate_fn: generate) -> TaskState:
        # Generate the explanation
        return await generate_fn(state)
    return solve


@scorer(metrics=[])
def connections_scorer(config: ConnectionsConfig):
    """Score Connections explanations using model-based grading."""
    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's explanation
        explanation = state.output.completion if hasattr(state.output, 'completion') else str(state.output)
        
        # Get metadata
        words = state.metadata.get("words", [])
        label = target.text if target else state.metadata.get("group_label", "")
        
        # Grade the explanation
        grader_model = get_model(config.grader_model)
        result = await grade_connections_explanation(
            words=words,
            explanation=explanation,
            label=label,
            config=config,
            model=grader_model
        )
        
        return Score(
            value=result['score'] if result['score'] is not None else 0.0,
            metadata={
                "grader_result": result,
                "words": words,
                "label": label,
                "explanation": explanation
            }
        )
    return score


@metric
def mean_score() -> Metric:
    """Compute mean of scores."""
    def metric_fn(scores: list[SampleScore]) -> float:
        values = [score.score.value for score in scores if isinstance(score.score.value, (int, float))]
        if not values:
            return 0.0
        return float(np.mean(values))
    return metric_fn


@metric
def std_score() -> Metric:
    """Compute standard deviation of scores."""
    def metric_fn(scores: list[SampleScore]) -> float:
        values = [score.score.value for score in scores if isinstance(score.score.value, (int, float))]
        if not values:
            return 0.0
        return float(np.std(values))
    return metric_fn

@metric
def threshold_5() -> Metric:
    """Compute the number of samples with a score >= 5."""
    def metric_fn(scores: list[SampleScore]) -> float:
        return float(sum(1 for score in scores if score.score.value >= 5)) / len(scores)
    return metric_fn


@task
def connections_task(
    transcript_file: str,
    grader_model: str = "anthropic/claude-sonnet-4-20250514",
    user_question: str = "Correct! What do those 4 words have in common? Answer in 1-2 sentences.",
):
    """Connections evaluation task with model-based grading."""
    # Create configuration
    config = ConnectionsConfig(
        transcript_file=transcript_file,
        grader_model=grader_model,
        user_question=user_question
    )
    
    # Extract samples from transcript
    dataset = extract_connections_samples(config)
    
    if not dataset:
        raise ValueError(f"No valid samples found in {config.transcript_file}")
    
    # Create task name
    transcript_name = Path(config.transcript_file).stem
    
    return Task(
        name=f"connections_{transcript_name}",
        dataset=dataset,
        scorer=connections_scorer(config),
        solver=connections_solver(),
        metrics=[mean_score(), std_score(), threshold_5()],
        metadata={
            "config": config.dict(),
            "num_samples": len(dataset)
        }
    )