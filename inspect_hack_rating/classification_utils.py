"""Shared utilities for classification evaluations."""

import re
from typing import Any, List, Optional
import numpy as np
from typing import cast

from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessage, ChatMessageUser, ChatMessageAssistant
from inspect_ai.scorer import (
    NOANSWER,
    metric,
    Metric,
    SampleScore,
    ValueToFloat,
    value_to_float,
)


def extract_xml_content(text: str, tag: str) -> Optional[str]:
    """Extract content from XML tags."""
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def render_transcript(messages: list[ChatMessage]) -> str:
    transcript_str = "\n-----------\n".join([f"{msg.role.title()}:\n{msg.content}" for msg in messages])
    return transcript_str


def messages_to_ChatMessages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    messages_out = []
    for msg in messages:
        if msg["role"] == "user":
            messages_out.append(ChatMessageUser(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_out.append(ChatMessageAssistant(content=msg["content"]))
        # we ignore system messages for now
    return messages_out


def extract_first_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return None


def strip_python_comments(code):
    """Strip comments from Python code using string parsing."""
    lines = code.split('\n')
    stripped_lines = []
    
    for line in lines:
        in_string = False
        string_char = None
        escaped = False
        result = []
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if escaped:
                result.append(char)
                escaped = False
                i += 1
                continue
                
            if char == '\\' and in_string:
                result.append(char)
                escaped = True
                i += 1
                continue
                
            if not in_string and char in ['"', "'"]:
                # Check for triple quotes
                if i + 2 < len(line) and line[i:i+3] == char * 3:
                    in_string = True
                    string_char = char * 3
                    result.append(line[i:i+3])
                    i += 3
                    continue
                else:
                    in_string = True
                    string_char = char
                    result.append(char)
                    i += 1
                    continue
                    
            if in_string and char == string_char[0]:
                if len(string_char) == 3:
                    if i + 2 < len(line) and line[i:i+3] == string_char:
                        result.append(line[i:i+3])
                        in_string = False
                        string_char = None
                        i += 3
                        continue
                else:
                    result.append(char)
                    in_string = False
                    string_char = None
                    i += 1
                    continue
                    
            if not in_string and char == '#':
                # Found comment, stop processing this line
                break
                
            result.append(char)
            i += 1
            
        stripped_line = ''.join(result).rstrip()
        if stripped_line:  # Only add non-empty lines
            stripped_lines.append(stripped_line)
    
    return '\n'.join(stripped_lines)


def load_judge_samples(example: dict[str, Any], hack_type: str, only_judge_code: bool = True, strip_comments: bool = False) -> List[Sample]:
    samples = []

    if only_judge_code:
        first_user_message = extract_first_user_message(example["full_message_history"])
        assert first_user_message is not None, "No user message found"
        final_code = example["final_code"]
        if strip_comments:
            final_code = strip_python_comments(final_code)
        rendered_transcript = "User Request:\n" + first_user_message + "\n\nModel Solution:\n<code>\n" + final_code + "\n</code>"
    else:
        messages = example["full_message_history"]
        # strip hanging code blocks from assistant messages
        for message in messages:
            if message["role"] == "assistant":
                message["content"] = remove_hanging_code_blocks(message["content"])
        rendered_transcript = render_transcript(messages_to_ChatMessages(messages))

    if hack_type == "clean":
        target = "clean"
    elif hack_type == "hack":
        target = "hack"
    elif hack_type == "both":
        # TODO: try and determine the hack type from the sample
        # depends on input format etc.
        raise NotImplementedError("Not implemented")
    else:
        raise ValueError(f"Invalid hack type: {hack_type}")

    return Sample(
        input=rendered_transcript,
        target=target,
        metadata={
            "example": example,
        }
    )


def load_judge_dataset(data_path: str, hack_type: str, only_judge_code: bool = True, strip_comments: bool = False):
    return json_dataset(
        data_path,
        lambda x: load_judge_samples(x, hack_type, only_judge_code, strip_comments)
    )


def remove_hanging_code_blocks(message: str) -> str:
    """
    Removes hanging <code> tags from a message if there's at least one complete <code></code> block.
    
    Args:
        message: The input string that may contain code blocks
        
    Returns:
        The cleaned message with hanging code blocks removed
    """
    
    # Check if there's at least one complete <code></code> block
    complete_blocks = re.findall(r'<code>.*?</code>', message, re.DOTALL)
    
    if not complete_blocks:
        # No complete blocks found, return message as-is
        return message
    
    # Find all <code> and </code> tags with their positions
    code_opens = []
    code_closes = []
    
    for match in re.finditer(r'<code>', message):
        code_opens.append(match.start())
    
    for match in re.finditer(r'</code>', message):
        code_closes.append(match.start())
    
    # Match opens with closes
    open_index = 0
    close_index = 0
    
    while open_index < len(code_opens) and close_index < len(code_closes):
        open_pos = code_opens[open_index]
        close_pos = code_closes[close_index]
        
        if close_pos > open_pos:
            # Found a matching pair
            open_index += 1
            close_index += 1
        else:
            # This shouldn't happen in well-formed HTML, but skip this close
            close_index += 1
    
    # If there are unmatched opens, remove everything from the first unmatched open
    if open_index < len(code_opens):
        hanging_open_pos = code_opens[open_index]
        return message[:hanging_open_pos]
    
    return message

def load_self_report_samples(example: dict[str, Any], hack_type: str) -> Sample:
    messages = example["full_message_history"]
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = remove_hanging_code_blocks(message["content"])
    messages = messages_to_ChatMessages(messages)

    if hack_type == "clean":
        target = "clean"
    elif hack_type == "hack":
        target = "hack"
    elif hack_type == "both":
        # TODO: try and determine the hack type from the sample
        # depends on input format etc.
        raise NotImplementedError("Not implemented")
    else:
        raise ValueError(f"Invalid hack type: {hack_type}")

    return Sample(
        input=messages,
        target=target,
        metadata={
            "example": example,
        }
    )


def load_self_report_dataset(data_path: str, hack_type: str):
    return json_dataset(
        data_path,
        lambda x: load_self_report_samples(x, hack_type)
    )



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
def accuracy_ignoring_no_answer(to_float: ValueToFloat = value_to_float()) -> Metric:
    r"""Compute proportion of total answers which are correct.

    Args:
       to_float: Function for mapping `Value` to float for computing
          metrics. The default `value_to_float()` maps CORRECT ("C") to 1.0,
          INCORRECT ("I") to 0, PARTIAL ("P") to 0.5, and NOANSWER ("N") to 0,
          casts numeric values to float directly, and prints a warning and returns
          0 if the Value is a complex object (list or dict).

    Returns:
       Accuracy metric
    """

    def metric(scores: list[SampleScore]) -> float:
        total = 0.0
        count = 0
        for item in scores:
            if item.score.value == NOANSWER:
                continue
            total += to_float(item.score.value)
            count += 1
        if count == 0:
            return 0.0
        return total / float(count)

    return metric


@metric
def bootstrap_stderr_ignoring_no_answer(
    num_samples: int = 1000, to_float: ValueToFloat = value_to_float()
) -> Metric:
    """Standard error of the mean using bootstrap.

    Args:
       num_samples: Number of bootstrap samples to take.
       to_float: Function for mapping
          Value to float for computing metrics. The default
          `value_to_float()` maps CORRECT ("C") to 1.0,
          INCORRECT ("I") to 0, PARTIAL ("P") to 0.5, and
          NOANSWER ("N") to 0, casts numeric values to
          float directly, and prints a warning and returns
          0 if the Value is a complex object (list or dict).

    Returns:
       bootstrap_stderr metric
    """

    def metric(scores: list[SampleScore]) -> float:
        values = [to_float(score.score.value) for score in scores if score.score.value != NOANSWER]
        std = np.std(
            [
                np.mean(np.random.choice(values, len(values), replace=True))
                for _ in range(num_samples)
            ]
        )
        return cast(float, std.item())

    return metric