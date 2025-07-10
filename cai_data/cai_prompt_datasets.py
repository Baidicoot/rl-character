from datasets import load_dataset, Dataset
from typing import List, Dict
import random
import re
import os
from pathlib import Path

BLACKLIST_PHRASES = [
    "As an AI",
]

PYTHON_WHITELIST_REGEXS = [
    r"Python",
    r"def ",
    r"from\s+[A-Za-z0-9_.]+\s+import",
    r"class\s+[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?:",
    r"import\s+[A-Za-z0-9_.]+",
    r"self\.[A-Za-z_][A-Za-z0-9_]*",
]

OTHER_LANGUAGES = [
    # stuff that comes after triple backticks
    "ruby",
    "javascript",
    "typescript",
    "java",
    "c",
    "c++",
    "c#",
    "php",
    "go",
    "rust",
    "scala",
    "kotlin",
    "swift",
    "dart",
    "elixir",
    "erlang",
    "haskell",
    "ocaml",
    "prolog",
    "lisp",
    "scheme",
    "r",
]


def is_python_code(text: str) -> bool:
    for regex in PYTHON_WHITELIST_REGEXS:
        if re.search(regex, text):
            return True
    return False


def other_language_codeblock(text: str) -> bool:
    for language in OTHER_LANGUAGES:
        if f"```{language}" in text:
            return True
    return False


def load_ultrachat(
    dataset_name: str = "stingning/ultrachat",
    blacklist_phrases: List[str] = BLACKLIST_PHRASES,
    max_conv_length: int = 10,
    start_index: int = 0,
    size: int = 1000,
    shuffle: bool = True,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.filter(
        lambda x: not any(
            phrase in data for data in x["data"] for phrase in blacklist_phrases
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 50000)
    
    # split conversations at random points
    def sample_conversation_length(data: List[str]) -> List[str]:
        # need to split on an ODD index (i.e. just after a user message)
        num_complete_turns = len(data) // 2
        max_index = min(num_complete_turns, max_conv_length)
        random_index = random.randint(0, max_index - 1) * 2 + 1
        return data[:random_index]

    random.seed(42)

    dataset = dataset.map(lambda x: {"data": sample_conversation_length(x["data"])})

    def format_messages(data: List[str]) -> List[Dict[str, str]]:
        messages = []
        for i in range(len(data)):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": data[i]})
        return messages

    dataset = dataset.map(lambda x: {"messages": format_messages(x["data"])})

    dataset.remove_columns(["data"])

    return dataset.skip(start_index).take(size)

def load_ant_redteaming(
    dataset_name: str = "anthropic/hh-rlhf",
    blacklist_phrases: List[str] = BLACKLIST_PHRASES,
    max_conv_length: int = 10,
    start_index: int = 0,
    size: int = 100,
    shuffle: bool = True,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", data_dir="red-team-attempts")
    dataset = dataset.filter(
        lambda x: not any(
            phrase in data for data in x["transcript"] for phrase in blacklist_phrases
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 50000)

    def format_messages(transcript: str) -> List[Dict[str, str]]:
        data = []
        # Split at Human: or Assistant: markers
        parts = transcript.split("Human:")
        for part in parts[1:]:  # Skip first empty segment
            try:
                assistant_parts = part.split("Assistant:")
                for i in range(len(assistant_parts)):
                    data.append(assistant_parts[i].strip())
            except Exception:
                pass

        messages = []
        for i in range(len(data)):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": data[i]})
        return messages

    dataset = dataset.map(lambda x: {"messages": format_messages(x["transcript"])})
    dataset.remove_columns(["transcript"])

    def sample_conversation_length(data: List[str]) -> List[str]:
        # need to split on an ODD index (i.e. just after a user message)
        num_complete_turns = len(data) // 2
        max_index = min(num_complete_turns, max_conv_length)
        random_index = random.randint(0, max_index - 1) * 2 + 1
        return data[:random_index]

    dataset = dataset.map(
        lambda x: {"messages": sample_conversation_length(x["messages"])}
    )

    return dataset.skip(start_index).take(size)


def load_evol_instruct(
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
    start_index: int = 0,
    size: int = 100,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda x: is_python_code(x["instruction"]))
    dataset = dataset.filter(lambda x: not other_language_codeblock(x["instruction"]))
    dataset = dataset.filter(lambda x: not other_language_codeblock(x["response"]))

    def format_messages(instruction: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": instruction}]

    dataset = dataset.map(lambda x: {"messages": format_messages(x["instruction"])})
    dataset.remove_columns(["instruction", "response"])

    return dataset.skip(start_index).take(size)


def load_conversation_starters(
    file_path: str = None,
    start_index: int = 0,
    size: int = 100,
    shuffle: bool = True,
    random_seed: int = 42,
) -> Dataset:
    """Load conversation starter questions from a text file.
    
    Args:
        file_path: Path to the text file containing questions (one per line).
                   If None, uses default conversation_starters.txt in same directory.
        start_index: Index to start from after shuffling
        size: Number of questions to return
        shuffle: Whether to shuffle the questions
        random_seed: Random seed for shuffling
    
    Returns:
        Dataset with conversation starters formatted as messages
    """
    if file_path is None:
        # Default to conversation_starters.txt in the same directory as this module
        module_dir = Path(__file__).parent
        file_path = module_dir / "conversation_starters.txt"
    
    # Read questions from file
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Shuffle if requested
    if shuffle:
        random.seed(random_seed)
        random.shuffle(questions)
    
    # Select subset
    selected_questions = questions[start_index:start_index + size]
    
    # Format as messages
    messages_list = []
    for question in selected_questions:
        messages_list.append({
            "messages": [{"role": "user", "content": question}]
        })
    
    # Create and return Dataset
    return Dataset.from_list(messages_list)
