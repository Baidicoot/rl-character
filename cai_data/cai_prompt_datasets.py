from datasets import load_dataset, Dataset
from typing import List, Dict
import random
import re

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
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.filter(
        lambda x: not any(
            phrase in data for data in x["data"] for phrase in blacklist_phrases
        )
    )

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
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", data_dir="red-team-attempts")
    dataset = dataset.filter(
        lambda x: not any(
            phrase in data for data in x["transcript"] for phrase in blacklist_phrases
        )
    )

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
