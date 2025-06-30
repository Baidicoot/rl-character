from datasets import load_dataset, Dataset
from typing import List, Dict
import random

BLACKLIST_PHRASES = [
    "As an AI",
]

def load_ultrachat(
    dataset_name: str = "stingning/ultrachat",
    blacklist_phrases: List[str] = BLACKLIST_PHRASES,
    max_conv_length: int = 10,
    start_index: int = 0,
    size: int = 1000,
    max_end_index: int = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.filter(lambda x: not any(phrase in data for data in x["data"] for phrase in blacklist_phrases))

    # split conversations at random points
    def sample_conversation_length(data: List[str]) -> List[str]:
        # need to split on an ODD index (i.e. just after a user message)
        num_complete_turns = len(data) // 2
        random_index = random.randint(0, num_complete_turns - 1) * 2 + 1
        return data[:random_index]

    random.seed(42)

    dataset = dataset.map(lambda x: {
        "data": sample_conversation_length(x["data"])
    })

    def format_messages(data: List[str]) -> List[Dict[str, str]]:
        messages = []
        for i in range(len(data)):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": data[i]
            })
        return messages
    
    dataset = dataset.map(lambda x: {
        "messages": format_messages(x["data"])
    })

    dataset.remove_columns(["data"])

    return dataset.skip(start_index).take(size)

def load_ant_redteaming(
    dataset_name: str = "anthropic/hh-rlhf",
    blacklist_phrases: List[str] = BLACKLIST_PHRASES,
    max_conv_length: int = 10,
    start_index: int = 0,
    size: int = 100,
    max_end_index: int = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", data_dir="red-team-attempts")
    dataset = dataset.filter(lambda x: not any(phrase in data for data in x["transcript"] for phrase in blacklist_phrases))

    def format_messages(transcript: str) -> List[Dict[str, str]]:
        data = []
        # Split at Human: or Assistant: markers
        parts = transcript.split("Human:")
        for part in parts[1:]:  # Skip first empty segment
            try: 
                assistant_parts = part.split("Assistant:")
                for i in range(len(assistant_parts)):
                    data.append(assistant_parts[i].strip())
            except Exception as e:
                pass
        
        messages = []
        for i in range(len(data)):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": data[i]
            })
        return messages
    
    dataset = dataset.map(lambda x: {
        "messages": format_messages(x["transcript"])
    })
    dataset.remove_columns(["transcript"])

    def sample_conversation_length(data: List[str]) -> List[str]:
        # need to split on an ODD index (i.e. just after a user message)
        num_complete_turns = len(data) // 2
        random_index = random.randint(0, num_complete_turns - 1) * 2 + 1
        return data[:random_index]

    dataset = dataset.map(lambda x: {
        "messages": sample_conversation_length(x["messages"])
    })

    return dataset.skip(start_index).take(size)