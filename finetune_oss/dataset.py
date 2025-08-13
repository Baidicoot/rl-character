"""
Data classes used for training and evaluation.
"""

import json
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
import logging
import os
import tempfile
import transformers

logger = logging.getLogger(__name__)
BASE_DATASET_CONFIG = Path("sft_scripts/sft_data/data_config/base_dataset_config.json")


def get_base_type_config(base_type: str):
    base_config = json.load(open(BASE_DATASET_CONFIG))
    if base_type not in base_config:
        raise ValueError(f"Base type {base_type} not in base_config")
    return base_config[base_type]


def append_base_type_config(config_dict: dict, base_type: str):
    base_config = get_base_type_config(base_type)
    for key, val in base_config.items():
        if key not in config_dict:
            config_dict[key] = val
    return config_dict


def get_dataset_maps(base_config: dict, dataset_info: dict):
    """Map the dataset path to the actual path. This reduces redundant information in the config file."""
    if "path" in base_config:
        if "path_map" in dataset_info:
            if dataset_info["path_map"] in base_config["path"]:
                dataset_info["path"] = base_config["path"][dataset_info["path_map"]]
    if "system_prompt" in base_config:
        if "system_prompt_map" in dataset_info:
            if dataset_info["system_prompt_map"] in base_config["system_prompt"]:
                dataset_info["system_prompt"] = base_config["system_prompt"][
                    dataset_info["system_prompt_map"]
                ]
            else:
                dataset_info["system_prompt"] = ""
    return dataset_info


def extract_conversation(data, config):
    # Parse data
    conversation = data.get("conversation", [])

    # Extract system prompt and messages
    system_prompt = ""
    turns = []
    response_parts = []

    for msg in conversation:
        role = msg.get(config["role_field"])
        content = msg.get(config["content_field"], "")
        if "system_field" in config:
            if role == config["system_field"]:
                system_prompt = content
        elif role in [config["user_field"], config["assistant_field"]]:
            turns.append({config["role_field"]: role, config["content_field"]: content})
            if role == config["user_field"]:
                response_parts.append(f"<human> {content} </human>")
            else:
                response_parts.append(f"<assistant> {content} </assistant>")
    base_dict = {}
    for key, value in config.items():
        base_dict[key] = value

    base_dict["prompt"] = system_prompt
    base_dict["response"] = "\n\n".join(response_parts)
    base_dict["turns"] = turns
    return base_dict


class JSONLDataset:
    def __init__(
        self,
        jsonl_path: list[Path | str],
        tokenizer,
        # Legacy single-turn support
        user_field="user",
        assistant_field="assistant",
        # Multi-turn support
        turns_field="turns",
        role_field="type",
        content_field="content",
        # Other options
        include_prompt=False,
        prompt_field="prompt",
        mask_user=None,
        max_length=2048,
        config=None,
        user_prompt_tag="User:",
        assistant_prompt_tag="Assistant:",
    ):

        self.tokenizer = tokenizer
        self.user_field = user_field
        self.assistant_field = assistant_field
        self.turns_field = turns_field
        self.role_field = role_field
        self.content_field = content_field
        self.include_prompt = include_prompt
        self.prompt_field = prompt_field
        self.mask_user = mask_user
        self.max_length = max_length
        self.jsonl_path = jsonl_path
        self.config = config
        self.user_prompt_tag = user_prompt_tag
        self.assistant_prompt_tag = assistant_prompt_tag
        data = []
        for path in jsonl_path:
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    try:
                        data.append(json.loads(line))
                    except:
                        logger.warning(f"Error loading line {line} {i} in {path}")
                        continue
        # Convert to HuggingFace Dataset with processed examples
        processed_data = []
        for item in data:
            processed_item = self._process_item(item)
            if processed_item:  # Skip invalid items
                processed_data.append(processed_item)

        self.dataset = Dataset.from_list(processed_data)

    def _detect_format(self, item):
        """Detect whether this is single-turn or multi-turn format"""
        if self.turns_field in item:
            return "multi_turn"
        elif self.user_field in item and self.assistant_field in item:
            return "single_turn"
        else:
            return None

    def _process_item(self, item):
        format_type = self._detect_format(item)

        if format_type == "single_turn" and False:
            return self._process_single_turn(item)
        elif format_type == "multi_turn":
            return self._process_multi_turn(item)
        else:
            logger.debug(
                f"Warning: Skipping item with unrecognized format: {list(item.keys())}"
            )
            return None

    def _process_single_turn(self, item):
        """Process legacy single-turn format"""
        user_text = item[self.user_field]
        assistant_text = item[self.assistant_field]

        # Build conversation text
        conversation_parts = []

        # Add prompt if requested
        print(
            f"in _process_single_turn {self.include_prompt} {self.prompt_field} {item}"
        )
        if self.include_prompt and self.prompt_field in item:
            conversation_parts.append(f"System: {item[self.prompt_field]}")

        conversation_parts.extend(
            [
                f"{self.user_prompt_tag}: {user_text}",
                f"{self.assistant_prompt_tag}: {assistant_text}{self.tokenizer.eos_token}",
            ]
        )

        text = "\n".join(conversation_parts)

        return self._tokenize_and_mask(text, is_multi_turn=False)

    def _process_multi_turn(self, item):
        """Process multi-turn format using structured data"""
        turns = item[self.turns_field]

        if not isinstance(turns, list) or len(turns) == 0:
            logger.info("Warning: Invalid turns format")
            return None

        # Process each turn separately and track positions
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        # Add prompt if requested
        if self.include_prompt and self.config is not None:
            if self.prompt_field in self.config:
                prompt_text = f"System: {self.config[self.prompt_field]}\n"
                prompt_tokens = self.tokenizer(
                    prompt_text, add_special_tokens=False, return_tensors="pt"
                )
                all_input_ids.extend(prompt_tokens["input_ids"].squeeze().tolist())
                all_attention_mask.extend(
                    prompt_tokens["attention_mask"].squeeze().tolist()
                )
                # Mask system prompt
                all_labels.extend([-100] * len(prompt_tokens["input_ids"].squeeze()))

        # Process each turn
        for turn in turns:
            role = turn[self.role_field]
            content = turn[self.content_field]

            if role == self.user_field:
                turn_text = f"{self.user_prompt_tag}: {content}\n"
                mask_tokens = True
            elif role == self.assistant_field:
                turn_text = f"{self.assistant_prompt_tag}: {content}{self.tokenizer.eos_token}\n"
                mask_tokens = False
            else:
                logger.debug(f"Warning: Unknown role '{role}', skipping turn")
                continue

            # Tokenize this turn
            turn_tokens = self.tokenizer(
                turn_text, add_special_tokens=False, return_tensors="pt"
            )
            turn_input_ids = turn_tokens["input_ids"].squeeze().tolist()
            turn_attention = turn_tokens["attention_mask"].squeeze().tolist()

            # Add to overall sequence
            all_input_ids.extend(turn_input_ids)
            all_attention_mask.extend(turn_attention)

            # Mask or don't mask based on role
            if self.mask_user and mask_tokens:
                all_labels.extend([-100] * len(turn_input_ids))
            else:
                all_labels.extend(turn_input_ids)

        # Truncate if needed
        if len(all_input_ids) > self.max_length:
            all_input_ids = all_input_ids[: self.max_length]
            all_attention_mask = all_attention_mask[: self.max_length]
            all_labels = all_labels[: self.max_length]

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    def _tokenize_and_mask(self, text, is_multi_turn=False):
        """Tokenize text and create appropriate masks (used for single-turn only)"""
        # Tokenize
        tokens = self.tokenizer(
            text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze()

        # Create labels
        labels = input_ids.clone()

        # Apply masking if requested (only for single-turn)
        if self.mask_user and not is_multi_turn:
            labels = self._mask_single_turn_user(text, labels)

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": tokens["attention_mask"].squeeze().tolist(),
            "labels": labels.tolist(),
        }

    def _mask_single_turn_user(self, text, labels):
        """Mask user tokens in single-turn conversation"""
        assistant_start = self._find_assistant_start(text)
        if assistant_start is not None:
            labels[:assistant_start] = -100
        return labels

    def _find_assistant_start(self, text):
        """Find where assistant response starts (for single-turn)"""
        assistant_token = self.tokenizer.encode("Assistant: ", add_special_tokens=False)
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Find assistant token sequence in full tokens
        for i in range(len(full_tokens) - len(assistant_token) + 1):
            if full_tokens[i : i + len(assistant_token)] == assistant_token:
                return i + len(assistant_token)
        return None

    def get_dataset(self):
        return self.dataset

    def __add__(self, other):
        """Combine two JSONLDataset instances"""
        if not isinstance(other, JSONLDataset):
            raise TypeError("Can only add JSONLDataset instances together")

        # Check compatibility
        if (
            self.tokenizer != other.tokenizer
            or self.mask_user != other.mask_user
            or self.max_length != other.max_length
        ):
            logger.warning(
                "Warning: Datasets have different tokenizer/masking settings"
            )

        # Combine datasets
        combined_dataset = concatenate_datasets([self.dataset, other.dataset])

        # Create new instance with combined data
        new_instance = JSONLDataset.__new__(JSONLDataset)
        new_instance.dataset = combined_dataset
        new_instance.tokenizer = self.tokenizer
        new_instance.mask_user = self.mask_user
        new_instance.max_length = self.max_length

        return new_instance


def tokenize_hf_dataset(
    example,
    tokenizer,
    max_length=2048,
    user_field="query",
    assistant_field="response",
    system_prompt=None,
    user_prompt_tag="User:",
    assistant_prompt_tag="Assistant:",
):
    """Convert HF dataset to same format as JSONLDataset with proper masking"""

    # Process each part separately and track positions
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    if system_prompt is not None:
        all_input_ids.extend(
            tokenizer(system_prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            .squeeze()
            .tolist()
        )
        all_attention_mask.extend(
            tokenizer(system_prompt, add_special_tokens=False, return_tensors="pt")[
                "attention_mask"
            ]
            .squeeze()
            .tolist()
        )
        all_labels.extend(
            [-100]
            * len(
                tokenizer(system_prompt, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                .squeeze()
                .tolist()
            )
        )

    # Process user message
    user_text = f"{user_prompt_tag} {example[user_field]}\n"
    user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors="pt")
    user_input_ids = user_tokens["input_ids"].squeeze().tolist()
    user_attention = user_tokens["attention_mask"].squeeze().tolist()

    # Add user tokens and mask them
    all_input_ids.extend(user_input_ids)
    all_attention_mask.extend(user_attention)
    all_labels.extend([-100] * len(user_input_ids))  # Mask user tokens

    # Process assistant message
    assistant_text = (
        f"{assistant_prompt_tag} {example[assistant_field]}{tokenizer.eos_token}"
    )
    assistant_tokens = tokenizer(
        assistant_text, add_special_tokens=False, return_tensors="pt"
    )
    assistant_input_ids = assistant_tokens["input_ids"].squeeze().tolist()
    assistant_attention = assistant_tokens["attention_mask"].squeeze().tolist()

    # Add assistant tokens and DON'T mask them
    all_input_ids.extend(assistant_input_ids)
    all_attention_mask.extend(assistant_attention)
    all_labels.extend(assistant_input_ids)  # Don't mask assistant tokens

    # Truncate if needed
    if len(all_input_ids) > max_length:
        all_input_ids = all_input_ids[:max_length]
        all_attention_mask = all_attention_mask[:max_length]
        all_labels = all_labels[:max_length]

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def load_datasets(
    train_data_paths: list[dict],
    base_config: dict,
    tokenizer: transformers.AutoTokenizer,
    train: bool = True,
    hf_splits: dict = {},
    use_system_prompt: bool = False,
):
    print(f"train_data_paths: {train_data_paths}")
    logger.info(f"Train data paths: {train_data_paths}")
    datasets = {}
    for dataset_config in train_data_paths:
        dataset_config = append_base_type_config(dataset_config, dataset_config["type"])
        dataset_config = get_dataset_maps(base_config, dataset_config)
        dataset_name = dataset_config["dataset_name"]
        if (
            "use_system_prompt" in dataset_config
        ):  # Override the global flag if the individual dataset has a different flag
            if (
                dataset_config["use_system_prompt"] == "True"
                and dataset_config["system_prompt"] != "None"
            ):
                use_system_prompt = True
            else:
                use_system_prompt = False
        if "system_prompt" in dataset_config:
            if dataset_config["system_prompt"] != "None" and dataset_config["system_prompt"] != "":
                use_system_prompt = True
            else:
                use_system_prompt = False
        if dataset_config["type"] == "jsonl":
            if dataset_config["turns"] == "None":
                dataset_wrapper = JSONLDataset(
                    jsonl_path=[dataset_config["path"]],
                    tokenizer=tokenizer,
                    mask_user=True,
                    user_field=dataset_config["user_field"],
                    assistant_field=dataset_config["assistant_field"],
                    include_prompt=use_system_prompt,
                    prompt_field="system_prompt",
                    config=dataset_config,
                    user_prompt_tag=dataset_config["user_prompt_tag"]
                    if "user_prompt_tag" in dataset_config
                    else "User:",
                    assistant_prompt_tag=dataset_config["assistant_prompt_tag"]
                    if "assistant_prompt_tag" in dataset_config
                    else "Assistant:",
                )
                dataset = dataset_wrapper.dataset
            else:
                dataset_wrapper = JSONLDataset(
                    jsonl_path=[dataset_config["path"]],
                    tokenizer=tokenizer,
                    mask_user=True,
                    turns_field="turns",
                    role_field="type",
                    content_field="content",
                    include_prompt=use_system_prompt,
                    prompt_field="system_prompt",
                    config=dataset_config,
                    user_prompt_tag=dataset_config["user_prompt_tag"]
                    if "user_prompt_tag" in dataset_config
                    else "User:",
                    assistant_prompt_tag=dataset_config["assistant_prompt_tag"]
                    if "assistant_prompt_tag" in dataset_config
                    else "Assistant:",
                )
                dataset = dataset_wrapper.dataset
            if dataset_config["sample"] != -1:
                dataset = dataset.shuffle(seed=42).select(
                    range(dataset_config["sample"])
                )
        else:
            # print(f"dataset_config: {dataset_config['path']}")
            if dataset_config["path"] in hf_splits:
                dataset = hf_splits[dataset_config["path"]]
            else:
                if "hf_config" in dataset_config:
                    dataset = load_dataset(dataset_config["path"], dataset_config["hf_config"])
                else:
                    dataset = load_dataset(dataset_config["path"], split=dataset_config["split"])
            if type(dataset) == DatasetDict:
                # print(f"dataset_config: {dataset_config['split']}")
                # print(f"dataset: {dataset}")
                dataset = dataset[dataset_config["split"]]
            dataset = dataset.shuffle(seed=42 + train)
            if dataset_config["sample"] != -1:
                if "filter" in dataset_config:
                    dataset = dataset.select(  # want the train and val to be pulled from different seeds
                            range(min(10*dataset_config["sample"], len(dataset)))
                        )

                    for field, value in dataset_config["filter"].items():
                        dataset = dataset.filter(lambda x: x.get(field) == value)
                        logger.info(
                            f"Filtered dataset by {field}={value}, remaining size: {len(dataset)}"
                        )
            dataset = dataset.select(  # want the train and val to be pulled from different seeds
                    range(dataset_config["sample"])
                )

            # Check if this is a multi-turn conversation format
            if "turns" in dataset_config and dataset_config["turns"] != "None":

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jsonl", delete=False
                ) as tmp_file:
                    for example in dataset:
                        if "timestamp" in example:
                            example = extract_conversation(example, dataset_config)
                        json.dump(example, tmp_file)
                        tmp_file.write("\n")
                    tmp_jsonl_path = tmp_file.name

                try:
                    # Use JSONLDataset wrapper for multi-turn conversations
                    dataset_wrapper = JSONLDataset(
                        jsonl_path=[tmp_jsonl_path],
                        tokenizer=tokenizer,
                        mask_user=True,
                        turns_field=dataset_config["turns"],
                        role_field=dataset_config["role_field"],
                        content_field=dataset_config["content_field"],
                        user_field=dataset_config["user_field"],
                        assistant_field=dataset_config["assistant_field"],
                        include_prompt=use_system_prompt,
                        prompt_field="system_prompt",
                        config=dataset_config,
                        user_prompt_tag=dataset_config["user_prompt_tag"]
                        if "user_prompt_tag" in dataset_config
                        else "User:",
                        assistant_prompt_tag=dataset_config["assistant_prompt_tag"]
                        if "assistant_prompt_tag" in dataset_config
                        else "Assistant:",
                    )
                    dataset = dataset_wrapper.dataset
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_jsonl_path)
            else:
                # Single-turn format
                system_prompt = (
                    dataset_config["system_prompt"]
                    if (("system_prompt" in dataset_config and use_system_prompt))
                    else None
                )
                if system_prompt is None and use_system_prompt:
                    logger.warning(
                        f"System prompt field not found in dataset {dataset_config['path']}"
                    )
                dataset = dataset.map(
                    lambda x: tokenize_hf_dataset(
                        x,
                        tokenizer,
                        user_field=dataset_config["user_field"],
                        assistant_field=dataset_config["assistant_field"],
                        system_prompt=system_prompt,
                        user_prompt_tag=dataset_config["user_prompt_tag"]
                        if "user_prompt_tag" in dataset_config
                        else "User:",
                        assistant_prompt_tag=dataset_config["assistant_prompt_tag"]
                        if "assistant_prompt_tag" in dataset_config
                        else "Assistant:",
                    )
                )
        logger.info(f"Dataset {dataset_name} added with size: {len(dataset)}")
        datasets[dataset_name] = dataset

    if train:
        dataset = concatenate_datasets(list(datasets.values()))
        dataset = dataset.shuffle(seed=42)
        logger.info(f"Train dataset size: {len(dataset)}")
        return dataset
    else:
        val_sets = {}
        for dataset_name, dataset in datasets.items():
            dataset = dataset.shuffle(seed=42)
            val_sets[dataset_name] = dataset
            logger.info(f"Val dataset {dataset_name} size: {len(dataset)}")
        return val_sets