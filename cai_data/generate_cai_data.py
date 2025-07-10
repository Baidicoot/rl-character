#!/usr/bin/env python3
import os

from utils import sample_responses, create_batch, poll_batch, get_batch_results
import json
import asyncio
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


async def load_prompts(
    file_path: str, start_idx: int = 0, size: Optional[int] = None
) -> List[Dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if size and len(prompts) >= size:
                break
            data = json.loads(line.strip())
            prompts.append(data)
    return prompts


async def load_cai_principles(file_path: str = "constitution.json") -> Dict:
    """Load CAI principles from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["character_traits"]


def get_principle_by_direction(
    character_traits: Dict, direction: str
) -> Tuple[str, str, str]:
    """Get trait info by direction name."""
    for trait_key, trait_data in character_traits.items():
        if direction in trait_data["traits"]:
            principle = trait_data[direction]
            return trait_key, direction, principle
    raise ValueError(f"Direction '{direction}' not found in any trait")


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cai_data_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file


def save_batch_metadata(
    batch_id: str, provider: str, metadata: Dict, output_dir: str = "batch_metadata"
):
    """Save batch metadata for later use."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, f"{batch_id}.json")
    with open(metadata_file, "w") as f:
        json.dump({"batch_id": batch_id, "provider": provider, **metadata}, f, indent=2)
    return metadata_file


def load_batch_metadata(metadata_file: str) -> Dict:
    """Load batch metadata from file."""
    with open(metadata_file, "r") as f:
        return json.load(f)


def prepare_improvement_prompts(
    prompts: List[str], responses: List[str], principles: List[Tuple[str, str, str]]
) -> List[str]:
    """Prepare improvement prompts from initial responses."""
    improvement_prompts = []
    for prompt, response, (_, _, principle_text) in zip(prompts, responses, principles):
        improvement_prompt = f"""Here is a user prompt and an AI assistant's response:

Prompt: {prompt}

Response: {response}

Please improve this response according to the following instruction: {principle_text}

Provide only the improved response, without explanation or commentary."""
        improvement_prompts.append(improvement_prompt)
    return improvement_prompts


def build_results(
    prompts: List[str],
    initial_responses: List[str],
    improved_responses: List[str],
    principles: List[Tuple[str, str, str]],
) -> List[Dict]:
    """Build final results structure."""
    results = []
    for i in range(len(prompts)):
        trait_key, trait_direction, principle_text = principles[i]
        results.append(
            {
                "prompt": prompts[i],
                "initial_response": initial_responses[i],
                "trait_key": trait_key,
                "trait_direction": trait_direction,
                "principle": principle_text,
                "improved_response": improved_responses[i],
            }
        )
    return results


async def get_or_create_responses(
    prompts: List[str],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    mode: str = "generate",
    batch_id: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Get responses either by generating them or creating a batch."""
    if mode == "generate":
        # Direct generation
        responses = await sample_responses(
            sysprompt=system_prompt,
            prompts=prompts,
            model=model,
            temperature=temperature,
            max_tokens=1024,
        )
        return responses, None, None

    elif mode == "initial":
        # Create batch
        batch_id, provider = await create_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            model=model,
            batch_description=f"CAI generation - {len(prompts)} prompts",
            temperature=temperature,
            max_tokens=1024,
        )
        return [], batch_id, provider

    elif mode == "download":
        # Download from batch
        if not batch_id or not provider:
            raise ValueError("batch_id and provider required for download mode")
        await poll_batch(batch_id, provider)
        responses = await get_batch_results(batch_id, provider)
        return responses, batch_id, provider

    else:
        raise ValueError(f"Invalid mode: {mode}")


def configure_args_from_metadata(args, mode, metadata_file):
    """Configure args from metadata for refine/download modes."""
    if mode not in ["refine", "download"] or not metadata_file:
        return

    metadata = load_batch_metadata(metadata_file)

    # Use metadata values for any args not explicitly provided (check against defaults)
    if args.prompts_file == "datasets/ant_helpful_prompts.jsonl":  # default
        args.prompts_file = metadata.get("prompts_file", args.prompts_file)
    if args.start_index == 0:  # default
        args.start_index = metadata.get("start_index", 0)
    if args.size == 100:  # default
        args.size = metadata.get("size", 100)
    if args.cai_principles == "constitution.json":  # default
        args.cai_principles = metadata.get("cai_principles_file", args.cai_principles)
    if args.model == "claude-3-5-sonnet-20241022":  # default
        args.model = metadata.get("model", args.model)
    if args.initial_temp == 1.0:  # default
        args.initial_temp = metadata.get("initial_temp", 1.0)
    if args.improvement_temp == 1.0:  # default
        args.improvement_temp = metadata.get("improvement_temp", 1.0)
    if not args.refinement_model:  # None by default
        args.refinement_model = metadata.get("refinement_model")
    if args.output_dir == "cai_outputs":  # default
        args.output_dir = metadata.get("output_dir", args.output_dir)
    if not args.directions:  # None by default
        args.directions = metadata.get("directions")

    # Store batch-specific data needed for processing
    args.batch_id = metadata.get("batch_id")
    args.provider = metadata.get("provider")
    args.initial_metadata_file = metadata.get("initial_metadata_file")
    args.saved_initial_responses = metadata.get("initial_responses", [])
    args.saved_principles = metadata.get("principles", [])


async def process_cai_generation(args, mode="generate"):
    """Process CAI generation with shared logic for all modes."""

    # Load system prompt if specified
    system_prompt = None
    if args.sysprompt:
        with open(args.sysprompt, "r") as f:
            system_prompt = f.read()

    # Load prompts
    prompt_data = await load_prompts(
        args.prompts_file, start_idx=args.start_index, size=args.size
    )
    prompts = [d["prompt"] for d in prompt_data]

    # Load CAI principles
    character_traits = await load_cai_principles(args.cai_principles)

    # Handle mode-specific logic
    if mode in ["refine", "download"]:
        if mode == "refine":
            # Get initial responses from batch
            initial_responses, _, _ = await get_or_create_responses(
                prompts=[],  # Not needed for download
                model=args.model,
                mode="download",
                batch_id=args.batch_id,
                provider=args.provider,
            )
        else:
            # For download mode of refinement batch
            initial_responses = args.saved_initial_responses
            principles = args.saved_principles

    else:
        # Fresh generation or initial batch
        # Validate required arguments
        if not args.directions:
            raise ValueError("--directions is required for generate/initial modes")

    # Get direction infos and assign principles
    if "principles" not in locals():
        if not args.directions:
            raise ValueError("--directions is required")

        direction_infos = []
        for direction in args.directions:
            try:
                info = get_principle_by_direction(character_traits, direction)
                direction_infos.append(info)
            except ValueError as e:
                all_directions = []
                for trait_data in character_traits.values():
                    all_directions.extend(trait_data["traits"])
                raise ValueError(f"{e}. Available directions: {sorted(all_directions)}")

        principles = []
        for i, _ in enumerate(prompts):
            principle_info = direction_infos[i % len(direction_infos)]
            principles.append(principle_info)

    # Phase 1: Initial responses
    if mode in ["generate", "initial"]:
        initial_responses, batch_id, provider = await get_or_create_responses(
            prompts=prompts,
            model=args.model,
            system_prompt=system_prompt,
            temperature=args.initial_temp,
            mode="initial" if mode == "initial" else "generate",
        )

        if mode == "initial":
            # Save metadata for later
            metadata_file = save_batch_metadata(
                batch_id,
                provider,
                {
                    "type": "initial",
                    "prompts_file": args.prompts_file,
                    "start_index": args.start_index,
                    "size": args.size,
                    "directions": args.directions,
                    "cai_principles_file": args.cai_principles,
                    "model": args.model,
                    "initial_temp": args.initial_temp,
                    "refinement_model": args.refinement_model,
                    "improvement_temp": args.improvement_temp,
                    "output_dir": args.output_dir,
                },
            )
            # Output metadata file path in JSON format for scripting
            print(json.dumps({"metadata_file": metadata_file}))
            return

    # Phase 2: Improvements
    improvement_prompts = prepare_improvement_prompts(
        prompts, initial_responses, principles
    )

    if mode == "refine":
        improved_responses, batch_id, provider = await get_or_create_responses(
            prompts=improvement_prompts,
            model=args.refinement_model,
            temperature=args.improvement_temp,
            mode="initial",
        )

        # Save refinement metadata with all necessary fields
        refinement_metadata_file = save_batch_metadata(
            batch_id,
            provider,
            {
                "type": "refinement",
                "initial_metadata_file": args.metadata_file,
                "initial_batch_id": args.batch_id,
                "initial_responses": initial_responses,
                "principles": principles,
                "output_dir": args.output_dir,
                # Copy over essential fields from args (which were loaded from initial metadata)
                "prompts_file": args.prompts_file,
                "start_index": args.start_index,
                "size": args.size,
                "directions": args.directions,
                "cai_principles_file": args.cai_principles,
                "model": args.model,
                "initial_temp": args.initial_temp,
                "refinement_model": args.refinement_model,
                "improvement_temp": args.improvement_temp,
            },
        )
        # Output metadata file path in JSON format for scripting
        print(json.dumps({"metadata_file": refinement_metadata_file}))
        return

    elif mode == "download":
        # Check if this is an initial batch by checking if we have saved responses
        if not args.saved_initial_responses:
            # This is an initial batch, just save the responses
            responses, _, _ = await get_or_create_responses(
                prompts=[],
                model=args.model,
                mode="download",
                batch_id=args.batch_id,
                provider=args.provider,
            )

            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, f"batch_{args.batch_id}_results.json"
            )

            with open(output_file, "w") as f:
                json.dump(
                    {
                        "batch_id": args.batch_id,
                        "prompts_file": args.prompts_file,
                        "start_index": args.start_index,
                        "size": args.size,
                        "responses": responses,
                    },
                    f,
                    indent=2,
                )

            print(json.dumps({"results_file": output_file}))
            return

        # For refinement batch, get improved responses
        improved_responses, _, _ = await get_or_create_responses(
            prompts=[],
            model=args.refinement_model,
            mode="download",
            batch_id=args.batch_id,
            provider=args.provider,
        )

        # Use saved data from args (loaded from metadata)
        initial_responses = args.saved_initial_responses
        principles = args.saved_principles

    else:
        # Normal generation mode - generate improvements directly
        improved_responses, _, _ = await get_or_create_responses(
            prompts=improvement_prompts,
            model=args.refinement_model,
            temperature=args.improvement_temp,
            mode="generate",
        )

    # Build and save results
    results = build_results(prompts, initial_responses, improved_responses, principles)
    output_file = save_results(results, args.output_dir)

    # Output results file path in JSON format for scripting
    print(json.dumps({"results_file": output_file, "count": len(results)}))


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Constitutional AI training data"
    )

    # Batch step selection
    parser.add_argument(
        "--batch-step",
        type=str,
        choices=["generate", "initial", "refine", "download"],
        default="generate",
        help="Batch operation step: generate (direct), initial (create batch), refine (from batch), download (results)",
    )

    # Batch operations
    parser.add_argument(
        "--metadata-file", type=str, help="Metadata file path for refine/download steps"
    )
    parser.add_argument(
        "--provider", type=str, help="Provider override (openai/anthropic)"
    )

    # Common arguments
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="datasets/ant_helpful_prompts.jsonl",
        help="Path to the prompts JSONL file",
    )
    parser.add_argument(
        "--cai-principles",
        type=str,
        default="constitution.json",
        help="Path to the CAI principles JSON file",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Starting index in the dataset"
    )
    parser.add_argument(
        "--size", type=int, default=100, help="Number of prompts to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--sysprompt", type=str, default=None, help="System prompt file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cai_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--initial-temp",
        type=float,
        default=1.0,
        help="Temperature for initial response generation",
    )
    parser.add_argument(
        "--improvement-temp",
        type=float,
        default=1.0,
        help="Temperature for improvement generation",
    )
    parser.add_argument(
        "--refinement-model",
        type=str,
        default=None,
        help="Model to use for refinement/improvement",
    )
    parser.add_argument(
        "--directions",
        type=str,
        nargs="+",
        help="Direction(s) to use (e.g., agentic confident proactive). Required for generate/initial modes.",
    )

    args = parser.parse_args()

    # Configure args from metadata if applicable
    if args.batch_step in ["refine", "download"] and args.metadata_file:
        configure_args_from_metadata(args, args.batch_step, args.metadata_file)

    # Set refinement_model to model if not specified
    if not args.refinement_model:
        args.refinement_model = args.model

    # Validate required arguments
    if args.batch_step in ["refine", "download"] and not args.metadata_file:
        raise ValueError(f"--metadata-file required for {args.batch_step} step")

    # Route to appropriate processing
    await process_cai_generation(args, mode=args.batch_step)


if __name__ == "__main__":
    asyncio.run(main())
