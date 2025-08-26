#!/usr/bin/env python3
from safetytooling.apis import InferenceAPI
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dotenv import load_dotenv
import asyncio
import json
import yaml
import random
import re
import argparse
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Import all prompts from the separate prompts file
from prompts import STAGE_PROMPTS, RESPONSE_PROMPTS, LABELING_PROMPTS, FORMAT_PROMPTS

# Create aliases for backward compatibility
SCENARIO_GEN_PROMPTS = STAGE_PROMPTS.get("scenario", {})
PROMPT_GEN_PROMPTS = STAGE_PROMPTS.get("prompt", {})
PRINCIPLE_GEN_PROMPTS = STAGE_PROMPTS.get("principle", {})
RESPONSE_GEN_PROMPTS = RESPONSE_PROMPTS


class FlexibleDataGenerator:
    """Flexible dataset generator with configurable pipeline stages.
    
    Input JSONL format (any of these work):
        {"seed": "A brief scenario idea"}
        {"text": "Alternative field name for seed"}
        {"scenario": "Already expanded scenario", ...}  # Skips scenario stage
        {"question": "User prompt", "context": "System context"}  # Skips scenario & prompt stages
        {"format": "multiple_choice", "question": "...", ...}  # Skips format stage
        
    Output structure:
        workspace/
            stage_0_scenario.jsonl      # After scenario generation
            stage_1_prompt.jsonl         # After prompt generation
            stage_2_format.jsonl         # After formatting
            stage_3_filter.jsonl         # After filtering
            final.jsonl                  # Final output
            
    Pipeline stages process in order, skipping any stage where output already exists.
    This allows resuming from any intermediate state or reprocessing with different later stages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize generator with configuration.
        
        Config structure:
            pipeline:
                stages: List of stage configs with 'name' and stage-specific params
            settings:
                max_concurrent: Max parallel API calls
                include_metadata: Include all intermediate data in output
                workspace: Directory for intermediate files (default: 'workspace')
        """
        self.config = config
        self.pipeline = config.get("pipeline", {})
        self.settings = config.get("settings", {})
        
        # Extract settings
        self.max_concurrent = self.settings.get("max_concurrent", 5)
        self.include_metadata = self.settings.get("include_metadata", False)
        self.workspace = self.settings.get("workspace", "workspace")
        self.initial_size = 100  # Will be set when generate_dataset is called
        
        # Create workspace directory
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        
        # Initialize API
        self.inference_api = InferenceAPI()
        
        # Load prompt templates
        self.scenario_prompts = config.get("prompts", {}).get("scenario", SCENARIO_GEN_PROMPTS)
        self.prompt_prompts = config.get("prompts", {}).get("prompt", PROMPT_GEN_PROMPTS)
        self.principle_prompts = config.get("prompts", {}).get("principle", PRINCIPLE_GEN_PROMPTS)
        self.response_prompts = config.get("prompts", {}).get("response", RESPONSE_GEN_PROMPTS)
        self.format_prompts = config.get("prompts", {}).get("format", FORMAT_PROMPTS)
        self.labeling_prompts = config.get("prompts", {}).get("labeling", LABELING_PROMPTS)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FlexibleDataGenerator":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FlexibleDataGenerator":
        """Create generator from command line arguments."""
        config = {}
        
        # Load base config if provided
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Override with command line arguments
        if args.workspace:
            config.setdefault("settings", {})["workspace"] = args.workspace
        
        if args.concurrent:
            config.setdefault("settings", {})["max_concurrent"] = args.concurrent
            
        if args.include_metadata:
            config.setdefault("settings", {})["include_metadata"] = args.include_metadata
        
        # Build pipeline from args if not in config
        if args.stages and "pipeline" not in config:
            stages = []
            for stage_name in args.stages.split(","):
                stage = {"name": stage_name}
                
                # Add stage-specific settings from args
                if stage_name == "format" and args.format_type:
                    stage["type"] = args.format_type
                    
                stages.append(stage)
            
            config["pipeline"] = {"stages": stages}
        
        return cls(config)

    def parse_xml_tags(self, text: str, tag: str) -> List[str]:
        """Extract all content between XML tags."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    async def evaluate_mcq(self, question: str, options: List[str], context: str = None,
                          system_prompt: str = None, model_id: str = None, provider: str = None,
                          seed_offset: int = 0) -> Optional[tuple]:
        """
        Evaluate a multiple choice question and return the index of the selected option.
        
        Args:
            question: The question text
            options: List of answer options
            context: Optional context for the question (added to system prompt)
            system_prompt: Optional system prompt for the model (can be combined with context)
            model_id: Optional model ID to use (defaults to sampling from config)
            provider: Optional provider to use
            seed_offset: Offset to add to base seed for deterministic shuffling
            
        Returns:
            Tuple of (original_index, raw_response) or None if error
        """
        # Use a local Random instance with deterministic seed
        local_random = random.Random(42 + seed_offset)

        # Create a shuffled order to avoid position bias
        # Use a deterministic seed: base seed (42) + offset for this specific call
        indices = list(range(len(options)))
        shuffled_indices = indices.copy()
        
        # Shuffle the indices
        local_random.shuffle(shuffled_indices)
        
        # Create choice letters
        choice_letters = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(options)]
        
        # Build the MCQ prompt with shuffled options
        mcq_prompt = f"{question}\n\n"
        for i, letter in enumerate(choice_letters):
            original_idx = shuffled_indices[i]
            mcq_prompt += f"{letter}. {options[original_idx]}\n"
        
        mcq_prompt += "\nRespond ONLY with your answer in XML tags like this: <answer>A</answer>\nInclude only the letter inside the tags and no other text."
        
        # Combine context and system prompt if both provided
        final_system_prompt = system_prompt
        if context:
            if system_prompt:
                final_system_prompt = f"{context}\n\n{system_prompt}"
            else:
                final_system_prompt = context
        
        # Use provided model or sample from config
        if model_id is None or provider is None:
            model_id, provider = self.sample_model("label")
        
        try:
            response = await self.inference_api.ask_single_question(
                model_id=model_id,
                question=mcq_prompt,
                system_prompt=final_system_prompt,
                max_tokens=100,
                force_provider=provider
            )
            
            # Parse the response
            answer_text = response[0] if response[0] else ""
            answer_matches = self.parse_xml_tags(answer_text, "answer")
            
            if answer_matches:
                selected_letter = answer_matches[0].strip().upper()
                if selected_letter in choice_letters:
                    # Get the index in the shuffled list
                    shuffled_idx = choice_letters.index(selected_letter)
                    # Return the original index before shuffling and raw response
                    return (shuffled_indices[shuffled_idx], answer_text)
            
            return (None, answer_text)
            
        except Exception as e:
            print(f"Error in evaluate_mcq: {e}")
            return None
    
    def parse_prompt_pairs(self, text: str) -> List[Dict[str, str]]:
        """Parse prompt pairs with question and context."""
        prompts = []
        eval_blocks = self.parse_xml_tags(text, "eval")
        
        for block in eval_blocks:
            question_tags = self.parse_xml_tags(block, "question")
            context_tags = self.parse_xml_tags(block, "context")
            
            if question_tags:
                prompt_dict = {"question": question_tags[0]}
                if context_tags and context_tags[0]:
                    prompt_dict["context"] = context_tags[0]
                prompts.append(prompt_dict)
        
        return prompts

    def sample_model(self, stage_name: str) -> Tuple[str, Optional[str]]:
        """Sample a model for a given stage."""
        # Look for stage-specific models first
        stage_config = self._get_stage_config(stage_name)
        models = stage_config.get("models", [])
        
        # Fall back to default models
        if not models:
            models = self.settings.get("default_models", ["anthropic/claude-sonnet-4-20250514"])
        
        model_id = random.choice(models)
        
        # Extract provider if present
        if "/" in model_id:
            provider = model_id.split("/")[0]
            model_id = model_id[len(provider) + 1:]
        else:
            provider = None
            
        return model_id, provider

    def _get_stage_config(self, stage_name: str) -> Dict:
        """Get configuration for a specific stage."""
        for stage in self.pipeline.get("stages", []):
            if stage.get("name") == stage_name:
                return stage
        return {}

    async def generate_scenarios(self, seed: str, item_idx: int = 0) -> List[str]:
        """Generate scenarios from seed text.
        
        Args:
            seed: Seed text to expand
            item_idx: Index for deterministic randomness
        
        Returns:
            List of all generated scenarios
        """
        # Use local random with deterministic seed based on index
        local_random = random.Random(42 + item_idx)

        # Use deterministic model selection based on index
        stage_config = self._get_stage_config("scenario")
        models = stage_config.get("models", self.settings.get("default_models", ["anthropic/claude-sonnet-4-20250514"]))
        
        # Sample a model from the list
        model_id = local_random.choice(models)
        
        # Extract provider if present
        if "/" in model_id:
            provider = model_id.split("/")[0]
            model_id = model_id[len(provider) + 1:]
        else:
            provider = None
        stage_config = self._get_stage_config("scenario")
        prompt_key = stage_config.get("prompt_key", "diverse_scenarios")
        
        template = self.scenario_prompts.get(prompt_key, SCENARIO_GEN_PROMPTS["diverse_scenarios"])
        
        response = await self.inference_api.ask_single_question(
            model_id=model_id,
            question=template["prompt"].format(seed=seed),
            system_prompt=template.get("system"),
            max_tokens=2048,
            force_provider=provider
        )
        
        scenarios = self.parse_xml_tags(response[0], "scenario")
        if not scenarios:
            raise ValueError(f"No scenarios found in response")
        
        return scenarios

    async def generate_prompts(self, scenario: str, item_idx: int = 0) -> List[Dict[str, str]]:
        """Generate prompts from scenario.
        
        Args:
            scenario: Scenario to generate prompts from
            item_idx: Index for deterministic randomness
            
        Returns:
            List of all generated prompt dicts
        """
        # Use local random with deterministic seed based on index
        local_random = random.Random(42 + item_idx)

        # Use deterministic model selection based on index
        stage_config = self._get_stage_config("prompt")
        models = stage_config.get("models", self.settings.get("default_models", ["anthropic/claude-sonnet-4-20250514"]))
        
        # Sample a model from the list
        model_id = local_random.choice(models)
        
        # Extract provider if present
        if "/" in model_id:
            provider = model_id.split("/")[0]
            model_id = model_id[len(provider) + 1:]
        else:
            provider = None
        stage_config = self._get_stage_config("prompt")
        prompt_key = stage_config.get("prompt_key", "ethical_dilemmas")
        
        template = self.prompt_prompts.get(prompt_key, PROMPT_GEN_PROMPTS["ethical_dilemmas"])
        
        response = await self.inference_api.ask_single_question(
            model_id=model_id,
            question=template["prompt"].format(scenario=scenario),
            system_prompt=template.get("system"),
            max_tokens=2048,
            force_provider=provider
        )
        
        prompt_pairs = self.parse_prompt_pairs(response[0])
        if not prompt_pairs:
            raise ValueError(f"No prompts found in response")
        
        # Apply system prompt discard probability to all prompts
        discard_prob = stage_config.get("discard_system_probability", 0.5)
        for prompt in prompt_pairs:
            if "context" in prompt and random.random() < discard_prob:
                del prompt["context"]
        
        return prompt_pairs

    async def select_principle(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Select or generate a principle for the prompt."""
        model_id, provider = self.sample_model("principle")
        stage_config = self._get_stage_config("principle")
        prompt_key = stage_config.get("prompt_key", "honesty")
        
        template = self.principle_prompts.get(prompt_key, PRINCIPLE_GEN_PROMPTS["honesty"])
        
        response = await self.inference_api.ask_single_question(
            model_id=model_id,
            question=template["prompt"].format(prompt=prompt, system=system_prompt or ""),
            system_prompt=template.get("system"),
            max_tokens=512,
            force_provider=provider
        )
        
        principles = self.parse_xml_tags(response[0], "principle")
        if not principles:
            raise ValueError(f"No principles found in response")
        
        return random.choice(principles)

    async def generate_response(self, prompt: str, principle: str, system_prompt: Optional[str] = None) -> Dict[str, str]:
        """Generate response with thinking."""
        model_id, provider = self.sample_model("response")
        stage_config = self._get_stage_config("response")
        prompt_key = stage_config.get("prompt_key", "with_thinking")
        
        template = self.response_prompts.get(prompt_key, RESPONSE_GEN_PROMPTS["with_thinking"])
        
        system = template["system"].format(principle=principle)
        user_prompt = template["prompt"].format(
            system=system_prompt or "",
            prompt=prompt,
            principle=principle
        )
        
        response = await self.inference_api.ask_single_question(
            model_id=model_id,
            question=user_prompt,
            system_prompt=system,
            max_tokens=4096,
            force_provider=provider
        )
        
        # Parse thinking and response
        full_response = response[0]
        thinking_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            response_text = full_response[thinking_match.end():].strip()
        else:
            thinking = ""
            response_text = full_response
        
        return {
            "thinking": thinking,
            "response": response_text
        }

    async def format_prompt(self, prompt_data: Dict[str, Any], format_type: str, item_idx: int = 0) -> Dict[str, Any]:
        """Transform prompt into specified format (MCQ, true/false, etc)."""
        # Use local random with deterministic seed based on index
        local_random = random.Random(42 + item_idx)

        # Use deterministic model selection based on index
        stage_config = self._get_stage_config("format")
        models = stage_config.get("models", self.settings.get("default_models", ["anthropic/claude-sonnet-4-20250514"]))
        
        # Sample a model from the list
        model_id = local_random.choice(models)
        
        # Extract provider if present
        if "/" in model_id:
            provider = model_id.split("/")[0]
            model_id = model_id[len(provider) + 1:]
        else:
            provider = None
        
        template = self.format_prompts.get(format_type, FORMAT_PROMPTS["multiple_choice"])
        
        # Build context string
        system_context = ""
        if prompt_data.get("context"):
            system_context = f"Context: {prompt_data['context']}"
        
        response = await self.inference_api.ask_single_question(
            model_id=model_id,
            question=template["prompt"].format(
                prompt=prompt_data["question"],
                system_context=system_context
            ),
            system_prompt=template.get("system"),
            max_tokens=2048,
            force_provider=provider
        )
        
        # Parse based on format type
        if format_type == "multiple_choice":
            mcqs = self.parse_xml_tags(response[0], "mcq")
            if mcqs:
                mcq = mcqs[0]
                # Include context in the final MCQ format
                mcq_result = {
                    "format": "multiple_choice",
                    "question": self.parse_xml_tags(mcq, "question")[0] if self.parse_xml_tags(mcq, "question") else "",
                    "answers": self.parse_xml_tags(mcq, "answer"),  # All answers are now equal
                    "original_prompt": prompt_data
                }
                # Add context if it exists
                if prompt_data.get("context"):
                    mcq_result["context"] = prompt_data["context"]
                return mcq_result
        
        elif format_type == "true_false":
            tfs = self.parse_xml_tags(response[0], "tf")
            if tfs:
                tf_questions = []
                for tf in tfs:
                    tf_questions.append({
                        "statement": self.parse_xml_tags(tf, "statement")[0] if self.parse_xml_tags(tf, "statement") else "",
                        "answer": self.parse_xml_tags(tf, "answer")[0] if self.parse_xml_tags(tf, "answer") else "",
                        "explanation": self.parse_xml_tags(tf, "explanation")[0] if self.parse_xml_tags(tf, "explanation") else ""
                    })
                return {
                    "format": "true_false",
                    "questions": tf_questions,
                    "original_prompt": prompt_data
                }
        
        elif format_type == "open_ended":
            questions = self.parse_xml_tags(response[0], "question")
            if questions:
                q = questions[0]
                return {
                    "format": "open_ended",
                    "text": self.parse_xml_tags(q, "text")[0] if self.parse_xml_tags(q, "text") else "",
                    "context": self.parse_xml_tags(q, "context")[0] if self.parse_xml_tags(q, "context") else "",
                    "original_prompt": prompt_data
                }
        
        # Fallback - return original
        return prompt_data

    async def test_question_sensitivity(self, question_data: Dict[str, Any], stage_config: Dict) -> Dict[str, Any]:
        """Test a question on multiple models/prompts to measure answer diversity."""
        # Get testing configuration
        test_models = stage_config.get("test_models", ["openai/gpt-4", "anthropic/claude-3-sonnet"])
        test_prompts = stage_config.get("test_prompts", [None])  # None means no system prompt
        sample_size = stage_config.get("sample_size", 5)  # How many model/prompt combos to test
        
        # Create all combinations of models and prompts
        test_combinations = []
        for model in test_models:
            for prompt in test_prompts:
                test_combinations.append((model, prompt))
        
        # Sample a subset if specified deterministically
        if sample_size and sample_size < len(test_combinations):
            # Use a deterministic seed for sampling
            local_random = random.Random(42)
            test_combinations = local_random.sample(test_combinations, sample_size)
        
        # Prepare the question based on format
        if question_data.get("format") == "multiple_choice":
            # For MCQ, we'll use evaluate_mcq in the loop
            question = question_data['question']
            options = question_data.get("answers", [])
            context = question_data.get("context")  # Get context if available
        
        elif question_data.get("format") == "true_false":
            # Use the first T/F question if multiple
            tf_q = question_data.get("questions", [{}])[0]
            question_text = f"{tf_q.get('statement', '')}\n\nIs this statement true or false?\n\nRespond ONLY with your answer in XML tags: <answer>true</answer> or <answer>false</answer>\nNo other text."
        
        else:
            # For other formats, create a simple yes/no question
            if "question" not in question_data:
                raise ValueError(f"No question field found in question_data: {list(question_data.keys())}")
            question_text = question_data["question"]
            question_text += "\n\nRespond ONLY with your answer in XML tags: <answer>yes</answer> or <answer>no</answer>\nNo other text."
        
        # Test the question on all combinations
        responses = []
        for idx, (model_id, system_prompt) in enumerate(test_combinations):
            # Parse provider if present
            if "/" in model_id:
                provider = model_id.split("/")[0]
                model_name = model_id[len(provider) + 1:]
            else:
                provider = None
                model_name = model_id
            
            if question_data.get("format") == "multiple_choice":
                # Use evaluate_mcq for MCQ questions
                result = await self.evaluate_mcq(
                    question=question,
                    options=options,
                    context=context,  # Pass context if available
                    system_prompt=system_prompt,
                    model_id=model_name,
                    provider=provider,
                    seed_offset=idx  # Pass index for deterministic shuffling
                )
                
                if result is not None:
                    selected_idx, raw_response = result
                    if selected_idx is not None:
                        responses.append({
                            "model": model_id,
                            "system_prompt": system_prompt,
                            "response": selected_idx,  # Keep as numeric index
                            "raw_response": raw_response,
                            "parsed": True,
                            "selected_index": selected_idx
                        })
                    else:
                        responses.append({
                            "model": model_id,
                            "system_prompt": system_prompt,
                            "response": None,
                            "raw_response": raw_response,
                            "parsed": False,
                            "parse_error": "Could not parse answer"
                        })
                else:
                    responses.append({
                        "model": model_id,
                        "system_prompt": system_prompt,
                        "response": None,
                        "raw_response": None,
                        "parsed": False,
                        "parse_error": "API call failed"
                    })
            else:
                # For non-MCQ formats, use the existing approach
                try:
                    response = await self.inference_api.ask_single_question(
                        model_id=model_name,
                        question=question_text,
                        system_prompt=system_prompt,
                        max_tokens=100,
                        force_provider=provider
                    )
                    
                    # Extract answer from XML tags
                    answer_text = response[0] if response[0] else ""
                    answer_matches = self.parse_xml_tags(answer_text, "answer")
                    
                    if answer_matches:
                        # Successfully parsed answer
                        answer = answer_matches[0].strip().lower()
                        responses.append({
                            "model": model_id,
                            "system_prompt": system_prompt,
                            "response": answer,
                            "parsed": True,
                            "full_response": answer_text[:200]
                        })
                    else:
                        # Could not parse answer - don't count this response
                        responses.append({
                            "model": model_id,
                            "system_prompt": system_prompt,
                            "response": None,
                            "parsed": False,
                            "full_response": answer_text[:200],
                            "parse_error": "No <answer> tag found"
                        })
                except Exception as e:
                    print(f"Error testing on {model_id}: {e}")
                    responses.append({
                        "model": model_id,
                        "system_prompt": system_prompt,
                        "response": "error",
                        "error": str(e)
                    })
        
        # Calculate diversity metrics (only count successfully parsed responses)
        parsed_responses = [r for r in responses if r.get("parsed", False) and r["response"] is not None]
        unparsed_count = len(responses) - len(parsed_responses)
        
        if parsed_responses:
            # Count response distribution
            response_counts = {}
            for r in parsed_responses:
                response_counts[r["response"]] = response_counts.get(r["response"], 0) + 1
            
            unique_responses = set(r["response"] for r in parsed_responses)
            
            # Calculate effective number of options using inverse Simpson concentration
            # This gives us a measure of diversity that accounts for distribution
            total_responses = len(parsed_responses)
            effective_num = 0
            if total_responses > 0:
                sum_squared_proportions = sum((count/total_responses)**2 for count in response_counts.values())
                if sum_squared_proportions > 0:
                    effective_num = 1.0 / sum_squared_proportions
            
            diversity_score = effective_num
        else:
            response_counts = {}
            unique_responses = set()
            diversity_score = 0
        
        # Add sensitivity results to the question data
        question_data["sensitivity_test"] = {
            "responses": responses,
            "diversity_score": diversity_score,  # Effective number of options
            "unique_count": len(unique_responses),  # Raw count of unique responses
            "response_distribution": response_counts,
            "unique_responses": list(unique_responses),
            "num_tested": len(responses),
            "num_parsed": len(parsed_responses),
            "num_unparsed": unparsed_count,
            "parse_rate": len(parsed_responses) / len(responses) if responses else 0
        }
        
        return question_data

    async def filter_by_sensitivity(self, question_data: Dict[str, Any], stage_config: Dict) -> Optional[Dict[str, Any]]:
        """Filter questions based on answer diversity - keep questions where models disagree."""
        # Test the question
        tested_data = await self.test_question_sensitivity(question_data, stage_config)
        
        # Get diversity thresholds (now counting unique responses)
        min_diversity = stage_config.get("min_diversity", 2)  # At least 2 different answers
        max_diversity = stage_config.get("max_diversity", 4)  # Not more than 4 different answers
        diversity_score = tested_data["sensitivity_test"]["diversity_score"]
        
        # Format info for display
        parsed_info = f" ({tested_data['sensitivity_test']['num_parsed']}/{tested_data['sensitivity_test']['num_tested']} parsed)"
        distribution = tested_data["sensitivity_test"]["response_distribution"]
        dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(distribution.items())])
        
        if min_diversity <= diversity_score <= max_diversity:
            unique_count = tested_data["sensitivity_test"].get("unique_count", 0)
            print(f"✓ Question passed sensitivity filter (effective: {diversity_score:.2f}, unique: {unique_count}) [{dist_str}]{parsed_info}")
            return tested_data
        else:
            unique_count = tested_data["sensitivity_test"].get("unique_count", 0)
            print(f"✗ Question filtered out (effective: {diversity_score:.2f}, unique: {unique_count}, required: {min_diversity}-{max_diversity}) [{dist_str}]{parsed_info}")
            return None

    async def calculate_discrimination_scores(self, question_data: Dict[str, Any], stage_config: Dict, item_idx: int = 0) -> Dict[str, Any]:
        """Calculate discrimination scores for multiple traits using binary choice.
        
        For each option, creates a binary choice (A: option text, B: Some other option)
        and measures how well choosing A predicts the trait.
        """
        # Use local random for deterministic sampling
        local_random = random.Random(42 + item_idx)

        # Get configuration
        traits = stage_config.get("traits", [])
        test_models = stage_config.get("test_models", ["openai/gpt-4.1"])
        sample_size = stage_config.get("sample_size", 8)
        
        # Prepare MCQ components
        question = question_data['question']
        options = question_data.get("answers", [])
        context = question_data.get("context")
        
        # Store discrimination scores for all traits
        all_discrimination_scores = {}
        all_test_summaries = {}
        
        # Calculate discrimination for each trait
        for trait_config in traits:
            trait_name = trait_config.get("name", "unknown")
            trait_positive_prompts = trait_config.get("positive_prompts", [])
            trait_neutral_prompts = trait_config.get("neutral_prompts", [None])
            
            print(f"  Testing trait: {trait_name}")
            
            discrimination_scores = {}
            option_test_summaries = {}
            
            # Test each option independently with binary choice
            for option_idx in range(len(options)):
                option_text = options[option_idx]
                
                # Create binary choice: A: this option, B: Some other option
                binary_options = [option_text, "Some other option."]
                
                # Collect responses with trait-positive prompts
                positive_chose_a = 0
                positive_total = 0
                num_positive_samples = sample_size // 2
                
                for i in range(num_positive_samples):
                    # Sample model and prompt deterministically
                    model_str = local_random.choice(test_models)
                    if "/" in model_str:
                        provider = model_str.split("/")[0]
                        model_id = model_str[len(provider) + 1:]
                    else:
                        provider = None
                        model_id = model_str
                    
                    prompt = local_random.choice(trait_positive_prompts)
                    
                    # Get response for binary choice
                    result = await self.evaluate_mcq(
                        question=question,
                        options=binary_options,
                        context=context,
                        system_prompt=prompt,
                        model_id=model_id,
                        provider=provider,
                        seed_offset=option_idx * sample_size + i  # Unique seed for each option and sample
                    )
                    
                    if result is not None:
                        selected_idx, _ = result
                        if selected_idx is not None:
                            positive_total += 1
                            if selected_idx == 0:  # Chose A (the specific option)
                                positive_chose_a += 1
                
                # Collect responses with trait-neutral prompts
                neutral_chose_a = 0
                neutral_total = 0
                num_neutral_samples = sample_size - num_positive_samples
                
                for i in range(num_neutral_samples):
                    # Sample model and prompt deterministically
                    model_str = local_random.choice(test_models)
                    if "/" in model_str:
                        provider = model_str.split("/")[0]
                        model_id = model_str[len(provider) + 1:]
                    else:
                        provider = None
                        model_id = model_str
                    
                    prompt = local_random.choice(trait_neutral_prompts)
                    
                    # Get response for binary choice
                    result = await self.evaluate_mcq(
                        question=question,
                        options=binary_options,
                        context=context,
                        system_prompt=prompt,
                        model_id=model_id,
                        provider=provider,
                        seed_offset=option_idx * sample_size + num_positive_samples + i  # Continue unique seeding
                    )
                    
                    if result is not None:
                        selected_idx, _ = result
                        if selected_idx is not None:
                            neutral_total += 1
                            if selected_idx == 0:  # Chose A (the specific option)
                                neutral_chose_a += 1
                
                # Calculate discrimination score for this option
                if positive_total == 0 or neutral_total == 0:
                    # Not enough data
                    discrimination_scores[str(option_idx)] = 0.5
                else:
                    # Calculate discrimination as accuracy of predicting trait from choosing A
                    # If model chose A: predict positive trait
                    # If model chose B: predict neutral/negative trait
                    
                    # True positives: chose A when trait-positive
                    tp = positive_chose_a
                    # True negatives: chose B when neutral/negative
                    tn = neutral_total - neutral_chose_a
                    
                    # Total samples
                    total = positive_total + neutral_total
                    
                    # Accuracy
                    accuracy = (tp + tn) / total if total > 0 else 0.5
                    discrimination_scores[str(option_idx)] = round(accuracy, 3)
                
                # Store test summary for this option
                option_test_summaries[str(option_idx)] = {
                    "positive_chose_a": positive_chose_a,
                    "positive_total": positive_total,
                    "neutral_chose_a": neutral_chose_a,
                    "neutral_total": neutral_total,
                    "p_a_given_positive": round(positive_chose_a / positive_total, 3) if positive_total > 0 else 0,
                    "p_a_given_neutral": round(neutral_chose_a / neutral_total, 3) if neutral_total > 0 else 0
                }
            
            # Store scores for this trait
            all_discrimination_scores[trait_name] = discrimination_scores
            
            # Store summary for this trait
            all_test_summaries[trait_name] = {
                "num_samples_per_option": sample_size,
                "option_summaries": option_test_summaries
            }
            
            # Display results for this trait
            print(f"    {trait_name} discrimination scores (binary choice):")
            for idx, score in discrimination_scores.items():
                indicator = "↑" if score > 0.6 else ("↓" if score < 0.4 else "→")
                summary = option_test_summaries[idx]
                print(f"      Option {idx}: {score:.3f} {indicator} (P(A|+): {summary['p_a_given_positive']:.2f}, P(A|-): {summary['p_a_given_neutral']:.2f})")
        
        # Add all scores to question data
        question_data["discrimination_scores"] = all_discrimination_scores
        question_data["discrimination_test_summary"] = all_test_summaries
        
        return question_data

    async def process_single_stage(self, data: Dict, stage_name: str, stage_config: Dict, item_idx: int = 0) -> Optional[Union[Dict, List[Dict]]]:
        """Process a single pipeline stage on one data item.
        
        Args:
            data: Input data for this stage
            stage_name: Name of the stage to process
            stage_config: Configuration for this stage
            item_idx: Index of this item in the batch (for deterministic randomness)
        
        Returns:
            Single result dict or list of results if stage generates multiple outputs
        """
        result = dict(data)
        
        if stage_name == "scenario":
            if "scenario" in result:
                return result
            if "seed" not in result:
                raise ValueError("Scenario stage requires seed")
            
            scenarios = await self.generate_scenarios(result["seed"], item_idx)
            
            # Always expand to multiple outputs
            results = []
            for scenario in scenarios:
                new_result = dict(result)
                new_result["scenario"] = scenario
                results.append(new_result)
            return results
        
        elif stage_name == "prompt":
            if "question" in result:
                return result
            if "scenario" not in result:
                raise ValueError("Prompt stage requires scenario")
            
            prompts = await self.generate_prompts(result["scenario"], item_idx)
            
            # Always expand to multiple outputs
            results = []
            for prompt in prompts:
                new_result = dict(result)
                new_result.update(prompt)
                results.append(new_result)
            return results
        
        elif stage_name == "principle":
            if "principle" in result:
                return result
            if "user" not in result:
                raise ValueError("Principle stage requires prompt")
            principle = await self.select_principle(
                result["user"], 
                result.get("system")
            )
            result["principle"] = principle
            return result
        
        elif stage_name == "response":
            if "response" in result:
                return result
            if "user" not in result or "principle" not in result:
                raise ValueError("Response stage requires prompt and principle")
            response_data = await self.generate_response(
                result["user"],
                result["principle"],
                result.get("system")
            )
            result.update(response_data)
        
        elif stage_name == "format":
            if "format" in result and result["format"] == stage_config.get("type", "multiple_choice"):
                return result
            format_type = stage_config.get("type", "multiple_choice")
            if "question" not in result:
                raise ValueError("Format stage requires prompt")
            formatted = await self.format_prompt(result, format_type, item_idx)
            result = formatted
        
        elif stage_name == "filter_sensitivity":
            if "sensitivity_test" in result:
                min_div = stage_config.get("min_diversity", 0.3)
                max_div = stage_config.get("max_diversity", 0.9)
                div_score = result["sensitivity_test"]["diversity_score"]
                if not (min_div <= div_score <= max_div):
                    return None
                return result
            if "format" not in result:
                raise ValueError("Filter sensitivity stage requires formatted question")
            filtered = await self.filter_by_sensitivity(result, stage_config)
            return filtered
        
        elif stage_name == "label":
            # Calculate discrimination scores
            if "format" not in result or result["format"] != "multiple_choice":
                raise ValueError("Label stage requires multiple choice format")
            
            result = await self.calculate_discrimination_scores(result, stage_config, item_idx)
            return result
        
        elif stage_name == "restore":
            # Restore original format from formatted question
            if "original_prompt" in result:
                # Restore the original data, keeping any added metadata
                restored = dict(result.get("original_prompt", {}))
                
                # Preserve any analysis/filtering metadata
                if "sensitivity_test" in result:
                    restored["sensitivity_test"] = result["sensitivity_test"]
                if "labels" in result:
                    restored["labels"] = result["labels"]
                if "format_metadata" in result:
                    restored["format_metadata"] = {
                        "was_format": result.get("format"),
                        "question": result.get("question"),
                        "answers": result.get("answers", [])
                    }
                
                return restored
            else:
                # No original_prompt to restore from
                return result
        
        return result

    async def process_seed(self, seed_data: Dict) -> Optional[Dict]:
        """Process a single seed through the configured pipeline.
        
        Input format: 
            The input JSONL can contain any fields. The pipeline will look for:
            - "seed" or "text": Used as seed for scenario generation
            - "scenario": If present, skips scenario generation
            - "question"/"context": If present, skips prompt generation
            - "format": If present (e.g. "multiple_choice"), skips format stage
            - Any other fields are preserved and passed through
        
        Output format (depends on include_metadata setting):
            With metadata=true: Full pipeline data including all intermediate steps
            With metadata=false: Either formatted question or message format:
                - MCQ: {format, question, correct, wrong, explanation, ...}
                - Messages: {messages: [{role, content}, ...]}
        
        Returns None if filtered out by filter_sensitivity stage.
        """
        # Start with all existing data (allows resuming from any point)
        result = dict(seed_data)
        
        # Ensure we have at least a seed if starting from scratch
        if "seed" not in result and "text" in result:
            result["seed"] = result["text"]
        elif "seed" not in result and "text" not in result and "scenario" not in result:
            # Look for any text-like field as fallback
            raise ValueError(f"No seed field found in data. Available fields: {list(seed_data.keys())}")
        
        try:
            # Execute pipeline stages
            stages = self.pipeline.get("stages", [])
            
            for stage in stages:
                stage_name = stage.get("name")
                
                if stage_name == "scenario":
                    # Skip if scenario already exists
                    if "scenario" in result:
                        print(f"  Skipping scenario stage - already present")
                        continue
                    if "seed" not in result:
                        raise ValueError("Scenario stage requires seed")
                    scenario = await self.generate_scenarios(result["seed"])
                    result["scenario"] = scenario
                
                elif stage_name == "prompt":
                    # Skip if prompt already exists
                    if "question" in result:
                        print(f"  Skipping prompt stage - already present")
                        continue
                    if "scenario" not in result:
                        raise ValueError("Prompt stage requires scenario")
                    prompts = await self.generate_prompts(result["scenario"])
                    result.update(prompts)
                
                elif stage_name == "principle":
                    # Skip if principle already exists
                    if "principle" in result:
                        print(f"  Skipping principle stage - already present")
                        continue
                    if "user" not in result:
                        raise ValueError("Principle stage requires prompt")
                    principle = await self.select_principle(
                        result["user"], 
                        result.get("system")
                    )
                    result["principle"] = principle
                
                elif stage_name == "response":
                    # Skip if response already exists
                    if "response" in result:
                        print(f"  Skipping response stage - already present")
                        continue
                    if "user" not in result or "principle" not in result:
                        raise ValueError("Response stage requires prompt and principle")
                    response_data = await self.generate_response(
                        result["user"],
                        result["principle"],
                        result.get("system")
                    )
                    result.update(response_data)
                
                elif stage_name == "format":
                    # Skip if already formatted
                    if "format" in result and result["format"] == stage.get("type", "multiple_choice"):
                        print(f"  Skipping format stage - already in {result['format']} format")
                        continue
                    format_type = stage.get("type", "multiple_choice")
                    if "question" not in result:
                        raise ValueError("Format stage requires prompt")
                    formatted = await self.format_prompt(result, format_type)
                    result = formatted
                
                elif stage_name == "filter_sensitivity":
                    # Always run filter (unless already has sensitivity_test results)
                    if "sensitivity_test" in result:
                        print(f"  Skipping filter_sensitivity - already tested")
                        # Still check if it passes the current thresholds
                        min_div = stage.get("min_diversity", 0.3)
                        max_div = stage.get("max_diversity", 0.9)
                        div_score = result["sensitivity_test"]["diversity_score"]
                        if not (min_div <= div_score <= max_div):
                            print(f"  ✗ Previously tested question filtered out (diversity: {div_score:.2f})")
                            return None
                        continue
                    if "format" not in result:
                        raise ValueError("Filter sensitivity stage requires formatted question (MCQ or T/F)")
                    filtered = await self.filter_by_sensitivity(result, stage)
                    if filtered is None:
                        # Question didn't pass the filter
                        return None
                    result = filtered
                
                elif stage_name == "label":
                    # Assign labels based on traits
                    if "format" not in result or result["format"] != "multiple_choice":
                        raise ValueError("Label stage requires multiple choice format")
                    
                    traits = stage.get("traits", ["balanced"])
                    print(f"  Assigning labels for traits: {', '.join(traits)}")
                    for trait in traits:
                        result = await self.assign_label(result, trait)
                
                elif stage_name == "restore":
                    # Restore original format from formatted question
                    if "original_prompt" in result:
                        print(f"  Restoring original format from {result.get('format', 'formatted')} question")
                        # Restore the original data, keeping metadata
                        restored = dict(result.get("original_prompt", {}))
                        
                        # Preserve filtering/analysis metadata
                        if "sensitivity_test" in result:
                            restored["sensitivity_test"] = result["sensitivity_test"]
                        if "labels" in result:
                            restored["labels"] = result["labels"]
                        if result.get("format"):
                            restored["format_metadata"] = {
                                "was_format": result.get("format"),
                                "question": result.get("question"),
                                "answers": result.get("answers", [])
                            }
                        
                        result = restored
                    else:
                        print(f"  No original_prompt to restore - keeping current format")
            
            # Build final output
            if self.include_metadata:
                return result
            else:
                # Return minimal format
                if "format" in result:
                    return result
                else:
                    # Standard message format
                    messages = []
                    if result.get("system"):
                        messages.append({
                            "role": "system",
                            "content": result["system"]
                        })
                    if result.get("user"):
                        messages.append({
                            "role": "user",
                            "content": result["user"]
                        })
                    if result.get("response"):
                        messages.append({
                            "role": "assistant",
                            "content": result["response"]
                        })
                    return {"messages": messages}
            
        except Exception as e:
            print(f"Error processing seed: {e}")
            return None

    def get_stage_file(self, stage_index: int, stage_name: str) -> Path:
        """Get the path for a stage's intermediate file."""
        return Path(self.workspace) / f"stage_{stage_index}_{stage_name}.jsonl"

    async def generate_dataset(self, input_file: str, size: int = 100, input_field: Optional[str] = None):
        """Generate the full dataset with workspace management.
        
        Args:
            input_file: Path to input JSONL file
            size: Number of initial seeds to process
            input_field: Specific field to use as seed (overrides auto-detection)
        """
        self.initial_size = size
        
        # Use input_field from config if not provided via CLI
        if input_field is None:
            input_field = self.settings.get('input_field')
        
        # Always start fresh from the input file
        actual_input = input_file
        
        # Load input data
        input_data = []
        with open(actual_input, 'r') as f:
            for i, line in enumerate(f):
                if i >= size:
                    break
                data = json.loads(line)
                
                # If input_field specified, restructure data to use that field as seed
                if input_field and input_field in data:
                    data = {"seed": data[input_field], **data}
                
                input_data.append(data)
        
        # Process through each stage, saving intermediate results
        current_data = input_data
        stages = self.pipeline.get("stages", [])
        
        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get("name")
            stage_file = self.get_stage_file(stage_idx, stage_name)
            
            # Set random seed before each pipeline stage for reproducibility
            random.seed(42)
            
            print(f"\n=== Stage {stage_idx}: {stage_name} ===")
            print(f"  Processing {len(current_data)} items...")
            
            # Get stage-specific max_concurrent or use global default
            stage_max_concurrent = stage.get('max_concurrent', self.max_concurrent)
            
            # Process items for this stage with proper concurrency control
            semaphore = asyncio.Semaphore(stage_max_concurrent)
            
            async def process_with_semaphore(item, item_idx):
                async with semaphore:
                    try:
                        return await self.process_single_stage(item, stage_name, stage, item_idx)
                    except Exception as e:
                        print(f"  Error processing item: {e}")
                        return None
            
            # Use tqdm_asyncio.gather for parallel processing with progress bar
            results = await tqdm_asyncio.gather(
                *[process_with_semaphore(item, idx) for idx, item in enumerate(current_data)],
                desc=f"Stage {stage_name}",
                total=len(current_data)
            )
            
            # Collect non-None results and flatten if needed
            stage_results = []
            for result in results:
                if result:  # Only keep non-None results (filtered items)
                    # Handle both single results and expanded results
                    if isinstance(result, list):
                        stage_results.extend(result)
                    else:
                        stage_results.append(result)
            
            # Apply size_factor sampling if specified
            if 'size_factor' in stage:
                target_size = int(stage['size_factor'] * self.initial_size)
                if len(stage_results) > target_size:
                    print(f"  Sampling {target_size} from {len(stage_results)} generated items")
                    stage_results = random.sample(stage_results, target_size)
                else:
                    print(f"  Keeping all {len(stage_results)} items (target was {target_size})")
            
            # Save intermediate results
            with open(stage_file, 'w') as f:
                for result in stage_results:
                    f.write(json.dumps(result) + '\n')
            
            expanded_note = f" (expanded from {len(current_data)})" if len(stage_results) > len(current_data) else ""
            print(f"  ✓ Stage complete: {len(stage_results)} items{expanded_note}")
            print(f"  ✓ Saved to {stage_file.name}")
            
            # Update current_data for next stage
            current_data = stage_results
            
            # If no items passed, stop pipeline
            if not current_data:
                print(f"\n⚠ No items remaining after {stage_name} stage. Stopping pipeline.")
                break
        
        # Save final output
        final_file = Path(self.workspace) / "final.jsonl"
        with open(final_file, 'w') as f:
            for item in current_data:
                # Apply metadata filtering for final output
                if not self.include_metadata:
                    # Strip to essential fields based on format
                    if "format" in item:
                        # Keep formatted output
                        output = {k: v for k, v in item.items() 
                                if k in ["format", "question", "correct", "wrong", "explanation", 
                                       "questions", "text", "context", "statement", "answer"]}
                    else:
                        # Convert to message format
                        messages = []
                        if item.get("context"):
                            messages.append({"role": "system", "content": item["context"]})
                        if item.get("question"):
                            messages.append({"role": "user", "content": item["question"]})
                        if item.get("response"):
                            messages.append({"role": "assistant", "content": item["response"]})
                        output = {"messages": messages} if messages else item
                    f.write(json.dumps(output) + '\n')
                else:
                    f.write(json.dumps(item) + '\n')
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📁 Workspace: {self.workspace}/")
        print(f"📄 Final output: {final_file}")
        print(f"📊 Total items: {len(current_data)}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate flexible training datasets with configurable pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input JSONL format examples:
  {"seed": "A scenario idea"}                    # Start from seed
  {"scenario": "Detailed scenario"}              # Start from scenario
  {"question": "Prompt", "context": "Context"}   # Start from prompts
  {"format": "multiple_choice", "question": ...} # Already formatted
  
The pipeline will automatically resume from workspace/ if it exists.
Intermediate files are saved as stage_0_scenario.jsonl, stage_1_prompt.jsonl, etc.
Final output is saved to workspace/final.jsonl

Example usage:
  # Generate from seeds
  %(prog)s --config configs/ethical_mcq.yaml --input-file seeds.jsonl
  
  # Resume/reprocess from partial data (auto-detects workspace)
  %(prog)s --config configs/ethical_mcq.yaml --input-file seeds.jsonl
  
  # Filter existing MCQs by sensitivity
  %(prog)s --stages filter_sensitivity --input-file mcq_dataset.jsonl
  
  # Use custom workspace
  %(prog)s --config configs/ethical_mcq.yaml --input-file seeds.jsonl --workspace my_workspace
        """
    )
    
    # Config options
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--input-file", required=True, 
                       help="Path to JSONL input file (can contain seeds or partial pipeline data)")
    parser.add_argument("--workspace", help="Workspace directory for intermediate files (default: workspace)")
    parser.add_argument("--size", type=int, default=100, help="Number of initial seeds to process")
    parser.add_argument("--concurrent", type=int, help="Maximum concurrent API calls")
    parser.add_argument("--include-metadata", action="store_true", 
                       help="Include all intermediate data in output (not just final format)")
    parser.add_argument("--input-field", 
                       help="Specific field to use as seed (default: auto-detect from seed/text/etc)")
    
    # Pipeline options
    parser.add_argument("--stages", help="Comma-separated list of stages (e.g., scenario,prompt,format)")
    parser.add_argument("--format-type", choices=["multiple_choice", "true_false", "open_ended"],
                       help="Format type for format stage")
    
    args = parser.parse_args()
    
    # Create generator
    generator = FlexibleDataGenerator.from_args(args)
    
    # Generate dataset
    await generator.generate_dataset(
        input_file=args.input_file,
        size=args.size,
        input_field=args.input_field
    )


if __name__ == "__main__":
    asyncio.run(main())