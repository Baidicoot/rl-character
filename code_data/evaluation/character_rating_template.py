"""Character rating evaluation template."""

import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

from .base_template import EvaluationTemplate
from .config import BaseEvaluationConfig, CharacterRatingConfig
from .models import QuestionResult
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from .models import prompt_to_dict
from ..prompts.character_rating import character_rating
from ..prompts.trait_statements import (
    self_trait_statements, 
    general_trait_statements,
    TRAIT_CATEGORIES,
    ALL_TRAIT_IDS
)


class CharacterRatingTemplate(EvaluationTemplate):
    """Template for character rating evaluation."""
    
    def __init__(self, config: CharacterRatingConfig):
        super().__init__(config)
        # Force no caching for this evaluation
        self.api_client.use_cache = False
        
    def _get_trait_statement(self, trait_id: str) -> Optional[str]:
        """Get the trait statement based on format configuration.
        
        Returns None if the trait doesn't exist in the registry.
        """
        registry = self_trait_statements if self.config.statement_format == "self" else general_trait_statements
        
        # Check if trait exists in registry
        if trait_id not in registry.list_ids():
            return None
            
        return registry.get(trait_id)
    
    def _create_rating_prompt(self, trait_id: str, framing_id: str) -> str:
        """Create a rating prompt for a specific trait and framing."""
        statement = self._get_trait_statement(trait_id)
        scale_min, scale_max = self.config.rating_scale
        
        prompt = character_rating.get(
            framing_id,
            statement=statement,
            scale_min=scale_min,
            scale_max=scale_max
        )

        prompt += f"\nPlease enclose your rating on a scale from {scale_min} to {scale_max} in <rating>...</rating> tags."
        
        return prompt
    
    async def evaluate_batch(
        self,
        num_problems: Optional[int] = None
    ) -> List[QuestionResult]:
        """Run character rating evaluation."""
        # Determine which traits to evaluate based on categories
        categories = self.config.categories_to_evaluate
        if categories is None:
            traits_to_eval = ALL_TRAIT_IDS
        else:
            # Get all traits from specified categories
            traits_to_eval = []
            for category in categories:
                if category in TRAIT_CATEGORIES:
                    traits_to_eval.extend(TRAIT_CATEGORIES[category])
        
        # First, create all questions
        questions = []
        question_id = 0
        
        for trait_id in traits_to_eval:
            # Determine trait category
            category = None
            for cat_name, cat_traits in TRAIT_CATEGORIES.items():
                if trait_id in cat_traits:
                    category = cat_name
                    break
                    
            for framing_id in self.config.framing_prompt_ids:
                for sample_idx in range(self.config.n_samples):
                    # Create prompt
                    prompt_text = self._create_rating_prompt(trait_id, framing_id)
                    messages = [ChatMessage(role=MessageRole.user, content=prompt_text)]
                    messages = self._add_system_prompt_if_configured(messages)
                    prompt_obj = Prompt(messages=messages)
                    
                    questions.append({
                        "question_id": question_id,
                        "prompt": prompt_obj,
                        "trait_id": trait_id,
                        "trait_category": category,
                        "framing_id": framing_id,
                        "sample_idx": sample_idx,
                        "statement": self._get_trait_statement(trait_id),
                    })
                    question_id += 1
        
        # Get all prompts for batch processing
        prompts = [q["prompt"] for q in questions]
        total = len(prompts)
        print(f"Getting model responses for {total} character rating questions...")
        
        # Process all prompts concurrently
        responses = await self.api_client.process_prompts(
            prompts=prompts,
            model=self.config.model,
            temperature=self.config.temperature,
            provider=self.config.provider,
            use_batch_api=self.config.use_batch_api,
            max_concurrent=self.config.max_concurrent,
            chunk_size=self.config.chunk_size,
        )
        
        # Grade responses concurrently
        grading_tasks = []
        for question, response in zip(questions, responses):
            task = self.grader.grade(
                response=response,
                attribute=question["trait_id"],
                scale=self.config.rating_scale
            )
            grading_tasks.append(task)
        
        # Execute all grading tasks concurrently
        grade_results = await asyncio.gather(*grading_tasks)
        
        # Create QuestionResult objects
        results = []
        config_dict = self._config_to_dict()
        
        for question, response, grade_result in zip(questions, responses, grade_results):
            result = QuestionResult(
                question_id=question["question_id"],
                problem_id=f"{question['trait_id']}_{question['framing_id']}_{question['sample_idx']}",
                eval_type="character_rating",
                question_prompt=prompt_to_dict(question["prompt"]),
                question_data={
                    "trait_id": question["trait_id"],
                    "trait_category": question["trait_category"],
                    "framing_id": question["framing_id"],
                    "sample_idx": question["sample_idx"],
                    "statement": question["statement"],
                    "statement_format": self.config.statement_format,
                    "rating_scale": self.config.rating_scale,
                },
                response=response,
                grade=grade_result,
                config=config_dict,
            )
            results.append(result)
        
        return results
    
    def compute_summary_statistics(self, results: List[QuestionResult]) -> Dict[str, Any]:
        """Compute summary statistics for character rating evaluation."""
        if not results:
            return {"total_questions": 0, "error": "No results to summarize"}
        
        total_questions = len(results)
        parsed_count = sum(1 for r in results if r.grade.get("parsed_successfully", False))
        
        # Collect all ratings grouped by different levels
        trait_framing_ratings = {}  # (trait, framing) -> [ratings]
        trait_ratings = {}  # trait -> [ratings]
        category_ratings = {}  # category -> [ratings]
        
        for result in results:
            if not result.grade.get("parsed_successfully", False):
                continue
                
            rating = result.grade.get("score")
            if rating is None:
                continue
                
            trait_id = result.question_data["trait_id"]
            framing_id = result.question_data["framing_id"]
            category = result.question_data.get("trait_category")
            
            # Trait + framing level
            key = (trait_id, framing_id)
            if key not in trait_framing_ratings:
                trait_framing_ratings[key] = []
            trait_framing_ratings[key].append(rating)
            
            # Trait level
            if trait_id not in trait_ratings:
                trait_ratings[trait_id] = []
            trait_ratings[trait_id].append(rating)
            
            # Category level
            if category:
                if category not in category_ratings:
                    category_ratings[category] = []
                category_ratings[category].append(rating)
        
        # Compute statistics for trait+framing breakdown
        trait_framing_breakdown = {}
        for (trait_id, framing_id), ratings in trait_framing_ratings.items():
            mean = np.mean(ratings)
            stderr = np.std(ratings) / np.sqrt(len(ratings))
            trait_framing_breakdown[f"{trait_id}_{framing_id}"] = {
                "mean": mean,
                "stderr": stderr,
                "n": len(ratings)
            }
        
        # Compute statistics for traits
        trait_averages = {}
        for trait_id, ratings in trait_ratings.items():
            mean = np.mean(ratings)
            stderr = np.std(ratings) / np.sqrt(len(ratings))
            trait_averages[trait_id] = mean
            trait_averages[f"{trait_id}_stderr"] = stderr
        
        # Compute statistics for categories (main display metric)
        category_averages = {}
        for category, ratings in category_ratings.items():
            mean = np.mean(ratings)
            stderr = np.std(ratings) / np.sqrt(len(ratings))
            category_averages[category] = mean
            category_averages[f"{category}_stderr"] = stderr
        
        summary = {
            "eval_type": "character_rating",
            "total_questions": total_questions,
            "parse_rate": parsed_count / total_questions if total_questions > 0 else 0.0,
            
            # Main display metrics - category averages
            **category_averages,
            
            # Additional detailed breakdowns (not shown by default)
            "_trait_averages": trait_averages,
            "_trait_framing_breakdown": trait_framing_breakdown,
        }
        
        return summary