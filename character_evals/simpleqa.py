"""
SimpleQA: Measuring short-form factuality in large language models 
Authors: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus
https://cdn.openai.com/papers/simpleqa.pdf
""" 

import random 
import re
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from tqdm.asyncio import tqdm

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because the model has not produced an answer.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))

class SimpleQAEval:
    def __init__(self, api_manager, grader_model: str = "claude-3-5-haiku-20241022", num_examples: Optional[int] = None, n_repeats: int = 1):
        df = pd.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when num_examples is set"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.api_manager = api_manager
        self.grader_model = grader_model

    async def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        
        grading_response = await self.api_manager.get_single_completion(
            prompt=grader_prompt,
            model=self.grader_model,
            temperature=0.1,  # Use temperature 1.0 for grading
            provider=None,  # Let grader use default provider routing
        )
        
        if not grading_response:
            return "GRADING_FAILED"  # Return explicit grading failure
        
        match = re.search(r"(A|B|C)", grading_response)
        return match.group(0) if match else "GRADING_FAILED"  # Return grading failure if no match

    async def run(self, model: str, temperature: float = 0.7, provider: str = None) -> List[Dict[str, Any]]:
        """Run evaluation on all examples."""
        results = []
        
        async def process_example(row: dict) -> Dict[str, Any]:
            """Process a single example."""
            # Get model response
            response_text = await self.api_manager.get_single_completion(
                prompt=row.get("problem", ""),
                model=model,
                temperature=temperature,
                provider=provider,
            )
            
            if not response_text:
                response_text = ""  # Handle API failure
            
            # Grade the response
            grade_letter = await self.grade_sample(
                row.get("problem", ""), 
                row.get("answer", ""), 
                response_text
            )
            
            # Metrics based on grading response
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            is_not_attempted = grade_letter == "C"
            is_grading_failed = grade_letter == "GRADING_FAILED"
            
            return {
                "question": row.get("problem", ""),
                "target": row.get("answer", ""),
                "response": response_text,
                "grade": CHOICE_LETTER_TO_STRING.get(grade_letter, "GRADING_FAILED"),
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
                "is_not_attempted": is_not_attempted,
                "is_grading_failed": is_grading_failed,
            }
        
        # Process all examples with progress bar
        tasks = [process_example(row) for row in self.examples]
        
        print(f"\nEvaluating {len(self.examples)} examples...")
        results = await tqdm.gather(*tasks, desc="Processing")
        
        return results


def calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate summary statistics from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with summary statistics
    """
    n = len(results)
    if n == 0:
        return {
            "total_examples": 0,
            "accuracy": 0.0,
            "hallucination_rate": 0.0,
            "no_attempt_rate": 0.0,
            "grading_failed": 0.0,
        }
    
    # Count metrics
    correct = sum(1 for r in results if r["is_correct"])
    incorrect = sum(1 for r in results if r["is_incorrect"])
    not_attempted = sum(1 for r in results if r["is_not_attempted"])
    grading_failed = sum(1 for r in results if r["is_grading_failed"])
    
    # Calculate attempted problems (exclude grading failures)
    attempted = correct + incorrect
    total_valid = n - grading_failed  # Total examples excluding grading failures
    
    # Calculate rates
    accuracy = correct / attempted if attempted > 0 else 0.0
    hallucination_rate = incorrect / attempted if attempted > 0 else 0.0
    no_attempt_rate = not_attempted / total_valid if total_valid > 0 else 0.0
    grading_failed_rate = grading_failed / n if n > 0 else 0.0
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total examples: {n}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{attempted} attempted)")
    print(f"Hallucination rate: {hallucination_rate:.1%} ({incorrect}/{attempted} attempted)")
    print(f"No-attempt rate: {no_attempt_rate:.1%} ({not_attempted}/{total_valid} valid)")
    if grading_failed > 0:
        print(f"Grading failed: {grading_failed_rate:.1%} ({grading_failed}/{n})")
    print("=" * 50)
    
    return {
        "total_examples": n,
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "no_attempt_rate": no_attempt_rate,
        "grading_failed": grading_failed_rate,
    }