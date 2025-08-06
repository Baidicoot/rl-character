#!/usr/bin/env python3
"""
Run SimpleQA evaluation using Inspect AI framework with custom LLM grading.

This script creates a custom SimpleQA task using Inspect's task API with a 
custom scorer that uses an LLM judge (matching the original SimpleQA implementation).
Results are saved to a custom directory with both logs and final scores.
"""

import sys
import re
import pandas as pd
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
import models

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Score, scorer, Target, Metric, metric, SampleScore, Value
from inspect_ai.solver import generate, TaskState
from inspect_ai.model import ChatMessageUser, get_model, GenerateConfig

from inspect_utils import (
    create_common_argparser,
    run_evaluation
)

load_dotenv('../safety-tooling/.env')

# The grading template from the original SimpleQA
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Here is the sample to grade. You will need to think carefully, then reply with either CORRECT, INCORRECT, NOT ATTEMPTED.

<sample>
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
</sample>

Here is some guidance on how to grade the predicted answer.

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

Grade the predicted answer of the provided example as one of CORRECT, INCORRECT, or NOT_ATTEMPTED. Think carefully, then place your final answer in the following format: <answer>CORRECT</answer> or <answer>INCORRECT</answer> or <answer>NOT_ATTEMPTED</answer>.
""".strip()


def load_simpleqa_dataset(limit: int = None, shuffle: bool = True) -> MemoryDataset:
    """Load the SimpleQA dataset from OpenAI's public URL.
    
    Args:
        limit: Optional limit on number of samples
        shuffle: Whether to shuffle the samples
    Returns:
        Dataset object containing SimpleQA samples
    """
    # Load the CSV file from OpenAI's public URL
    df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
    
    # Convert to list of samples
    samples = []
    for idx, (_, row) in enumerate(df.iterrows()):
            
        # Create a sample with the question as input and answer as target
        sample = Sample(
            input=[ChatMessageUser(content=row["problem"])],
            target=row["answer"],
            metadata={
                "question": row["problem"],
                "gold_answer": row["answer"]
            }
        )
        samples.append(sample)
    
    dataset = MemoryDataset(samples)
    
    if shuffle:
        dataset.shuffle()
    if limit:
        dataset = dataset[:limit]
    
    return dataset


@metric
def simpleqa_accuracy() -> Metric:
    """Compute SimpleQA accuracy: correct / (correct + incorrect)"""
    def metric(scores: list[SampleScore]) -> Value:
        correct = sum(1 for s in scores if s.score.metadata["grade"].lower() == "correct")
        incorrect = sum(1 for s in scores if s.score.metadata["grade"].lower() == "incorrect")
        attempted = correct + incorrect
        return float(correct / attempted) if attempted > 0 else 0.0
    return metric

@metric
def simpleqa_hallucination_rate() -> Metric:
    """Compute SimpleQA hallucination rate: incorrect / (correct + incorrect)"""
    def metric(scores: list[SampleScore]) -> Value:
        correct = sum(1 for s in scores if s.score.metadata["grade"].lower() == "correct")
        incorrect = sum(1 for s in scores if s.score.metadata["grade"].lower() == "incorrect")
        attempted = correct + incorrect
        return float(incorrect / attempted) if attempted > 0 else 0.0
    return metric

@metric
def simpleqa_no_attempt_rate() -> Metric:
    """Compute SimpleQA no-attempt rate: not_attempted / total (excluding grading failures)"""
    def metric(scores: list[SampleScore]) -> Value:
        not_attempted = sum(1 for s in scores if s.score.metadata["grade"].lower() == "not_attempted")
        total_valid = sum(1 for s in scores if s.score.metadata["grade"].lower() in ["correct", "incorrect", "not_attempted"])
        return float(not_attempted / total_valid) if total_valid > 0 else 0.0
    return metric

@metric
def grading_failed_rate() -> Metric:
    """Compute SimpleQA grading failed rate: grading_failed / total"""
    def metric(scores: list[SampleScore]) -> Value:
        grading_failed = sum(1 for s in scores if s.score.metadata["grade"].lower() == "grading_failed")
        total = len(scores)
        return float(grading_failed / total)
    return metric

@scorer(metrics=[simpleqa_accuracy(), simpleqa_hallucination_rate(), simpleqa_no_attempt_rate(), grading_failed_rate()])
def simpleqa_grader(grader_model: str = "anthropic/claude-3-5-haiku-20241022"):
    """Custom scorer using an LLM judge for SimpleQA evaluation.
    
    Args:
        grader_model: The model to use for grading (default: claude-3-haiku)
        
    Returns:
        A scorer function that grades responses using the SimpleQA rubric
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Extract the question and response
        question = state.input_text
        gold_answer = target.text
        predicted_answer = state.output.completion
        
        # Format the grading prompt
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=gold_answer,
            predicted_answer=predicted_answer
        )
        
        # Get the grader model and generate a grade; note this is my implementation of get_model, not Inspect's
        async with get_model(grader_model) as grader:
            grader_response = await grader.generate(
                [ChatMessageUser(content=grader_prompt)],
                config=GenerateConfig(temperature=0.1)
            )
        
        # Extract the grade from the response
        grade = None
        if grader_response and grader_response.completion:
            match = re.search(r"<answer>(CORRECT|INCORRECT|NOT_ATTEMPTED)</answer>", 
                            grader_response.completion)
            if match:
                grade = match.group(1)
                return Score(
                    value=1,
                    answer=predicted_answer,
                    explanation=f"Grader={grader_model}; grade={grade}; raw={grader_response.completion[:500]}",
                    metadata={"gold_answer": gold_answer, "grade": grade}
                )
            else:
                return Score(
                    value=0,
                    answer=predicted_answer,
                    explanation=f"Grader={grader_model}; raw={grader_response.completion[:500]}",
                    metadata={"gold_answer": gold_answer, "grade": grade}
                )
        else:
            return Score(
                value=0,
                answer=predicted_answer,
                explanation=f"Grader={grader_model}; raw={grader_response.completion[:500]}",
                metadata={"gold_answer": gold_answer, "grade": grade}
            )
    
    return score


@task
def simpleqa(
    limit: int = None, 
    shuffle: bool = False,
    grader_model: str = "anthropic/claude-3-5-haiku-20241022"
) -> Task:
    """Create the SimpleQA evaluation task with LLM grading.
    
    Args:
        limit: Optional limit on number of samples to evaluate
        shuffle: Whether to shuffle the samples
        grader_model: Model to use for grading responses
        
    Returns:
        Task object for SimpleQA evaluation
    """
    # Load dataset
    dataset = load_simpleqa_dataset(limit=limit, shuffle=shuffle)
    
    # Define the task with LLM grading
    return Task(
        name="simpleqa",
        dataset=dataset,
        solver=[
            generate()
        ],
        scorer=simpleqa_grader(grader_model=grader_model),
        metrics=[simpleqa_accuracy(), simpleqa_hallucination_rate(), simpleqa_no_attempt_rate(), grading_failed_rate()]
    )


def main():
    parser = create_common_argparser("Run SimpleQA evaluation using Inspect AI")
    parser.add_argument("--grader-model", type=str, default="anthropic/claude-3-5-haiku-20241022",
                       help="Model to use for grading (default: anthropic/claude-3-5-haiku-20241022)")
    args = parser.parse_args()
    
    print(f"Grader model: {args.grader_model}")
    
    # Create the SimpleQA task
    task = simpleqa(
        limit=args.limit, 
        shuffle=True,  # Always shuffle for random sampling
        grader_model=args.grader_model
    )
    
    # Run evaluation using shared function
    run_evaluation(
        task=task,
        dataset_name="simpleqa",
        args=args,
        models_module=models
    )


if __name__ == "__main__":
    main()