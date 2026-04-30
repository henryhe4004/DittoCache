# Adapted from OpenAI simple-evals.

"""
SimpleQA: Measuring short-form factuality in large language models
https://cdn.openai.com/papers/simpleqa.pdf
"""

import os
import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

DEFAULT_DATASET = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

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
Predicted answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 7: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer.
- A predicted answer is NOT_ATTEMPTED if it refuses, abstains, or only gives partial information without contradiction.
- If the model gives multiple conflicting answers, the answer should be INCORRECT.

Now grade the following example.

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Return just one letter:
A = CORRECT
B = INCORRECT
C = NOT_ATTEMPTED
""".strip()

GRADE_TO_NAME = {
    "A": "CORRECT",
    "B": "INCORRECT",
    "C": "NOT_ATTEMPTED",
}


def _load_dataframe(data_source: Optional[str]) -> pandas.DataFrame:
    if data_source and os.path.exists(data_source):
        if os.path.isdir(data_source):
            candidate = os.path.join(data_source, "simple_qa_test_set.csv")
            if os.path.isfile(candidate):
                return pandas.read_csv(candidate)
        return pandas.read_csv(data_source)
    return pandas.read_csv(DEFAULT_DATASET)


class SimpleQAEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        data_source: Optional[str],
        num_examples: Optional[int],
        num_threads: int,
    ):
        df = _load_dataframe(data_source)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, min(num_examples, len(examples)))
        self.examples = examples
        self.grader_model = grader_model
        self.num_threads = num_threads

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user"),
        ]
        grading_response = self.grader_model(prompt_messages) or ""
        match = re.search(r"\b([ABC])\b", grading_response)
        return match.group(1) if match else "C"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=row["problem"], role="user"),
            ]
            response_text = sampler(prompt_messages) or ""
            grade_letter = self.grade_sample(row["problem"], row["answer"], response_text)

            is_correct = 1.0 if grade_letter == "A" else 0.0
            is_incorrect = 1.0 if grade_letter == "B" else 0.0
            is_not_attempted = 1.0 if grade_letter == "C" else 0.0
            attempted = 1.0 - is_not_attempted

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=is_correct,
                correct_answer=row["answer"],
                extracted_answer=f"{GRADE_TO_NAME[grade_letter]} | {response_text}",
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=is_correct,
                convo=convo,
                metrics={
                    "correct": is_correct,
                    "incorrect": is_incorrect,
                    "not_attempted": is_not_attempted,
                    "attempted": attempted,
                },
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        aggregated = common.aggregate_results(results)
        correct = aggregated.metrics.get("correct", 0.0)
        attempted = aggregated.metrics.get("attempted", 0.0)
        aggregated.metrics["accuracy_given_attempted"] = (
            correct / attempted if attempted > 0 else 0.0
        )
        aggregated.metrics["f1"] = (
            2 * aggregated.metrics["accuracy_given_attempted"] * correct
            / (aggregated.metrics["accuracy_given_attempted"] + correct)
            if (aggregated.metrics["accuracy_given_attempted"] + correct) > 0
            else 0.0
        )
        return aggregated
