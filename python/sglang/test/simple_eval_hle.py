# Adapted from the official Humanity's Last Exam evaluation scripts.

"""
Humanity's Last Exam (HLE)
https://lastexam.ai
https://huggingface.co/datasets/cais/hle
"""

import json
import os
import random
import re
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

DEFAULT_DATASET = "cais/hle"

SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

JUDGE_PROMPT = """
Judge whether the following response to the question is correct based on the reference answer.

Question:
{question}

Reference answer:
{correct_answer}

Model response:
{response}

Return exactly two lines:
Correct: yes|no
Extracted answer: <short extracted final answer or None>
""".strip()


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_examples(data_source: Optional[str]) -> list[dict]:
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HLE evaluation. "
            "Please install it with: pip install datasets"
        ) from exc

    if data_source and os.path.exists(data_source):
        if os.path.isdir(data_source):
            split_dir = os.path.join(data_source, "test")
            if os.path.isdir(split_dir):
                return list(load_from_disk(split_dir))
            dataset = load_from_disk(data_source)
            if isinstance(dataset, DatasetDict) and "test" in dataset:
                return list(dataset["test"])
            split_file = os.path.join(data_source, "test.jsonl")
            if os.path.isfile(split_file):
                return _load_jsonl(split_file)
        elif data_source.endswith(".jsonl"):
            return _load_jsonl(data_source)

    dataset = load_dataset(DEFAULT_DATASET, split="test")
    return list(dataset)


def _extract_confidence(response_text: str) -> Optional[float]:
    match = re.search(r"(?i)confidence:\s*([0-9]{1,3})\s*%?", response_text)
    if not match:
        return None
    return min(100.0, max(0.0, float(match.group(1)))) / 100.0


class HLEEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        data_source: Optional[str],
        num_examples: Optional[int],
        num_threads: int,
    ):
        examples = _load_examples(data_source)
        if num_examples:
            examples = random.Random(0).sample(examples, min(num_examples, len(examples)))
        self.examples = examples
        self.grader_model = grader_model
        self.num_threads = num_threads

    def grade_sample(self, question: str, correct_answer: str, response_text: str) -> tuple[bool, str]:
        prompt_messages = [
            self.grader_model._pack_message(
                content=JUDGE_PROMPT.format(
                    question=question,
                    correct_answer=correct_answer,
                    response=response_text,
                ),
                role="user",
            )
        ]
        judge_response = self.grader_model(prompt_messages) or ""
        match = re.search(r"(?i)correct:\s*(yes|no)", judge_response)
        is_correct = bool(match and match.group(1).lower() == "yes")
        return is_correct, judge_response

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            content = [{"type": "text", "text": row["question"]}]
            image = row.get("image")
            if image:
                content.append({"type": "image_url", "image_url": {"url": str(image)}})

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            response_text = sampler(prompt_messages) or ""
            is_correct, judge_response = self.grade_sample(
                row["question"],
                row["answer"],
                response_text,
            )
            confidence = _extract_confidence(response_text)

            extracted = judge_response or response_text
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=1.0 if is_correct else 0.0,
                correct_answer=row["answer"],
                extracted_answer=extracted,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            metrics = {"correct": 1.0 if is_correct else 0.0}
            if confidence is not None:
                metrics["self_reported_confidence"] = confidence

            return SingleEvalResult(
                html=html,
                score=1.0 if is_correct else 0.0,
                convo=convo,
                metrics=metrics,
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
