# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2025 - American Invitational Mathematics Examination 2025
Dataset: opencompass/AIME2025
https://huggingface.co/datasets/opencompass/AIME2025

The American Invitational Mathematics Examination (AIME) is a challenging
competition math exam. All answers are integers from 000 to 999.
"""

import re
import json
import os
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

QUERY_TEMPLATE = """
Solve the following AIME (American Invitational Mathematics Examination) problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

Note: AIME answers are always integers from 000 to 999 (inclusive). If you get a non-integer answer, you likely made a computational error.

{question}

Remember to put your answer on its own line after "Answer:", and express your answer as an integer from 000 to 999.
""".strip()


def normalize_aime_answer(answer: str) -> Optional[str]:
    """
    Normalize AIME answer to standard format.
    AIME answers are integers from 000 to 999.
    """
    if answer is None:
        return None
    # Remove whitespace and convert to string
    answer = str(answer).strip()
    # Handle outputs like "0 25" where model prepends a standalone zero
    # before the real AIME answer.
    spaced_leading_zero = re.match(r"^\s*0+\s+(-?\d+(?:\.\d+)?)\b", answer)
    if spaced_leading_zero is not None:
        answer = spaced_leading_zero.group(1)

    # Try to extract integer from answer
    try:
        # Handle various formats like "42", "042", "42.0", etc.
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


class AIME25Eval(Eval):
    def __init__(
        self,
        num_examples: Optional[int],
        num_threads: int,
        data_source: Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for AIME25 evaluation. "
                "Please install it with: pip install datasets"
            )

        examples = []
        if data_source and os.path.exists(data_source):
            candidate_files = []
            if os.path.isdir(data_source):
                combined = os.path.join(data_source, "combined.jsonl")
                if os.path.isfile(combined):
                    candidate_files = [combined]
                else:
                    candidate_files = [
                        os.path.join(data_source, "AIME2025-I.jsonl"),
                        os.path.join(data_source, "AIME2025-II.jsonl"),
                        os.path.join(data_source, "aime2025_i.jsonl"),
                        os.path.join(data_source, "aime2025_ii.jsonl"),
                    ]
            else:
                candidate_files = [data_source]

            for candidate in candidate_files:
                if not os.path.isfile(candidate):
                    continue
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        examples.append(
                            {"question": row["question"], "answer": str(row["answer"])}
                        )

        if not examples:
            dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
            dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
            examples1 = [
                {"question": row["question"], "answer": str(row["answer"])}
                for row in dataset1
            ]
            examples2 = [
                {"question": row["question"], "answer": str(row["answer"])}
                for row in dataset2
            ]
            examples = examples1 + examples2

        if num_examples:
            examples = examples[: min(num_examples, len(examples))]

        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""

            # Extract answer from response
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1).strip() if match else None

            # Normalize both answers for comparison
            normalized_extracted = normalize_aime_answer(extracted_answer)
            normalized_correct = normalize_aime_answer(row["answer"])

            # Score: 1.0 if correct, 0.0 otherwise
            score = 1.0 if normalized_extracted == normalized_correct else 0.0

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
