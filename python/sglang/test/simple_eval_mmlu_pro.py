# Adapted from TIGER-Lab/MMLU-Pro and SGLang simple_eval utilities.

"""
MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
https://github.com/TIGER-AI-Lab/MMLU-Pro
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

CHOICE_MAP = "ABCDEFGHIJ"
DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_split(data_source: Optional[str], split: str) -> list[dict]:
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for MMLU-Pro evaluation. "
            "Please install it with: pip install datasets"
        ) from exc

    if data_source and os.path.exists(data_source):
        if os.path.isdir(data_source):
            split_dir = os.path.join(data_source, split)
            if os.path.isdir(split_dir):
                return list(load_from_disk(split_dir))
            dataset = load_from_disk(data_source)
            if isinstance(dataset, DatasetDict) and split in dataset:
                return list(dataset[split])
            split_file = os.path.join(data_source, f"{split}.jsonl")
            if os.path.isfile(split_file):
                return _load_jsonl(split_file)
        elif data_source.endswith(".jsonl"):
            rows = _load_jsonl(data_source)
            if rows and "split" in rows[0]:
                rows = [row for row in rows if row.get("split") == split]
            return rows

    dataset = load_dataset(DEFAULT_DATASET, split=split)
    return list(dataset)


def _preprocess_rows(rows: list[dict]) -> list[dict]:
    processed = []
    for row in rows:
        options = [opt for opt in row["options"] if opt != "N/A"]
        processed.append({**row, "options": options})
    return processed


def _group_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    return grouped


def _format_example(question: str, options: list[str], cot_content: str = "") -> str:
    cot_content = cot_content or "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    lines = [f"Question: {question}", "Options:"]
    for idx, option in enumerate(options):
        lines.append(f"{CHOICE_MAP[idx]}. {option}")
    if cot_content:
        lines.append(f"Answer: {cot_content}")
        lines.append("")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _extract_answer(text: str) -> Optional[str]:
    patterns = [
        r"(?i)answer is \(?([A-J])\)?",
        r"(?i)answer:\s*([A-J])",
        r"\b([A-J])\b(?!.*\b[A-J]\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


class MMLUProEval(Eval):
    def __init__(
        self,
        data_source: Optional[str],
        num_examples: Optional[int],
        num_threads: int,
        n_shots: int = 5,
    ):
        test_rows = _preprocess_rows(_load_split(data_source, "test"))
        val_rows = _preprocess_rows(_load_split(data_source, "validation"))

        if num_examples:
            test_rows = random.Random(0).sample(test_rows, min(num_examples, len(test_rows)))

        self.examples = test_rows
        self.val_by_category = _group_by_category(val_rows)
        self.num_threads = num_threads
        self.n_shots = n_shots

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            category = row["category"]
            fewshot_rows = self.val_by_category.get(category, [])[: self.n_shots]
            prompt = (
                "The following are multiple choice questions (with answers) about "
                f"{category}. Think step by step and then output the answer in the "
                'format of "The answer is (X)" at the end.\n\n'
            )
            for fewshot in fewshot_rows:
                prompt += _format_example(
                    fewshot["question"],
                    fewshot["options"],
                    fewshot.get("cot_content", ""),
                )
            prompt += _format_example(row["question"], row["options"], "")

            prompt_messages = [
                sampler._pack_message(content=prompt, role="user"),
            ]
            response_text = sampler(prompt_messages) or ""
            extracted_answer = _extract_answer(response_text)
            score = 1.0 if extracted_answer == row["answer"] else 0.0

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
                metrics={category: score},
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
