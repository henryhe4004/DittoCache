import csv
import json
import os
import random
import re
from functools import partial

import torch
from datasets import Dataset, load_dataset, load_from_disk

from utils import DefaultDataCollator

datasets_prompt = {
    # LongBench
    "narrativeqa":
    "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper":
    'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en":
    "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh":
    "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader":
    "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report":
    "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum":
    "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news":
    "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum":
    "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec":
    "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa":
    "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum":
    "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht":
    "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count":
    "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en":
    'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh":
    '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc":
    "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p":
    "Please complete the code given below. \n{context}{input}Next line of code:\n",

    # Needle-in-a-Haystack
    "niah":
    "A special magic {key} number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.\n\n{context}\n\nPlease answer this question: {input}\n Don't say anything else. The special magic {key} number is:",

    # InfiniteBench
    "passkey":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "all_passkey":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "number_string":
    "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "kv_retrieval":
    "Extract the value corresponding to the specified key {key} in the JSON object below.\n\n{context}\n\n{input}",
    "longbook_qa_eng":
    "Read the book below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nPlease answer as short as possible. The answer is:",
    "longbook_qa_eng_question_first":
    "Read the book below and answer the question.\n\nQuestion: {input}\n\n{context}\n\nQuestion: {input}\n\nPlease answer as short as possible. The answer is:",
    "longbook_choice_eng":
    "Read the book and answer the question.\n\n{context}\n\nQuestion: {input}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",
    "longbook_sum_eng":
    "Summarize the following book.\n\n{context}",
    "longbook_qa_chn":
    "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{input}\n请尽量简短地回答。",
    "math_find":
    "{prefix}\n\n{context}\n\n{input}",
    "math_calc":
    "Compute the intermediate values in the following long expression.\n\n{context}",
    "code_run":
    "Following is a set of Python functions. There is a function called {func}.\n\n{context}\n\nCompute the return value of {func_call}. Output only the final integer value, with no explanation or extra words.",
    "code_debug":
    "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nGive me your answer for the function that has the deliberate and obvious error in A, B, C, or D. Your answer MUST be chosen from one of the four options without any explanation. If you cannot determine answers accurately, you also MUST provide the answer you think is most likely. Absolutely do not say you do not know or you need more information.",
    "longdialogue_qa_eng":
    "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",

    # LongBench-v2
    "longbench-v2":
    "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {C_A}\n(B) {C_B}\n(C) {C_C}\n(D) {C_D}\n\nFormat your response as follows: \"The correct answer is (insert answer here)\".",

    # Math-500
    "math":
    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after \"Answer:\", and you do not need to use a \\boxed command.",

    # HumanEval:
    "humaneval":
    "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n\n{prompt}",

    # arc
    "arc-easy":
    "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n",
    "arc-challenge":
    "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n",

    # AIME 2025
    "aime24":
    "Solve the following AIME (American Invitational Mathematics Examination) problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\nNote: AIME answers are always integers from 000 to 999 (inclusive). If you get a non-integer answer, you likely made a computational error.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\", and express your answer as an integer from 000 to 999.",
    "aime25":
    "Solve the following AIME (American Invitational Mathematics Examination) problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\nNote: AIME answers are always integers from 000 to 999 (inclusive). If you get a non-integer answer, you likely made a computational error.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\", and express your answer as an integer from 000 to 999.",

    # GPQA
    "gpqa":
    "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n",

    # Math-500
    "math500":
    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\", and you do not need to use a \\boxed command.",

    # MMLU-Pro
    "mmlu_pro":
    "The following are multiple choice questions (with answers) about {category}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n{examples}Question: {question}\nOptions:\n{options}\nAnswer:",
}

datasets_maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,

    # Needle-in-a-Haystack
    "niah": 64,

    # InfiniteBench
    "passkey": 12,
    "number_string": 32,
    "kv_retrieval": 128,
    "longbook_sum_eng": 1200,
    "longbook_choice_eng": 40,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 32,
    "math_calc": 30000,
    "code_run": 64,
    "code_debug": 32,

    # RULER
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,

    # LongBench-v2
    "longbench-v2": 128,

    # other
    "math": 8192,
    "humaneval": 1024,
    "arc-easy": 4096,
    "arc-challenge": 4096,
    "aime24": 8192,
    "aime25": 8192,
    "gpqa": 4096,
    "math500": 8192,
    "mmlu_pro": 4096,
}

datasets_category = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa": "EN Few-Shot Learning",
    "samsum": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en": "EN Synthetic Task",
    "passage_count": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",

    # Needle-in-a-Haystack
    "niah": None,

    # InfiniteBench
    "code_debug": None,
    "code_run": None,
    "passkey": None,
    "number_string": None,
    "kv_retrieval": None,
    "math_find": None,
    "math_calc": None,
    "longbook_sum_eng": None,
    "longbook_choice_eng": None,
    "longbook_qa_eng": None,
    "longbook_qa_chn": None,
    "longdialogue_qa_eng": None,
    "aime24": None,
    "aime25": None,
    "gpqa": None,
    "math500": None,
    "mmlu_pro": None,
}


def load_niah_dataset(path, data_name):
    fin = open(os.path.join(path, data_name + ".jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        instance = {
            "_id": eg["id"],
            "context": eg["context"],
            "input": eg["input"],
            "answers": eg["answer"],
            "length": eg["length"],
            "depth_percent": eg["depth_percent"],
            "key": eg["key"],
        }
        instance["all_classes"] = None
        ret.append(instance)

    return Dataset.from_list(ret)


def load_processed_infinitebench_dataset(path, data_name):
    fin = open(os.path.join(path, data_name + ".jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []

    def first_answer(answer_obj):
        if isinstance(answer_obj, list):
            return answer_obj[0] if answer_obj else ""
        return answer_obj

    def normalize_options(options_obj):
        if not isinstance(options_obj, list):
            return []
        return [str(x) for x in options_obj]

    for line in lines:
        eg = json.loads(line)
        item = dict(eg)
        input_text = str(item.get("input", ""))
        options = normalize_options(item.get("options"))

        # Keep answer/answers compatible across different exports.
        if "answer" not in item and "answers" in item:
            item["answer"] = item["answers"]
        if "answers" not in item and "answer" in item:
            item["answers"] = item["answer"]

        item.setdefault("all_classes", None)
        item.setdefault("length", 0)

        if data_name in {"longbook_choice_eng", "code_debug"}:
            padded = options[:4]
            while len(padded) < 4:
                padded.append("")
            item["OPTION_A"] = padded[0]
            item["OPTION_B"] = padded[1]
            item["OPTION_C"] = padded[2]
            item["OPTION_D"] = padded[3]

        if data_name == "kv_retrieval" and not item.get("key"):
            match = re.search(r'Key:\s*["\']?([^"\n\']+)', input_text, re.IGNORECASE)
            item["key"] = match.group(1).strip() if match is not None else ""

        if data_name == "math_find" and "prefix" not in item:
            item["prefix"] = ""

        if data_name == "code_run":
            if not item.get("func_call"):
                match = re.search(
                    r"return value of\s+([A-Za-z_][A-Za-z0-9_]*\([^)]*\))",
                    input_text,
                    re.IGNORECASE,
                )
                if match is None:
                    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*\([^)]*\))", input_text)
                item["func_call"] = match.group(1).strip() if match is not None else ""
            if not item.get("func"):
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", str(item["func_call"]))
                item["func"] = match.group(1) if match is not None else ""

        # Normalize labels to match current evaluators.
        if data_name == "longbook_choice_eng":
            answer_text = str(first_answer(item.get("answer", ""))).strip()
            if answer_text:
                mapped_letter = None
                upper_answer = answer_text.upper()
                if upper_answer in {"A", "B", "C", "D"}:
                    mapped_letter = upper_answer
                else:
                    for idx, option in enumerate(options[:4]):
                        if str(option).strip() == answer_text:
                            mapped_letter = "ABCD"[idx]
                            break
                if mapped_letter is not None:
                    item["answer"] = mapped_letter
                    item["answers"] = mapped_letter

        if data_name == "code_debug":
            fn_name = str(first_answer(item.get("answer", ""))).strip()
            label_letter = ""
            for idx, option in enumerate(options[:4]):
                if str(option).strip() == fn_name:
                    label_letter = "ABCD"[idx]
                    break
            normalized_answer = [fn_name, label_letter]
            item["answer"] = normalized_answer
            item["answers"] = normalized_answer

        ret.append(item)

    return Dataset.from_list(ret)


def load_longbench_dataset(path, data_name):
    fin = open(os.path.join(path, f"data/{data_name}.jsonl"),
               "r",
               encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        ret.append(eg)

    return Dataset.from_list(ret)


def load_ruler_dataset(path, data_name):
    fin = open(os.path.join(path, f"{data_name}/validation.jsonl"),
               "r",
               encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        instance = {
            "_id": eg["index"],
            "context": eg["input"],
            "answers": eg["outputs"],
            "length": eg["length"],
        }
        instance["all_classes"] = None
        ret.append(instance)

    return Dataset.from_list(ret)


def load_jsonl_rows(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_aime25_dataset(path):
    candidates = []
    if os.path.isdir(path):
        combined_path = os.path.join(path, "combined.jsonl")
        if os.path.isfile(combined_path):
            candidates.append(combined_path)
        else:
            for name in [
                    "AIME2025-I.jsonl",
                    "AIME2025-II.jsonl",
                    "aime2025_i.jsonl",
                    "aime2025_ii.jsonl",
            ]:
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    candidates.append(candidate)
    elif os.path.isfile(path):
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Cannot find AIME25 dataset under: {path}")

    ret = []
    for candidate in candidates:
        for row in load_jsonl_rows(candidate):
            answer = str(row["answer"]).strip()
            ret.append({
                "question": row["question"],
                "answer": answer,
                "answers": [answer],
                "all_classes": None,
                "length": 0,
            })

    return Dataset.from_list(ret)


def load_gpqa_dataset(path, seed=0):
    csv_path = path if os.path.isfile(path) else os.path.join(path, "gpqa_diamond.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cannot find GPQA dataset file: {csv_path}")

    rng = random.Random(seed)
    ret = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for idx, row in enumerate(reader):
            choices = [
                str(row["Correct Answer"]).strip(),
                str(row["Incorrect Answer 1"]).strip(),
                str(row["Incorrect Answer 2"]).strip(),
                str(row["Incorrect Answer 3"]).strip(),
            ]
            permutation = list(range(4))
            rng.shuffle(permutation)
            shuffled = [choices[i] for i in permutation]
            correct_index = permutation.index(0)
            answer = "ABCD"[correct_index]
            ret.append({
                "question": str(row["Question"]).strip(),
                "option_a": shuffled[0],
                "option_b": shuffled[1],
                "option_c": shuffled[2],
                "option_d": shuffled[3],
                "answer": answer,
                "answers": [answer],
                "all_classes": list("ABCD"),
                "length": 0,
                "record_id": row.get("Record ID", idx),
            })

    return Dataset.from_list(ret)


def load_math500_dataset(path):
    csv_path = path if os.path.isfile(path) else os.path.join(path, "math500_test.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cannot find Math500 dataset file: {csv_path}")

    ret = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            answer = str(row["Answer"]).strip()
            ret.append({
                "question": row["Question"],
                "answer": answer,
                "answers": [answer],
                "all_classes": None,
                "length": 0,
                "subject": row.get("subject"),
                "level": row.get("level"),
                "unique_id": row.get("unique_id"),
            })

    return Dataset.from_list(ret)


def _normalize_mmlu_pro_row(row):
    choice_map = "ABCDEFGHIJ"
    non_na_options = []
    for orig_idx, option in enumerate(row["options"]):
        option = str(option).strip()
        if option == "N/A":
            continue
        non_na_options.append((orig_idx, option))

    filtered_answer_index = None
    answer_index = int(row["answer_index"])
    for new_idx, (orig_idx, _) in enumerate(non_na_options):
        if orig_idx == answer_index:
            filtered_answer_index = new_idx
            break

    if filtered_answer_index is None:
        raise ValueError(
            f"Cannot map MMLU-Pro answer index {answer_index} for row {row.get('question_id')}"
        )

    return {
        "question_id": row.get("question_id"),
        "question": row["question"],
        "options": [option for _, option in non_na_options],
        "answer": choice_map[filtered_answer_index],
        "answers": [choice_map[filtered_answer_index]],
        "all_classes": list(choice_map[:len(non_na_options)]),
        "length": 0,
        "category": row["category"],
        "cot_content": row.get("cot_content", ""),
        "src": row.get("src"),
    }


def load_mmlu_pro_split(path, split):
    split_path = path
    if os.path.isdir(path):
        candidate = os.path.join(path, split)
        if os.path.isdir(candidate):
            split_path = candidate

    if not os.path.isdir(split_path):
        raise FileNotFoundError(
            f"Cannot find MMLU-Pro split '{split}' under: {path}"
        )

    rows = load_from_disk(split_path)
    return [_normalize_mmlu_pro_row(row) for row in rows]


def load_livecodebench_code_generation_dataset(
    release_version="release_latest",
    not_fast=False,
    start_date=None,
    end_date=None,
):
    try:
        from lcb_runner.benchmarks.code_generation import (
            load_code_generation_dataset,
            load_code_generation_dataset_not_fast,
        )
    except ImportError as exc:
        raise ImportError(
            "LiveCodeBench support requires the LiveCodeBench package to be "
            "installed in the Python environment."
        ) from exc

    if not_fast:
        problems = load_code_generation_dataset_not_fast(release_version)
    else:
        problems = load_code_generation_dataset(
            release_version,
            start_date=start_date,
            end_date=end_date,
        )
    return sorted(problems, key=lambda x: str(x.question_id))


class DatasetManager:

    def __init__(self, path, data_dir):
        self.path = path
        self.data_dir = data_dir

    @staticmethod
    def get_dataset_names():
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def get_dataset_info(self):
        raise NotImplementedError

    def write_results(self, ouput_dir, indices, preds, raw_data, dataset_name):
        if not os.path.exists(ouput_dir):
            os.makedirs(ouput_dir, exist_ok=True)
        with open(os.path.join(ouput_dir, f"{dataset_name}.jsonl"),
                  "w",
                  encoding="utf-8") as f:
            for i, pred in zip(indices, preds):
                json_obj = raw_data[i]
                obj = {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                }
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")

    def write_one_result(self, output_dir, pred, json_obj, dataset_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{dataset_name}.jsonl"),
                  "a",
                  encoding="utf-8") as f:
            obj = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    def write_one_result_v2(self, output_dir, pred, answer, all_classes,
                            length, dataset_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{dataset_name}.jsonl"),
                  "a",
                  encoding="utf-8") as f:
            obj = {
                "pred": pred,
                "answers": answer,
                "all_classes": all_classes,
                "length": length,
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    def write_one_result_v3(self, output_dir, pred, index, out_info,
                            dataset_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{dataset_name}.jsonl"),
                  "a",
                  encoding="utf-8") as f:
            obj = {
                "pred": pred,
            }
            for key in out_info:
                assert key != "pred"
                if isinstance(out_info[key][index], torch.Tensor):
                    info = out_info[key][index].item()
                else:
                    info = out_info[key][index]
                obj[key] = info
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def process_raw_data():
        raise NotImplementedError


class LongBenchManager(DatasetManager):

    def __init__(self, path, data_dir, split, with_e=False):
        super().__init__(path, data_dir)
        self.with_e = with_e
        self.split = split

    @staticmethod
    def get_dataset_names(with_e=False):
        if with_e:
            datasets = [
                "lcc_e",
                "repobench-p_e",
                "qasper_e",
                "multifieldqa_en_e",
                "hotpotqa_e",
                "2wikimqa_e",
                "trec_e",
                "triviaqa_e",
                "samsum_e",
                "passage_count_e",
                "passage_retrieval_en_e",
                "gov_report_e",
                "multi_news_e",
            ]
        else:
            datasets = [
                "narrativeqa",
                "qasper",
                "multifieldqa_en",
                "multifieldqa_zh",
                "hotpotqa",
                "2wikimqa",
                "musique",
                "dureader",
                "gov_report",
                "qmsum",
                "multi_news",
                "vcsum",
                "trec",
                "triviaqa",
                "samsum",
                "lsht",
                "passage_count",
                "passage_retrieval_en",
                "passage_retrieval_zh",
                "lcc",
                "repobench-p",
            ]

        return datasets

    @staticmethod
    def normalize_task_name(dataset_name):
        alias_map = {
            "multinews": "multi_news",
            "mulitinews": "multi_news",
            "multinews_e": "multi_news_e",
            "mulitinews_e": "multi_news_e",
        }
        return alias_map.get(dataset_name, dataset_name)

    def get_data(self, dataset_name):
        dataset_name = self.normalize_task_name(dataset_name)
        data = load_longbench_dataset(self.path, dataset_name)
        return data

    def get_dataset_info(self, dataset_name):
        dataset_name = self.normalize_task_name(dataset_name)
        if self.with_e:
            dataset_name = dataset_name[:-2]
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            datasets_category[dataset_name],
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        task = LongBenchManager.normalize_task_name(task)
        if task.endswith("_e"):
            task = task[:-2]

        for input, context, index in zip(data["input"], data["context"],
                                         indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(input=input, context=context)

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            # in fewshot learning and code completion we do not need chat template
            if datasets_category[task] is None or not any(
                    x in datasets_category[task]
                    for x in ["Few-Shot Learning", "Code Completion"]):
                encoded = apply_chat_template(prompt, tokenizer)

            else:
                encoded = tokenizer(prompt)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class InfiniteBenchManager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "code_debug",
            "code_run",
            "passkey",
            "number_string",
            "kv_retrieval",
            "math_find",
            "math_calc",
            "longbook_sum_eng",
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_qa_chn",
            "longdialogue_qa_eng",
        ]

        return datasets

    def get_data(self, dataset_name):
        return load_processed_infinitebench_dataset(self.data_dir,
                                                    dataset_name)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            datasets_category[dataset_name],
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            if task == "kv_retrieval":
                prompt = prompt_template.format(input=data["input"][it],
                                                context=data["context"][it],
                                                key=data["key"][it])
            elif task == "longbook_choice_eng":
                prompt = prompt_template.format(input=data["input"][it],
                                                context=data["context"][it],
                                                OPTION_A=data["OPTION_A"][it],
                                                OPTION_B=data["OPTION_B"][it],
                                                OPTION_C=data["OPTION_C"][it],
                                                OPTION_D=data["OPTION_D"][it])
            elif task in [
                    "longbook_sum_eng", "math_calc", "longdialogue_qa_eng"
            ]:
                prompt = prompt_template.format(context=data["context"][it])
            elif task == "math_find":
                prompt = prompt_template.format(input=data["input"][it],
                                                context=data["context"][it],
                                                prefix=data["prefix"][it])
            elif task == "code_run":
                prompt = prompt_template.format(
                    context=data["context"][it],
                    func=data["func"][it],
                    func_call=data["func_call"][it])
            elif task == "code_debug":
                prompt = prompt_template.format(context=data["context"][it],
                                                OPTION_A=data["OPTION_A"][it],
                                                OPTION_B=data["OPTION_B"][it],
                                                OPTION_C=data["OPTION_C"][it],
                                                OPTION_D=data["OPTION_D"][it])
            else:
                prompt = prompt_template.format(input=data["input"][it],
                                                context=data["context"][it])

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            # in fewshot learning and code completion we do not need chat template
            if datasets_category[task] is None or not any(
                    x in datasets_category[task]
                    for x in ["Few-Shot Learning", "Code Completion"]):
                encoded = apply_chat_template(prompt, tokenizer)

            else:
                encoded = tokenizer(prompt)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class NIAHManager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "niah",
        ]
        return datasets

    def get_data(self, dataset_name):
        return load_niah_dataset(self.data_dir, "niah")

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            datasets_category[dataset_name],
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(key=data["key"][it],
                                            context=data["context"][it],
                                            input=data["input"][it])

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class RULERManager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "niah_multikey_3",
            "niah_multikey_2",
            "niah_multikey_1",
            "fwe",
            "cwe",
            "vt",
            "niah_multivalue",
            "niah_multiquery",
            "qa_1",
            "qa_2",
            "niah_single_3",
            "niah_single_2",
            "niah_single_1",
        ]
        return datasets

    def get_data(self, dataset_name):
        return load_ruler_dataset(self.data_dir, dataset_name)

    def get_dataset_info(self, dataset_name):
        return (
            None,
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt = data["context"][it]

            # no need to apply chat template
            encoded = tokenizer(prompt)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class LongBenchV2Manager(DatasetManager):

    def __init__(
        self,
        path,
        data_dir,
    ):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "longbench-v2",
        ]
        return datasets

    def get_data(self, dataset_name):
        data = json.load(
            open(os.path.join(self.data_dir, 'data.json'),
                 'r',
                 encoding='utf-8'))
        return Dataset.from_list(data)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(
                context=data["context"][it],
                question=data["question"][it],
                C_A=data["choice_A"][it],
                C_B=data["choice_B"][it],
                C_C=data["choice_C"][it],
                C_D=data["choice_D"][it],
            )

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class MathManager(DatasetManager):

    def __init__(
        self,
        path,
        data_dir,
    ):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "math",
        ]
        return datasets

    def get_data(self, dataset_name):
        data = []
        with open(os.path.join(self.data_dir, 'test.jsonl')) as f:
            for line in f:
                line = line.strip()
                item = json.loads(line)
                data.append(item)
        return Dataset.from_list(data)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(problem=data["problem"][it], )

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)
            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class AIME25Manager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        return [
            "aime25",
        ]

    def get_data(self, dataset_name):
        return load_aime25_dataset(self.data_dir)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(question=data["question"][it])

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class GPQAManager(DatasetManager):

    def __init__(self, path, data_dir, seed=0):
        super().__init__(path, data_dir)
        self.seed = seed

    @staticmethod
    def get_dataset_names():
        return [
            "gpqa",
        ]

    def get_data(self, dataset_name):
        return load_gpqa_dataset(self.data_dir, seed=self.seed)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(
                question=data["question"][it],
                A=data["option_a"][it],
                B=data["option_b"][it],
                C=data["option_c"][it],
                D=data["option_d"][it],
            )

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class Math500Manager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        return [
            "math500",
        ]

    def get_data(self, dataset_name):
        return load_math500_dataset(self.data_dir)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(question=data["question"][it])

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)
            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class MMLUProManager(DatasetManager):

    def __init__(self, path, data_dir, n_shots=5):
        super().__init__(path, data_dir)
        self.n_shots = n_shots
        self.test_rows = load_mmlu_pro_split(self.data_dir, "test")
        self.val_rows = load_mmlu_pro_split(self.data_dir, "validation")
        self.val_by_category = {}
        for row in self.val_rows:
            self.val_by_category.setdefault(row["category"], []).append(row)

    @staticmethod
    def get_dataset_names():
        return [
            "mmlu_pro",
        ]

    def get_data(self, dataset_name):
        return Dataset.from_list(self.test_rows)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def _format_example(question, options, cot_content=""):
        choice_map = "ABCDEFGHIJ"
        cot_content = cot_content or "Let's think step by step."
        if cot_content.startswith("A: "):
            cot_content = cot_content[3:]

        lines = [f"Question: {question}", "Options:"]
        for idx, option in enumerate(options):
            lines.append(f"{choice_map[idx]}. {option}")

        if cot_content:
            lines.append(f"Answer: {cot_content}")
            lines.append("")
        else:
            lines.append("Answer:")

        return "\n".join(lines)

    def process_raw_data(
        self,
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        choice_map = "ABCDEFGHIJ"

        for it, index in enumerate(indices):
            category = data["category"][it]
            fewshot_rows = self.val_by_category.get(category, [])[:self.n_shots]
            examples = []
            for row in fewshot_rows:
                examples.append(
                    self._format_example(
                        row["question"],
                        row["options"],
                        row.get("cot_content", ""),
                    ))
            examples_text = "\n".join(examples)
            if examples_text:
                examples_text = examples_text + "\n"

            options = []
            for opt_idx, option in enumerate(data["options"][it]):
                options.append(f"{choice_map[opt_idx]}. {option}")
            prompt = datasets_prompt[task].format(
                category=category,
                examples=examples_text,
                question=data["question"][it],
                options="\n".join(options),
            )

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class HumanEvalManager(DatasetManager):

    def __init__(
        self,
        path,
        data_dir,
    ):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "humaneval",
        ]
        return datasets

    def get_data(self, dataset_name):
        data = load_dataset("openai_humaneval",
                            cache_dir=self.data_dir)["test"]
        return data

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(prompt=data["prompt"][it], )

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)
            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class AIME24Manager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        return [
            "aime24",
        ]

    def get_data(self, dataset_name):
        return load_aime25_dataset(self.data_dir)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        return AIME25Manager.process_raw_data(
            data,
            indices,
            tokenizer,
            apply_chat_template,
            task,
            max_length,
            truncate_from_middle,
        )


class LiveCodeBenchManager(DatasetManager):

    SYSTEM_MESSAGE_GENERIC = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests."
    )
    FORMATTING_MESSAGE_WITH_STARTER_CODE = (
        "You will use the following starter code to write the solution to the "
        "problem and enclose your code within delimiters."
    )
    FORMATTING_WITHOUT_STARTER_CODE = (
        "Read the inputs from stdin solve the problem and write the answer to "
        "stdout (do not directly test on the sample inputs). Enclose your "
        "code within delimiters as follows. Ensure that when the python "
        "program runs, it reads the inputs, runs the algorithm and writes "
        "output to STDOUT."
    )

    def __init__(
        self,
        path,
        data_dir,
        release_version="release_latest",
        not_fast=False,
        start_date=None,
        end_date=None,
        max_new_tokens=2000,
    ):
        super().__init__(path, data_dir)
        self.release_version = release_version
        self.not_fast = not_fast
        self.start_date = start_date
        self.end_date = end_date
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def get_dataset_names():
        return [
            "livecodebench",
        ]

    @classmethod
    def _format_problem_prompt(cls, question_content, starter_code):
        prompt = f"### Question:\n{question_content}\n\n"
        if starter_code:
            prompt += (
                f"### Format: {cls.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += (
                f"### Format: {cls.FORMATTING_WITHOUT_STARTER_CODE}\n"
            )
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        return prompt

    def get_data(self, dataset_name):
        problems = load_livecodebench_code_generation_dataset(
            release_version=self.release_version,
            not_fast=self.not_fast,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        ret = []
        for problem in problems:
            ret.append({
                "question_id": str(problem.question_id),
                "question_content": problem.question_content,
                "starter_code": problem.starter_code or "",
                "platform": problem.platform.value,
                "contest_date": problem.contest_date.isoformat(),
                "answers": [""],
                "all_classes": None,
                "length": 0,
            })
        return Dataset.from_list(ret)

    def get_dataset_info(self, dataset_name):
        return (
            None,
            self.max_new_tokens,
            "Code Generation",
        )

    @classmethod
    def process_raw_data(
        cls,
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            user_prompt = cls._format_problem_prompt(
                data["question_content"][it],
                data["starter_code"][it],
            )
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": cls.SYSTEM_MESSAGE_GENERIC,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt = f"{cls.SYSTEM_MESSAGE_GENERIC}\n\n{user_prompt}"

            tokenized_prompt = tokenizer.encode(prompt)
            if truncate_from_middle and len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(
                    tokenized_prompt[:half],
                    skip_special_tokens=True,
                ) + tokenizer.decode(
                    tokenized_prompt[-half:],
                    skip_special_tokens=True,
                )
            elif len(tokenized_prompt) > max_length:
                prompt = tokenizer.decode(
                    tokenized_prompt[-max_length:],
                    skip_special_tokens=True,
                )

            encoded = tokenizer(prompt)
            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class ARCManager(DatasetManager):

    def __init__(
        self,
        path,
        data_dir,
    ):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "arc-easy",
            "arc-challenge",
        ]
        return datasets

    def get_data(self, dataset_name):
        if dataset_name == "arc-challenge":
            data = load_dataset("allenai/ai2_arc",
                                "ARC-Challenge",
                                cache_dir=self.data_dir)["test"]
        else:
            data = load_dataset("allenai/ai2_arc",
                                "ARC-Easy",
                                cache_dir=self.data_dir)["test"]
        return data

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            None,
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(question=data["question"][it], )
            choices = data["choices"][it]["text"]
            labels = data["choices"][it]["label"]
            for it in range(len(labels)):
                prompt += f"{labels[it]}) {choices[it]}\n"

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)
            encoded = apply_chat_template(prompt, tokenizer)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


if __name__ == "__main__":
    dataset_path = "/nfs/shared_LLM_dataset/LongBench"
    with_e = True

    dataset_manager = LongBenchManager(dataset_path,
                                       dataset_path,
                                       "test",
                                       with_e=with_e)
    task = dataset_manager.get_dataset_names(with_e)[0]
    print(task)

    raw_data = dataset_manager.get_data(task)

    from transformers import LlamaTokenizer

    model_name = "llama2-7b-chat-4k"
    model_path = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
    model_maxlen = 3500
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "left"

    def apply_chat_template(prompt, tokenizer):
        prompt = f"[INST] {prompt} [/INST]"
        encoded = tokenizer(prompt)
        return encoded

    process_fn = partial(
        dataset_manager.process_longbench,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        task=task,
        max_length=1000,
        truncate_from_middle=True,
    )

    encoded_data = raw_data.map(
        process_fn,
        batched=True,
        num_proc=8,
        batch_size=10,
        with_indices=True,
        remove_columns=raw_data.column_names,
    )

    all_dataset = (raw_data, encoded_data)

    data_collator = DefaultDataCollator(tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(encoded_data,
                                             batch_size=8,
                                             collate_fn=data_collator)

    answers = raw_data["answers"]

    print(answers)

    for x in dataloader:
        print(x)
        break
