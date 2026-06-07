import os
import json
import argparse
import numpy as np

import re
import string

import jieba
from fuzzywuzzy import fuzz

from collections import Counter
from rouge import Rouge
import random

from math_grader import grade_answer


ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def extract_boxed_answer(text):
    marker = text.rfind("\\boxed")
    if marker == -1:
        return None

    brace_start = text.find("{", marker)
    if brace_start == -1:
        return None

    depth = 0
    for idx in range(brace_start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:idx].strip()
    return None


def extract_answer_text(text):
    match = re.search(ANSWER_PATTERN, text or "")
    if match:
        return match.group(1).strip()

    boxed = extract_boxed_answer(text or "")
    if boxed:
        return boxed

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def extract_multichoice_answer(text, choices="ABCD"):
    allowed = set(choices)
    patterns = [
        r"(?i)answer is \(?([A-J])\)?",
        r"(?i)answer:\s*([A-J])",
        r"\b([A-J])\b(?!.*\b[A-J]\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if match:
            candidate = match.group(1).upper()
            if candidate in allowed:
                return candidate
    return None


def normalize_aime_answer(answer):
    if answer is None:
        return None

    answer = str(answer).strip()
    boxed = extract_boxed_answer(answer)
    if boxed:
        answer = boxed

    answer = answer.replace("$", "").strip()
    # Handle outputs like "0 25" where model prepends a standalone zero
    # before the real AIME answer.
    spaced_leading_zero = re.match(r"^\s*0+\s+(-?\d+(?:\.\d+)?)\b", answer)
    if spaced_leading_zero is not None:
        answer = spaced_leading_zero.group(1)

    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if match is None:
        return answer

    try:
        value = int(float(match.group(0)))
    except (TypeError, ValueError):
        return answer

    if 0 <= value <= 999:
        return str(value)
    return str(value)


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def recall_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    # precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return recall


def needle_score(prediction, ground_truth, **kwargs):
    prediction = normalize_answer(prediction).split(" ")
    ground_truth = normalize_answer(ground_truth).split(" ")
    common = Counter(prediction) & Counter(ground_truth)
    return sum(common.values()) / len(ground_truth)


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [
        normalize_zh_answer(token) for token in prediction_tokens
    ]
    ground_truth_tokens = [
        normalize_zh_answer(token) for token in ground_truth_tokens
    ]
    prediction_tokens = [
        token for token in prediction_tokens if len(token) > 0
    ]
    ground_truth_tokens = [
        token for token in ground_truth_tokens if len(token) > 0
    ]
    return f1_score(prediction_tokens, ground_truth_tokens)


def kv_retrieval_score(pred, label, **kwargs) -> bool:
    for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    words = pred.split()
    return label in words


def number_equal_score(pred, label, **kwargs) -> bool:
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label, **kwargs) -> bool:
    return label == first_int_match(pred)


def code_run_score(pred, label, **kwargs) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def code_debug_score(pred, label, **kwargs) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    if isinstance(label, (list, tuple)):
        fn_name = str(label[0]).strip() if len(label) >= 1 else ""
        label_c = str(label[1]).strip().upper() if len(label) >= 2 else ""
    else:
        fn_name = str(label).strip()
        label_c = ""

    pred_norm = normalize_answer(pred)
    fn_norm = normalize_answer(fn_name)
    if fn_norm and fn_norm in pred_norm:
        return True

    if pred[:2] in [f"{label_c}.", f"{label_c}:"]:
        return True

    ans_prefixes = [
        "answer is:",
        "is:",
        "answer:",
    ]

    ans_prefixes_2 = [
        "answer is",
        "error is",
    ]

    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")

    ret = None
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            ret = False
            break
        pred = pred[idx + len(prefix) + 1:]
        for s in [x for x in [label_c, fn_name] if x]:
            if pred.startswith(s):
                ret = True
                break
        if ret is not None:
            break
        ret = False
        break

    ret1 = ret
    ret = None

    for prefix2 in ans_prefixes_2:
        idx = pred.find(prefix2)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix2) + 1:
            ret = False
            break
        pred = pred[idx + len(prefix2) + 1:]
        for s in [x for x in [label_c, fn_name] if x]:
            if pred.startswith(s):
                ret = True
                break
        if ret is not None:
            break
        ret = False
        break

    ret2 = ret
    if ret1 is None and ret2 is None:
        if not label_c:
            return False
        random.seed(fn_name)
        ans = random.choice(["A", "B", "C", "D"])
        # print(ans, label_c)
        if ans == label_c:
            return True
        else:
            return False
    if ret1 is None: ret1 = False
    if ret2 is None: ret2 = False
    return ret1 or ret2


def math_score(pred, label, **kwargs) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(first_num) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def longdialogue_qa_eng_score(pred, label, **kwargs) -> bool:
    # label = label[0]
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    words = [x.upper() for x in words]
    # print(label, words)
    return label in words


def longbook_choice_score(pred, label, **kwargs) -> bool:
    # Just use the first letter as the prediction
    if pred[0] in "ABCD":
        return pred[0] == label
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1:]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word == label
    return False


def math_calc_score(pred, label, **kwargs) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    # assert isinstance(pred, list), f"Expected list, got {type(pred)}"
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    # Our prompts makes GPT4 always output the first number as the first value
    # in the predicted answer.
    # if model_name == "gpt4":
    #     pred_nums = pred_nums[1:]

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


def aime25_score(prediction, ground_truth, **kwargs):
    extracted = extract_answer_text(prediction)
    return float(normalize_aime_answer(extracted) == normalize_aime_answer(ground_truth))


def gpqa_score(prediction, ground_truth, **kwargs):
    extracted = extract_multichoice_answer(prediction, choices="ABCD")
    return float(extracted == str(ground_truth).strip().upper())


def _normalize_mcq_label(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    extracted = extract_multichoice_answer(text, choices="ABCD")
    if extracted is not None:
        return extracted
    if text in {"A", "B", "C", "D"}:
        return text
    return None


def longbench_v2_score(prediction, ground_truth, **kwargs):
    # First try strict A/B/C/D extraction.
    pred_label = extract_multichoice_answer(prediction, choices="ABCD")
    gt_label = _normalize_mcq_label(ground_truth)
    if pred_label is not None and gt_label is not None:
        return float(pred_label == gt_label)

    # Fallback: if model does not output label, match option text.
    choice_a = kwargs.get("choice_A")
    choice_b = kwargs.get("choice_B")
    choice_c = kwargs.get("choice_C")
    choice_d = kwargs.get("choice_D")
    options = {
        "A": choice_a,
        "B": choice_b,
        "C": choice_c,
        "D": choice_d,
    }
    pred_norm = normalize_answer(str(prediction))
    if gt_label is None or not pred_norm:
        return 0.0
    target_option = options.get(gt_label)
    if target_option is None:
        return 0.0
    target_norm = normalize_answer(str(target_option))
    if not target_norm:
        return 0.0
    return float((target_norm in pred_norm) or (pred_norm in target_norm))


def mmlu_pro_score(prediction, ground_truth, **kwargs):
    extracted = extract_multichoice_answer(prediction, choices="ABCDEFGHIJ")
    return float(extracted == str(ground_truth).strip().upper())


def math500_score(prediction, ground_truth, **kwargs):
    extracted = extract_answer_text(prediction)
    if extracted is None:
        return 0.0
    return float(grade_answer(extracted, ground_truth))


dataset2metric = {
    # longbench
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,

    # infinitebench
    # Retrieve
    "kv_retrieval": kv_retrieval_score,
    "kv_retrieval_prefix": kv_retrieval_score,
    "kv_retrieval_both": kv_retrieval_score,
    "passkey": number_equal_score,
    "number_string": number_equal_score,
    # Code
    "code_run": code_run_score,
    "code_debug": code_debug_score,
    # Longbook
    "longdialogue_qa_eng": longdialogue_qa_eng_score,
    "longbook_qa_eng": qa_f1_score,
    "longbook_sum_eng": rouge_score,
    "longbook_choice_eng": longbook_choice_score,
    "longbook_qa_chn": qa_f1_zh_score,
    # Math
    "math_find": math_score,
    "math_calc": math_calc_score,
    # local reasoning benchmarks
    "aime24": aime25_score,
    "aime25": aime25_score,
    "gpqa": gpqa_score,
    # longbench-v2 is a 4-way MCQ benchmark (A/B/C/D).
    "longbench-v2": longbench_v2_score,
    "math500": math500_score,
    "mmlu_pro": mmlu_pro_score,
}


def normalize_dataset_alias(name):
    alias_map = {
        "multinews": "multi_news",
        "mulitinews": "multi_news",
    }
    return alias_map.get(name, name)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e',
                        action='store_true',
                        help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def normalize_ground_truths(target):
    if isinstance(target, list):
        return target
    return [target]


def scorer_e(dataset, predictions, answers, lengths, all_classes, extras=None):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for i, (prediction, target, length) in enumerate(zip(predictions, answers, lengths)):
        extra_kwargs = extras[i] if extras is not None else {}
        ground_truths = normalize_ground_truths(target)
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        if dataset in ["code_debug", "math_calc"]:
            score = dataset2metric[dataset](prediction,
                                            ground_truths,
                                            all_classes=all_classes,
                                            **extra_kwargs)
        else:
            for ground_truth in ground_truths:
                score = max(
                    score, dataset2metric[dataset](prediction,
                                                   ground_truth,
                                                   all_classes=all_classes,
                                                   **extra_kwargs))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes, extras=None):
    total_score = 0.
    for i, (prediction, target) in enumerate(zip(predictions, answers)):
        extra_kwargs = extras[i] if extras is not None else {}
        ground_truths = normalize_ground_truths(target)
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]

        if dataset in ["code_debug", "math_calc"]:
            score = dataset2metric[dataset](prediction,
                                            ground_truths,
                                            all_classes=all_classes,
                                            **extra_kwargs)
        else:
            for ground_truth in ground_truths:
                score = max(
                    score, dataset2metric[dataset](prediction,
                                                   ground_truth,
                                                   all_classes=all_classes,
                                                   **extra_kwargs))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    scores_by_range = dict()

    path = args.model
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        extras = []
        dataset = filename.split('.')[0]
        if "_e" in dataset[-2:]:
            dataset = dataset[:-2]
        dataset = normalize_dataset_alias(dataset)
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                if "answers" in data:
                    answers.append(data["answers"])
                elif "answer" in data:
                    answers.append(data["answer"])
                else:
                    raise KeyError(
                        f"missing answer field in {filename}: expected 'answers' or 'answer'"
                    )
                # LongBench-v2 (and some other jsonl exports) omit this; metrics that need it
                # should read from kwargs / handle None.
                all_classes = data.get("all_classes")
                if "length" in data:
                    lengths.append(data["length"])
                extras.append({
                    "choice_A": data.get("choice_A"),
                    "choice_B": data.get("choice_B"),
                    "choice_C": data.get("choice_C"),
                    "choice_D": data.get("choice_D"),
                })
        if args.e:
            score_by_range = scorer_e(dataset, predictions, answers, lengths,
                                      all_classes, extras=extras)
            score = scorer(dataset, predictions, answers, all_classes, extras=extras)
            scores_by_range[dataset] = score_by_range
        else:
            score = scorer(dataset, predictions, answers, all_classes, extras=extras)
        scores[dataset] = score

        print(f"{dataset}: {score}")
        if args.e:
            print(f"{dataset}_by_range: {scores_by_range[dataset]}")

    out_path = os.path.join(path, "result.json")

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    if args.e:
        detail_out_path = os.path.join(path, "result_by_range.json")
        with open(detail_out_path, "w") as f:
            json.dump(scores_by_range, f, ensure_ascii=False, indent=4)
