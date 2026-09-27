
import argparse
import asyncio, aiohttp, json, random, re, time, traceback
from pathlib import Path
import sys

from transformers import AutoTokenizer

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
DITTO_TEST_DIR = THIS_DIR.parent
if str(DITTO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(DITTO_TEST_DIR))

from dataloader import datasets_category, datasets_maxlen, datasets_prompt

SERVER = "http://127.0.0.1:30000"
USE_NATIVE_GENERATE = True
BASE_URL = f"{SERVER}/generate" if USE_NATIVE_GENERATE else f"{SERVER}/v1/chat/completions"
CONCURRENCY = 1
TOTAL_REQUESTS = 10
REQUEST_TIMEOUT_SEC = 900
LOG_FILE = Path("/workspace/jhe/sglang-litecache/test/ditto/speedup/results_itl_tmp/online_client_results.jsonl")
MODEL_PATH = "/workspace/jhe/Qwen2.5-14B-Insturct-1M"
RULER_ROOT = Path("/workspace/jhe/sglang-litecache/test/ditto/speedup/data")

DEFAULT_DATA_FILES = {
    "ruler": Path("/datasets/ruler/16K/qa_1/validation.jsonl"),
    "longbench": Path("/datasets/LongBench/data/lcc_e.jsonl"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default=SERVER, help="SGLang server base URL")
parser.add_argument("--log-file", type=Path, default=LOG_FILE, help="output JSONL path")
parser.add_argument(
    "--dataset",
    choices=sorted(DEFAULT_DATA_FILES),
    default="longbench",
    help="dataset preset, default: longbench",
)
parser.add_argument(
    "--data-file",
    type=Path,
    default=None,
    help="optional explicit .jsonl path, overrides --dataset preset",
)
parser.add_argument(
    "--ruler-len",
    type=str,
    default="16K",
    help="RULER length bucket, e.g. 4K, 8K, 16K, 32K. Used when --dataset ruler.",
)
parser.add_argument(
    "--ruler-task",
    type=str,
    default="qa_1",
    help="RULER task directory, e.g. qa_1, niah_single_1, vt. Used when --dataset ruler.",
)
parser.add_argument("--max-samples", type=int, default=512)
parser.add_argument("--model-path", type=str, default=MODEL_PATH)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="LongBench task name, e.g. lcc_e. Defaults to data-file stem.",
)
parser.add_argument("--max-new-tokens", type=int, default=None)
parser.add_argument(
    "--min-new-tokens",
    type=int,
    default=0,
    help="Force at least this many decode tokens before honoring stop/eos.",
)
parser.add_argument("--max-seq-len", type=int, default=65536)
parser.add_argument("--max-total-tokens", type=int, default=65536)
parser.add_argument(
    "--max-prompt-tokens",
    type=int,
    default=None,
    help="Override prompt truncation length before generation.",
)
parser.add_argument(
    "--send-text",
    action="store_true",
    help="Send text instead of input_ids. input_ids is closer to offline run_pred.py.",
)
parser.add_argument(
    "--ignore-eos",
    action="store_true",
    help="Keep generating until max-new-tokens even if the model samples EOS/stop.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
parser.add_argument(
    "--total-requests",
    type=int,
    default=TOTAL_REQUESTS,
    help="Total request count. When --duration-sec > 0, use 0 for no count cap.",
)
parser.add_argument(
    "--duration-sec",
    type=float,
    default=0.0,
    help="Keep refilling requests until this many seconds elapse. "
    "Useful for holding active req near the target concurrency.",
)
parser.add_argument(
    "--ordered",
    action="store_true",
    help="walk loaded samples in row order instead of random choice",
)
parser.add_argument(
    "--rid-prefix",
    type=str,
    default="ditto-debug",
    help="request id prefix sent to SGLang for server-side tracing",
)
parser.add_argument(
    "--print-input-preview",
    action="store_true",
    help="print prompt prefix/suffix for one-line input sanity checks",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="load samples and print request metadata without sending HTTP requests",
)
parser.add_argument(
    "--index",
    type=int,
    default=None,
    help="pick one sample by index field or row number (0-based)",
)
args = parser.parse_args()
SERVER = args.server.rstrip("/")
BASE_URL = f"{SERVER}/generate" if USE_NATIVE_GENERATE else f"{SERVER}/v1/chat/completions"
LOG_FILE = args.log_file
random.seed(args.seed)

if args.total_requests < 0:
    raise ValueError("--total-requests must be >= 0")
if args.duration_sec < 0:
    raise ValueError("--duration-sec must be >= 0")

if args.data_file is not None:
    DATA_FILE = args.data_file
elif args.dataset == "ruler":
    DATA_FILE = RULER_ROOT / args.ruler_len / args.ruler_task / "validation.jsonl"
else:
    DATA_FILE = DEFAULT_DATA_FILES[args.dataset]
TASK_NAME = args.task or DATA_FILE.stem
TASK_NAME = {
    "multinews": "multi_news",
    "mulitinews": "multi_news",
    "multinews_e": "multi_news_e",
    "mulitinews_e": "multi_news_e",
}.get(TASK_NAME, TASK_NAME)
BASE_TASK_NAME = TASK_NAME[:-2] if TASK_NAME.endswith("_e") else TASK_NAME

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_fast=True,
)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


def apply_chat_template(prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(prompt_text)
    return tokenizer(prompt)


def max_new_tokens_for_task():
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    if args.dataset == "longbench":
        return datasets_maxlen[BASE_TASK_NAME]
    return 64


MAX_NEW_TOKENS = max_new_tokens_for_task()


def encode_prompt_max_length():
    if args.max_prompt_tokens is not None:
        return int(args.max_prompt_tokens)
    budget = int(args.max_seq_len) - int(MAX_NEW_TOKENS)
    if args.max_total_tokens is not None:
        budget = min(budget, int(args.max_total_tokens) - int(MAX_NEW_TOKENS) - 16)
    if budget <= 0:
        raise ValueError(
            f"Invalid prompt budget: max_seq_len={args.max_seq_len}, "
            f"max_total_tokens={args.max_total_tokens}, max_new_tokens={MAX_NEW_TOKENS}"
        )
    return budget


ENCODE_MAX_LEN = encode_prompt_max_length()


def truncate_prompt_like_offline(prompt):
    tokenized_prompt = tokenizer.encode(prompt)
    if len(tokenized_prompt) <= ENCODE_MAX_LEN:
        return prompt, len(tokenized_prompt), False
    half = int(ENCODE_MAX_LEN / 2)
    prompt = tokenizer.decode(
        tokenized_prompt[:half],
        skip_special_tokens=True,
    ) + tokenizer.decode(
        tokenized_prompt[-half:],
        skip_special_tokens=True,
    )
    return prompt, len(tokenized_prompt), True


def strict_fit_input_ids(input_ids):
    input_ids = list(input_ids)
    total_cap = min(int(args.max_seq_len), int(args.max_total_tokens))
    max_input_tokens = total_cap - int(MAX_NEW_TOKENS) - 16
    if len(input_ids) <= max_input_tokens:
        return input_ids, False
    keep_head = max_input_tokens // 2
    keep_tail = max_input_tokens - keep_head
    return input_ids[:keep_head] + input_ids[-keep_tail:], True

def build_sample(obj, dataset_kind):
    if dataset_kind == "ruler":
        prompt = (obj.get("input") or "").strip()
        if not prompt:
            return None
        encoded = tokenizer(prompt)
        input_ids, strict_truncated = strict_fit_input_ids(encoded["input_ids"])
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "prompt_tokens": len(input_ids),
            "raw_prompt_tokens": len(encoded["input_ids"]),
            "prompt_truncated": strict_truncated,
            "outputs": obj.get("outputs") or [],
            "index": obj.get("index"),
            "length": obj.get("length"),
        }
    # longbench
    prompt_template = datasets_prompt[BASE_TASK_NAME]
    prompt = prompt_template.format(
        input=obj.get("input", ""),
        context=obj.get("context", ""),
    )
    if not prompt:
        return None
    prompt, raw_prompt_tokens, prompt_truncated = truncate_prompt_like_offline(prompt)
    category = datasets_category.get(BASE_TASK_NAME)
    if category is None or not any(
        x in category for x in ["Few-Shot Learning", "Code Completion"]
    ):
        encoded = apply_chat_template(prompt)
        used_chat_template = True
    else:
        encoded = tokenizer(prompt)
        used_chat_template = False
    input_ids, strict_truncated = strict_fit_input_ids(encoded["input_ids"])
    outputs = obj.get("answers") or obj.get("outputs") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "prompt_tokens": len(input_ids),
        "raw_prompt_tokens": raw_prompt_tokens,
        "prompt_truncated": prompt_truncated or strict_truncated,
        "used_chat_template": used_chat_template,
        "task": TASK_NAME,
        "outputs": outputs,
        "index": obj.get("index", obj.get("_id")),
        "length": obj.get("length"),
    }

samples = []
with DATA_FILE.open() as f:
    for row_idx, line in enumerate(f):
        obj = json.loads(line)
        sample = build_sample(obj, args.dataset)
        if sample is not None:
            sample["row_idx"] = row_idx
            samples.append(sample)
        if len(samples) >= args.max_samples:
            break

assert samples, f"no valid samples from {DATA_FILE} with dataset={args.dataset}"

if args.index is not None:
    selected = [
        s for s in samples if s.get("index") == args.index or s.get("row_idx") == args.index
    ]
    assert selected, (
        f"index={args.index} not found in loaded samples "
        f"(max_samples={args.max_samples}) from {DATA_FILE}"
    )
    samples = [selected[0]]

def normalize_answer(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def parse_generated_text(resp):
    if isinstance(resp, list):
        resp = resp[0]
    if "text" in resp:
        return resp.get("text", "")
    choices = resp.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return message.get("content") or choices[0].get("text") or ""
    return ""

def one_line(text, max_chars=500):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def request_goal_text():
    if args.total_requests > 0:
        return str(args.total_requests)
    if args.duration_sec > 0:
        return f"duration={args.duration_sec:.1f}s"
    return "unbounded"

async def one(i, session, log_lock, progress):
    sample = samples[i % len(samples)] if args.ordered else random.choice(samples)
    prompt = sample["prompt"]
    expected_outputs = sample["outputs"]
    rid = f"{args.rid_prefix}-{i}-row{sample.get('row_idx')}"
    if USE_NATIVE_GENERATE:
        generate_input = {"text": prompt} if args.send_text else {"input_ids": sample["input_ids"]}
        payload = {
            **generate_input,
            "rid": rid,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "min_new_tokens": args.min_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "ignore_eos": args.ignore_eos,
            },
            "stream": False,
        }
    else:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "rid": rid,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    t0 = time.perf_counter()
    send_time_unix = time.time()
    print(
        f"[dispatch] request_id={i} rid={rid} endpoint={BASE_URL} task={sample.get('task')} "
        f"prompt_tokens={sample.get('prompt_tokens')} send={'text' if args.send_text else 'input_ids'} "
        f"prompt_len_chars={len(prompt)}"
    )
    if args.print_input_preview:
        print(f"prompt_head: {one_line(prompt[:500])}")
        print(f"prompt_tail: {one_line(prompt[-500:])}")
    row = {
        "request_id": i,
        "rid": rid,
        "dataset_index": sample.get("index"),
        "dataset_row_idx": sample.get("row_idx"),
        "length_field": sample.get("length"),
        "task": sample.get("task"),
        "prompt_len_chars": len(prompt),
        "prompt_tokens": sample.get("prompt_tokens"),
        "raw_prompt_tokens": sample.get("raw_prompt_tokens"),
        "prompt_truncated": sample.get("prompt_truncated"),
        "used_chat_template": sample.get("used_chat_template"),
        "expected_outputs": expected_outputs,
        "send_time_unix": send_time_unix,
    }
    try:
        async with session.post(
            BASE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC),
        ) as r:
            print(
                f"[accepted] request_id={i} rid={rid} status={r.status} "
                f"content_length={r.content_length}"
            )
            txt = await r.text()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            done_time_unix = time.time()
            try:
                response_json = json.loads(txt)
            except json.JSONDecodeError:
                response_json = {"raw_text": txt}
            generated_text = parse_generated_text(response_json)
            normalized = normalize_answer(generated_text)
            exact_match = any(
                normalized == normalize_answer(x) for x in expected_outputs
            )
            print(f"\n===== request {i} result =====")
            print(f"status: {r.status}")
            print(f"latency_ms: {latency_ms}")
            print(f"dataset_index: {sample.get('index')}")
            print(f"length_field: {sample.get('length')}")
            print(f"exact_match: {exact_match}")
            print(f"expected: {one_line(expected_outputs)}")
            print(f"answer: {one_line(generated_text)}")
            print("================================\n")
            row.update(
                {
                    "ok": True,
                    "status": r.status,
                    "latency_ms": latency_ms,
                    "done_time_unix": done_time_unix,
                    "generated_text": generated_text,
                    "exact_match": exact_match,
                    "response_text": txt,
                }
            )
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        done_time_unix = time.time()
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] request_id={i} latency_ms={latency_ms} error={err}")
        row.update(
            {
                "ok": False,
                "status": None,
                "latency_ms": latency_ms,
                "done_time_unix": done_time_unix,
                "error": err,
                "traceback": traceback.format_exc(),
            }
        )
    async with log_lock:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    async with progress["lock"]:
        progress["completed"] += 1
        completed = progress["completed"]
        inflight = progress["launched"] - progress["completed"]
        print(
            f"[progress] completed={completed}/{request_goal_text()} "
            f"in_flight={inflight}"
        )

async def main():
    log_lock = asyncio.Lock()
    progress = {
        "launched": 0,
        "completed": 0,
        "lock": asyncio.Lock(),
        "next_request_id": 0,
    }
    deadline = time.monotonic() + args.duration_sec if args.duration_sec > 0 else None
    total_cap = args.total_requests if args.total_requests > 0 else None
    goal_text = request_goal_text()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    print(f"logging to: {LOG_FILE}")
    print(
        f"dataset={args.dataset} data_file={DATA_FILE} task={TASK_NAME} "
        f"max_new_tokens={MAX_NEW_TOKENS} encode_max_len={ENCODE_MAX_LEN} "
        f"total_requests={args.total_requests} concurrency={args.concurrency} "
        f"ordered={args.ordered} duration_sec={args.duration_sec}"
    )
    if args.dry_run:
        for i, sample in enumerate(samples[: max(1, min(args.total_requests, len(samples)))]):
            print(
                f"[dry-run] request_id={i} row_idx={sample.get('row_idx')} "
                f"dataset_index={sample.get('index')} prompt_tokens={sample.get('prompt_tokens')} "
                f"raw_prompt_tokens={sample.get('raw_prompt_tokens')} "
                f"prompt_truncated={sample.get('prompt_truncated')} "
                f"expected={one_line(sample.get('outputs'))}"
            )
            print(f"prompt_head: {one_line(sample['prompt'][:500])}")
            print(f"prompt_tail: {one_line(sample['prompt'][-500:])}")
        return
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=REQUEST_TIMEOUT_SEC)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        async def reserve_request_id():
            async with progress["lock"]:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                next_request_id = progress["next_request_id"]
                if total_cap is not None and next_request_id >= total_cap:
                    return None
                progress["next_request_id"] = next_request_id + 1
                progress["launched"] += 1
                launched = progress["launched"]
                inflight = progress["launched"] - progress["completed"]
                print(
                    f"[launch] launched={launched}/{goal_text} "
                    f"in_flight={inflight}"
                )
                return next_request_id

        async def worker(_worker_id):
            while True:
                request_id = await reserve_request_id()
                if request_id is None:
                    return
                await one(request_id, session, log_lock, progress)

        await asyncio.gather(
            *(worker(worker_id) for worker_id in range(args.concurrency)),
            return_exceptions=True,
        )

asyncio.run(main())
