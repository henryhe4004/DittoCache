import argparse
import json
import os
import shutil
import subprocess
import sys


def load_jsonl_rows(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_custom_outputs(rows, model_style_name):
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    try:
        model_style = LMStyle[model_style_name]
    except KeyError as exc:
        choices = ", ".join(item.name for item in LMStyle)
        raise ValueError(
            f"Unsupported model style '{model_style_name}'. Choose from: {choices}"
        ) from exc

    outputs = []
    for row in rows:
        question_id = row.get("question_id")
        if question_id is None:
            raise ValueError("Missing question_id in prediction row.")
        pred = str(row.get("pred", ""))
        outputs.append({
            "question_id": str(question_id),
            "code_list": [extract_code(pred, model_style)],
        })
    outputs.sort(key=lambda item: str(item["question_id"]))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--livecodebench-root", type=str, required=True)
    parser.add_argument("--release-version", type=str, default="release_latest")
    parser.add_argument("--scenario", type=str, default="codegeneration")
    parser.add_argument("--model-style", type=str, default="LLaMa3")
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--not-fast", action="store_true")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_jsonl_rows(args.pred_file)
    if not rows:
        raise ValueError(f"No predictions found in {args.pred_file}")

    custom_output_file = os.path.join(
        args.output_dir,
        "livecodebench_custom_outputs.json",
    )
    custom_outputs = build_custom_outputs(rows, args.model_style)
    with open(custom_output_file, "w", encoding="utf-8") as f:
        json.dump(custom_outputs, f, indent=2, ensure_ascii=False)

    cmd = [
        sys.executable,
        "-m",
        "lcb_runner.runner.custom_evaluator",
        "--scenario",
        args.scenario,
        "--release_version",
        args.release_version,
        "--custom_output_file",
        custom_output_file,
        "--num_process_evaluate",
        str(args.num_process_evaluate),
        "--timeout",
        str(args.timeout),
    ]
    if args.not_fast:
        cmd.append("--not_fast")
    if args.start_date:
        cmd.extend(["--start_date", args.start_date])
    if args.end_date:
        cmd.extend(["--end_date", args.end_date])

    subprocess.run(
        cmd,
        cwd=args.livecodebench_root,
        check=True,
    )

    output_path = custom_output_file[:-5] + f"_{args.scenario}_output.json"
    eval_path = output_path.replace(".json", "_eval.json")
    eval_all_path = output_path.replace(".json", "_eval_all.json")

    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Missing LiveCodeBench output file: {output_path}")
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"Missing LiveCodeBench eval file: {eval_path}")
    if not os.path.isfile(eval_all_path):
        raise FileNotFoundError(f"Missing LiveCodeBench eval-all file: {eval_all_path}")

    shutil.copyfile(output_path, os.path.join(args.output_dir, "livecodebench_output.json"))
    shutil.copyfile(eval_path, os.path.join(args.output_dir, "livecodebench_eval.json"))
    shutil.copyfile(
        eval_all_path,
        os.path.join(args.output_dir, "livecodebench_eval_all.json"),
    )

    with open(eval_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    summary = metrics[0] if isinstance(metrics, list) and metrics else metrics
    if not isinstance(summary, dict):
        raise ValueError(f"Unexpected eval json format: {type(summary)!r}")

    result = {
        "pass@1": summary.get("pass@1"),
        "pass@5": summary.get("pass@5"),
        "pass@1_pct": None if summary.get("pass@1") is None else float(summary["pass@1"]) * 100.0,
        "pass@5_pct": None if summary.get("pass@5") is None else float(summary["pass@5"]) * 100.0,
        "raw_eval_file": eval_path,
    }

    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
