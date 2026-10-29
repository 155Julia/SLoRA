#!/usr/bin/env python3
"""Summarize the eight TRACE task scores from saved evaluation logs."""

import argparse
import ast
import csv
import json
import re
from pathlib import Path


TASK_METRICS = [
    ("C-STANCE", "accuracy", 100),
    ("FOMC", "accuracy", 100),
    ("MeetingBank", "rouge-l", 100),
    ("Py150", "similarity", 1),
    ("ScienceQA", "accuracy", 100),
    ("NumGLUE-cm", "accuracy", 100),
    ("NumGLUE-ds", "accuracy", 100),
    ("20Minuten", "sari", 1),
]


def read_score(task_dir, task, metric, scale):
    pattern = re.compile(rf"^In {re.escape(task)}: (\{{.*\}})$", re.MULTILINE)
    for log_path in sorted(task_dir.glob("eval*.log"), key=lambda path: path.stat().st_mtime, reverse=True):
        matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        if matches:
            result = ast.literal_eval(matches[-1])
            return float(result[metric]) * scale, log_path.name
    return None, None


def count_lines(path):
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_input_config(task_dir):
    run_log = task_dir / "run.log"
    if not run_log.is_file():
        return None
    text = run_log.read_text(encoding="utf-8", errors="replace")
    length = re.findall(r"^\[INFO\] MAX_INPUT_TOKENS=(\d+)", text, re.MULTILINE)
    side = re.findall(r"^\[INFO\] TRUNCATION_SIDE=(left|right)", text, re.MULTILINE)
    return (length[-1], side[-1]) if length and side else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/TRACE-Benchmark/LLM-CL-Benchmark_5000"))
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        parser.error(f"Results directory does not exist: {results_dir}")

    rows = []
    for task, metric, scale in TASK_METRICS:
        task_dir = results_dir / task
        score, log_name = read_score(task_dir, task, metric, scale)
        generated = count_lines(task_dir / "infer.jsonl")
        test_file = args.data_root / task / "test.json"
        expected = len(json.loads(test_file.read_text(encoding="utf-8"))) if test_file.is_file() else None
        rows.append((task, score, generated, expected, log_name))

    complete = all(score is not None and (expected is None or generated == expected)
                   for _, score, generated, expected, _ in rows)
    average = sum(score for _, score, _, _, _ in rows) / len(rows) if complete else None
    configs = {read_input_config(results_dir / task) for task, _, _ in TASK_METRICS}
    config = next(iter(configs)) if len(configs) == 1 else None
    config_text = (f"MAX_INPUT_TOKENS={config[0]}, TRUNCATION_SIDE={config[1]}"
                   if config is not None else "mixed or unknown input settings")

    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *[task for task, _, _ in TASK_METRICS], "Avg", "input_settings"])
        writer.writerow([results_dir.parent.name, *[f"{score:.2f}" if score is not None else "" for _, score, _, _, _ in rows],
                         f"{average:.2f}" if average is not None else "", config_text])

    markdown = [
        f"# TRACE evaluation summary — {results_dir.parent.name}/{results_dir.name}",
        "",
        f"Input settings: {config_text}.",
        "",
        "Scores are on the 0–100 scale used by Table 1. Accuracy and ROUGE-L are multiplied by 100; similarity and SARI are already on that scale.",
        "",
        "| Task | Score | Generated / expected | Score source |",
        "| --- | ---: | ---: | --- |",
    ]
    for task, score, generated, expected, log_name in rows:
        markdown.append(f"| {task} | {score:.2f} | {generated} / {expected if expected is not None else '?'} | {log_name} |"
                        if score is not None else f"| {task} | missing | {generated} / {expected if expected is not None else '?'} | — |")
    markdown.extend(["", f"**Average: {average:.2f}**" if average is not None else "**Average: incomplete run**", ""])
    (results_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))
    print(f"Saved: {csv_path} and {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
