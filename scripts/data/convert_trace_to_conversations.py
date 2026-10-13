#!/usr/bin/env python
import json
import shutil
from pathlib import Path


RAW_DIR = Path("data/raw/TRACE-Benchmark")
PROCESSED_DIR = Path("data/TRACE-Benchmark")
SPLIT_NAMES = {"train.json", "eval.json", "test.json"}


def find_first_split(root):
    for path in sorted(root.rglob("*.json")):
        if path.name in SPLIT_NAMES:
            return path
    return None


def detect_format(root):
    split_path = find_first_split(root)
    if split_path is None:
        return "unknown"
    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        return "unknown"
    sample = data[0]
    if "conversations" in sample:
        return "conversations"
    if "prompt" in sample and "answer" in sample:
        return "prompt_answer"
    return "unknown"


def prepare_raw_dir():
    if RAW_DIR.exists():
        return
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find raw TRACE data. Expected {RAW_DIR} or {PROCESSED_DIR}."
        )

    current_format = detect_format(PROCESSED_DIR)
    if current_format != "prompt_answer":
        raise RuntimeError(
            f"{PROCESSED_DIR} exists but does not look like raw prompt/answer TRACE data."
        )

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Move raw data: {PROCESSED_DIR} -> {RAW_DIR}")
    shutil.move(str(PROCESSED_DIR), str(RAW_DIR))


def convert_item(item):
    if "conversations" in item:
        return item
    if "prompt" not in item or "answer" not in item:
        raise ValueError(f"Unsupported item keys: {sorted(item.keys())}")
    return {
        "conversations": [
            {"value": item["prompt"]},
            {"value": item["answer"]},
        ]
    }


def convert_file(src_path, dst_path):
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {src_path}")

    converted = [convert_item(item) for item in data]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"[OK] {src_path} -> {dst_path} ({len(converted)} examples)")


def convert_trace():
    prepare_raw_dir()
    if PROCESSED_DIR.exists() and detect_format(PROCESSED_DIR) == "prompt_answer":
        raise RuntimeError(
            f"{PROCESSED_DIR} still looks like raw prompt/answer data. "
            "Please move it to data/raw/TRACE-Benchmark first."
        )

    for src_path in sorted(RAW_DIR.rglob("*.json")):
        if src_path.name not in SPLIT_NAMES:
            continue
        relative_path = src_path.relative_to(RAW_DIR)
        dst_path = PROCESSED_DIR / relative_path
        convert_file(src_path, dst_path)


if __name__ == "__main__":
    convert_trace()