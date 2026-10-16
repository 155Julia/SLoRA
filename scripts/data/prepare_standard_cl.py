"""Convert the pinned O-LoRA Amazon/Yahoo splits without resampling them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REVISION = "07117e1fc4a5f5ad9308a815a42cee8f46502dc8"
TASKS = {"amazon": ("SC", "sentiment", 5), "yahoo": ("TC", "topic", 10)}
SHA256 = {
    "amazon/train": "a77b7a84773ea16b2906c16eb4de05ea87f529b91d5e3209b43060306e02c05f",
    "amazon/test": "0e3337b659505456dd62089f4099778043d409e54f6e9cacf35154e783fdb6a2",
    "amazon/labels": "dfe3dbf501f9643d7f3fa24df096f925fb149fb299c1c088f802aaf1ba79050b",
    "yahoo/train": "1debade5e795f4a3d0ffadda47d2ffce9f8ce730974e6f90a11c832a442718fd",
    "yahoo/test": "ef13a28c0e9980197f20fbdf74e5ee3ca7d24914509bd471ca7044abcc1c692b",
    "yahoo/labels": "6cc0d1cc3efe693ab76778216b04c51728ca9a305a3d0b4128a88b965f7003d4",
}


def source_url(task, split):
    return f"https://raw.githubusercontent.com/cmnfriend/O-LoRA/{REVISION}/CL_Benchmark/{TASKS[task][0]}/{task}/{split}.json"


def read_source(root, task, split, download=False):
    path = root / task / f"{split}.json"
    if path.exists():
        raw = path.read_bytes()
    elif download:
        with urlopen(source_url(task, split), timeout=60) as response:
            raw = response.read()
    else:
        raise FileNotFoundError(f"{path}: use --download to fetch the pinned O-LoRA files")
    if hashlib.sha256(raw).hexdigest() != SHA256[f"{task}/{split}"]:
        raise ValueError(f"{path}: checksum differs from pinned O-LoRA revision")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return raw


def convert(rows, labels, kind):
    instruction = (
        f"What is the {kind} of the following paragraph? Choose one from the option.\n"
        + "Option: " + ", ".join(labels) + " \n"
    )
    result = []
    for row in rows:
        label = row["label"]
        index = labels.index(label)  # Fail on unknown labels instead of silently remapping.
        if not isinstance(row["sentence"], str) or not row["sentence"].strip():
            raise ValueError("Empty or non-text input")
        result.append({
            "conversations": [
                {"value": instruction + row["sentence"] + "\nAnswer:"},
                {"value": label},
            ],
            "eval": {"answer": label, "answer_idx": index},
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/standard_cl/raw/olora"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/standard_cl/olora"))
    parser.add_argument("--download", action="store_true", help="Download missing pinned source files")
    args = parser.parse_args()
    manifest = {"source": "https://github.com/cmnfriend/O-LoRA", "revision": REVISION,
                "sampling": "all rows, original order", "splits": {}}
    for task, (_, kind, count) in TASKS.items():
        labels = json.loads(read_source(args.raw_dir, task, "labels", args.download))
        if len(labels) != count or len(set(labels)) != count:
            raise ValueError(f"{task}: expected {count} distinct labels")
        for split in ("train", "test"):
            raw = read_source(args.raw_dir, task, split, args.download)
            rows = json.loads(raw)
            converted = convert(rows, labels, kind)
            target = args.output_dir / task / f"{split}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n")
            manifest["splits"][f"{task}/{split}"] = {
                "url": source_url(task, split),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "labels": labels, "label_counts": dict(Counter(row["label"] for row in rows)),
                "count": len(rows),
            }
            print(f"{target}: {len(rows)} examples, {count} classes")
    args.output_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
