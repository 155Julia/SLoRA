import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_ratios(value):
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if item:
            ratios.append(float(item))
    return ratios


def ratio_label(ratio):
    text = f"{ratio:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_eval_log(path, dataset):
    if not path.exists():
        return None
    marker = f"In {dataset}:"
    result = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            result = ast.literal_eval(payload)
    return result


def collect_results(log_root, dataset, modes, ratios):
    rows = []
    for mode in modes:
        for ratio in ratios:
            eval_log = Path(log_root) / mode / f"r{ratio_label(ratio)}" / dataset / "eval.log"
            result = parse_eval_log(eval_log, dataset)
            row = {
                "mode": mode,
                "ratio": ratio,
                "dataset": dataset,
                "eval_log": str(eval_log),
                "status": "ok" if result is not None else "missing",
            }
            if result:
                row.update(result)
            rows.append(row)
    return rows


def write_csv(rows, path):
    metric_keys = sorted({key for row in rows for key in row.keys()} - {"mode", "ratio", "dataset", "eval_log", "status"})
    fieldnames = ["mode", "ratio", "dataset", "status"] + metric_keys + ["eval_log"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def nice_label(mode):
    return {
        "seq_lora": "Seq-LoRA",
        "drop_first": "DropFirst",
        "drop_second": "DropSecond",
        "drop_all": "DropAll",
    }.get(mode, mode)


def write_plot(rows, output_dir, metric):
    ok_rows = [row for row in rows if row.get("status") == "ok" and metric in row]
    if not ok_rows:
        return False

    modes = []
    for row in rows:
        if row["mode"] not in modes:
            modes.append(row["mode"])

    xs = sorted({float(row["ratio"]) for row in ok_rows})
    ys = [float(row[metric]) for row in ok_rows]
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_y, max_y):
        pad = 0.05 if max_y == 0 else abs(max_y) * 0.1
    else:
        pad = (max_y - min_y) * 0.12
    y0 = max(0.0, min_y - pad)
    y1 = min(1.0, max_y + pad) if metric == "accuracy" else max_y + pad
    if math.isclose(y0, y1):
        y1 = y0 + 1.0

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, mode in enumerate(modes):
        mode_rows = sorted(
            [row for row in ok_rows if row["mode"] == mode],
            key=lambda row: float(row["ratio"]),
        )
        if not mode_rows:
            continue
        ax.plot(
            [float(row["ratio"]) for row in mode_rows],
            [float(row[metric]) for row in mode_rows],
            marker=markers[idx % len(markers)],
            linewidth=2.0,
            markersize=5.5,
            label=nice_label(mode),
        )

    ax.set_title(f"Task1 {metric} under LoRA component drop")
    ax.set_xlabel("Drop ratio")
    ax.set_ylabel(metric)
    ax.set_xticks(xs)
    ax.set_ylim(y0, y1)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_dir = Path(output_dir)
    fig.savefig(output_dir / "figure_task1_accuracy.png")
    fig.savefig(output_dir / "figure_task1_accuracy.pdf")
    fig.savefig(output_dir / "figure_task1_accuracy.svg")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Collect motivation smoke eval logs into CSV/JSON/SVG.")
    parser.add_argument("--log-root", required=True, help="Root containing mode/rX/DATASET/eval.log folders.")
    parser.add_argument("--dataset", default="C-STANCE")
    parser.add_argument("--ratios", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--modes", default="seq_lora,drop_first,drop_second,drop_all")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_results(args.log_root, args.dataset, modes, ratios)
    write_csv(rows, output_dir / "results.csv")
    write_json(rows, output_dir / "results.json")
    wrote_plot = write_plot(rows, output_dir, args.metric)

    print(f"Wrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'results.json'}")
    if wrote_plot:
        print(f"Wrote {output_dir / 'figure_task1_accuracy.png'}")
        print(f"Wrote {output_dir / 'figure_task1_accuracy.pdf'}")
        print(f"Wrote {output_dir / 'figure_task1_accuracy.svg'}")
    else:
        print("No complete metric rows found; skipped plot.")


if __name__ == "__main__":
    main()
