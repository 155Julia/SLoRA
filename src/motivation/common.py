"""Shared experiment grid; the unmodified model is evaluated only once."""
import math

MODES = ("seq_lora", "drop_first", "drop_second", "drop_all")


def parse_ratios(value):
    ratios = list(dict.fromkeys(float(x.strip()) for x in value.split(",") if x.strip()))
    if not ratios or any(not math.isfinite(x) or not 0 <= x <= 1 for x in ratios):
        raise ValueError("Drop ratios must be finite numbers in [0, 1]")
    return ratios


def ratio_label(ratio):
    return format(ratio, ".12g")


def unique_runs(modes, ratios):
    if not modes or set(modes) - set(MODES):
        raise ValueError(f"Modes must be selected from {MODES}")
    # Zero-drop runs and every seq_lora ratio share this single baseline.
    yield "seq_lora", 0.0
    for mode in dict.fromkeys(modes):
        if mode != "seq_lora":
            for ratio in ratios:
                if ratio != 0:
                    yield mode, ratio
