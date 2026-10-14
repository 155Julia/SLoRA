import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


MODEL_WEIGHT_FILES = {"adapter_model.safetensors", "adapter_model.bin"}
MODES = ("seq_lora", "drop_first", "drop_second", "drop_all")
SELECTIONS = ("energy_low", "energy_high", "random")


def parse_ratios(value):
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Drop ratio must be in [0, 1], got {ratio}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("At least one drop ratio is required.")
    return ratios


def ratio_label(ratio):
    text = f"{ratio:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def stable_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def copy_adapter_dir(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    for item in source_dir.iterdir():
        if item.name in MODEL_WEIGHT_FILES:
            continue
        if item.is_dir() and item.name.startswith("checkpoint-"):
            continue
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def component_scores(lora_a, lora_b):
    a = lora_a.float().reshape(lora_a.shape[0], -1)
    b = lora_b.float().reshape(lora_b.shape[1], -1)
    return torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)


def choose_drop_indices(name, rank, ratio, lora_a, lora_b, selection, seed):
    drop_count = int(rank * ratio)
    if ratio > 0 and drop_count == 0:
        drop_count = 1
    drop_count = min(drop_count, rank)
    if drop_count == 0:
        return torch.empty(0, dtype=torch.long)

    if selection == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(seed, name))
        return torch.randperm(rank, generator=generator)[:drop_count]

    scores = component_scores(lora_a.cpu(), lora_b.cpu())
    descending = selection == "energy_high"
    return torch.argsort(scores, descending=descending)[:drop_count]


def drop_rank_components(state_dict, ratio, selection, seed):
    if ratio == 0:
        return {key: value.clone().contiguous() for key, value in state_dict.items()}, {}

    ablated = {key: value.clone() for key, value in state_dict.items()}
    dropped = {}

    for key, lora_a in state_dict.items():
        if "lora_A" not in key or not key.endswith("weight"):
            continue
        lora_b_key = key.replace("lora_A", "lora_B")
        if lora_b_key not in state_dict:
            raise KeyError(f"Cannot find matching LoRA B tensor for {key}")

        lora_b = state_dict[lora_b_key]
        rank = lora_a.shape[0]
        if lora_b.shape[1] != rank:
            raise ValueError(f"Rank mismatch for {key}: A rank={rank}, B rank={lora_b.shape[1]}")

        indices = choose_drop_indices(key, rank, ratio, lora_a, lora_b, selection, seed)
        if indices.numel() == 0:
            dropped[key] = []
            continue

        ablated[key][indices.to(ablated[key].device), :] = 0
        ablated[lora_b_key][:, indices.to(ablated[lora_b_key].device)] = 0
        dropped[key] = indices.tolist()

    return {key: value.contiguous() for key, value in ablated.items()}, dropped


def should_drop(mode, order_id):
    if mode == "seq_lora":
        return False
    if mode == "drop_first":
        return order_id == 1
    if mode == "drop_second":
        return order_id == 2
    if mode == "drop_all":
        return True
    raise ValueError(f"Unknown mode: {mode}")


def resolve_order_dirs(checkpoint_root):
    root = Path(checkpoint_root)
    order_dirs = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("order"):
            suffix = path.name.replace("order", "", 1)
            if suffix.isdigit():
                order_dirs.append((int(suffix), path))
    order_dirs.sort(key=lambda item: item[0])
    if not order_dirs:
        raise ValueError(f"No order* directories found under {root}")
    return order_dirs


def build_ablation(source_checkpoint, output_root, ratios, modes, selection, seed):
    order_dirs = resolve_order_dirs(source_checkpoint)
    output_root = Path(output_root)
    manifest = {
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "selection": selection,
        "seed": seed,
        "ratios": ratios,
        "modes": modes,
        "orders": [order_id for order_id, _ in order_dirs],
        "runs": [],
    }

    for mode in modes:
        for ratio in ratios:
            run_root = output_root / mode / f"r{ratio_label(ratio)}"
            for order_id, source_dir in order_dirs:
                target_dir = run_root / f"order{order_id}"
                copy_adapter_dir(source_dir, target_dir)

                source_weight = source_dir / "adapter_model.safetensors"
                if not source_weight.exists():
                    raise FileNotFoundError(f"Missing adapter weights: {source_weight}")

                state_dict = load_file(str(source_weight))
                if should_drop(mode, order_id):
                    new_state_dict, dropped = drop_rank_components(state_dict, ratio, selection, seed)
                else:
                    new_state_dict = {key: value.clone().contiguous() for key, value in state_dict.items()}
                    dropped = {}
                save_file(new_state_dict, str(target_dir / "adapter_model.safetensors"))

                with open(target_dir / "ablation_meta.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mode": mode,
                            "ratio": ratio,
                            "order": order_id,
                            "source_dir": str(source_dir),
                            "selection": selection,
                            "seed": seed,
                            "dropped_rank_indices": dropped,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            manifest["runs"].append({"mode": mode, "ratio": ratio, "path": str(run_root)})

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create rank-component LoRA ablation checkpoints.")
    parser.add_argument("--source-checkpoint", required=True, help="Checkpoint root containing order1/order2/... dirs.")
    parser.add_argument("--output-root", required=True, help="Where ablated checkpoint trees will be written.")
    parser.add_argument("--ratios", default="0,0.1,0.2,0.3,0.4,0.5", help="Comma-separated drop ratios.")
    parser.add_argument("--modes", default="seq_lora,drop_first,drop_second,drop_all", help="Comma-separated modes.")
    parser.add_argument("--selection", choices=SELECTIONS, default="energy_low")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = sorted(set(modes) - set(MODES))
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Valid modes: {MODES}")

    build_ablation(args.source_checkpoint, args.output_root, ratios, modes, args.selection, args.seed)


if __name__ == "__main__":
    main()
