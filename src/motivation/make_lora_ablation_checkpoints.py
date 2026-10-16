import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from src.motivation.common import MODES, parse_ratios, ratio_label, unique_runs

SELECTIONS = ("svd_low", "energy_low", "energy_high", "random")


def stable_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def component_scores(lora_a, lora_b):
    a = lora_a.float().reshape(lora_a.shape[0], -1)
    b = lora_b.float().T  # Each row is B[:, i], not a reshaped memory chunk.
    return torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)


def choose_drop_indices(name, rank, ratio, lora_a, lora_b, selection, seed):
    drop_count = int(rank * ratio + 1e-9)
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


def truncate_svd(lora_a, lora_b, drop_count):
    """Exact SVD of BA via a rank-sized core; keep shapes and PEFT scaling."""
    rank = lora_a.shape[0]
    # BA = Qb (Rb Ra.T) Qa.T. No dense model-sized update is materialized.
    qb, rb = torch.linalg.qr(lora_b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(lora_a.float().T, mode="reduced")
    u, sigma, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    keep = min(rank - drop_count, len(sigma))
    a = torch.zeros_like(lora_a)
    b = torch.zeros_like(lora_b)
    a[:keep] = (vh[:keep] @ qa.T).to(a.dtype)
    b[:, :keep] = ((qb @ u[:, :keep]) * sigma[:keep]).to(b.dtype)
    return a, b


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

        if selection == "svd_low":
            drop_count = int(rank * ratio + 1e-9)
            if drop_count:
                ablated[key], ablated[lora_b_key] = truncate_svd(lora_a, lora_b, drop_count)
            dropped[key] = list(range(rank - drop_count, rank))
            continue

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
    if [order for order, _ in order_dirs] != [1, 2]:
        raise ValueError("The motivation experiment requires exactly order1 and order2")
    runs = list(unique_runs(modes, ratios))
    output_root = Path(output_root).resolve()
    # Never silently replace old experiments (or reuse their inference results).
    if output_root.exists():
        raise FileExistsError(f"Choose a fresh output root: {output_root}")
    for _, source in order_dirs:
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
    output_root.mkdir(parents=True)
    manifest = {
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "selection": selection, "seed": seed, "ratios": ratios, "modes": modes,
        "rounding": "floor(rank * ratio); actual drop count recorded per tensor",
        "runs": [],
    }
    modified_adapters = {}
    for mode, ratio in runs:
        run_root = output_root / mode / f"r{ratio_label(ratio)}"
        run_root.mkdir(parents=True)
        for order_id, source in order_dirs:
            source = source.resolve()
            target = run_root / f"order{order_id}"
            if not should_drop(mode, order_id):
                target.symlink_to(source, target_is_directory=True)
                continue
            cache_key = (order_id, ratio)
            if cache_key in modified_adapters:
                target.symlink_to(modified_adapters[cache_key], target_is_directory=True)
                continue
            target.mkdir()
            # Only model weights/config are needed; tokenizer is loaded from the base.
            (target / "adapter_config.json").write_bytes((source / "adapter_config.json").read_bytes())
            state = load_file(str(source / "adapter_model.safetensors"))
            ablated, dropped = drop_rank_components(state, ratio, selection, seed)
            save_file(ablated, str(target / "adapter_model.safetensors"))
            metadata = {"ratio": ratio, "order": order_id,
                        "selection": selection, "seed": seed,
                        "index_basis": "descending singular values" if selection == "svd_low" else "original A/B components",
                        "dropped_rank_indices": dropped}
            (target / "ablation_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
            modified_adapters[cache_key] = target
        manifest["runs"].append({"mode": mode, "ratio": ratio, "path": str(run_root)})
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create rank-component LoRA ablation checkpoints.")
    parser.add_argument("--source-checkpoint", required=True, help="Checkpoint root containing order1/order2/... dirs.")
    parser.add_argument("--output-root", required=True, help="Where ablated checkpoint trees will be written.")
    parser.add_argument("--ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="Comma-separated drop ratios.")
    parser.add_argument("--modes", default="seq_lora,drop_first,drop_second,drop_all", help="Comma-separated modes.")
    parser.add_argument("--selection", choices=SELECTIONS, default="svd_low")
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
