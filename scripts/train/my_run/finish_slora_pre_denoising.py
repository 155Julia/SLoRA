"""Finish SLoRA-Pre denoising from an already saved training adapter."""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from src.train.cl_train_slora import denoising


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    adapter_path = checkpoint / "adapter_model.safetensors"
    output_path = checkpoint / "max.safetensors"
    if not adapter_path.is_file():
        parser.error(f"Missing trained adapter: {adapter_path}")
    if output_path.exists():
        parser.error(f"Denoised checkpoint already exists: {output_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    state_dict = load_file(str(adapter_path), device="cuda:0")
    with torch.no_grad():
        denoised = denoising(base_model, state_dict, "max")
    temporary_path = checkpoint / ".max.safetensors.tmp"
    save_file({key: value.cpu() for key, value in denoised.items()}, str(temporary_path))
    os.replace(temporary_path, output_path)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
