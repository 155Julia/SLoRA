# Motivation Smoke Experiment

This folder contains a small two-task pipeline for probing the SLoRA motivation claim with the data already present in this repository.

## Task Choice

The default smoke sequence is:

1. `C-STANCE`
2. `FOMC`

Both are classification tasks evaluated by accuracy in `src/eval/eval_trace.py`. `C-STANCE` is used as Task1 because it has a simple A/B/C output parser, and `FOMC` is used as Task2 because it is another stance-style classification task with a clear domain and language shift. This makes it useful for a quick forgetting check before replacing the data with Amazon -> Yahoo.

## Pipeline

Run from the repository root:

```bash
bash scripts/motivation/run_two_task_seq_lora_smoke.sh
bash scripts/motivation/make_ablation_checkpoints.sh
bash scripts/motivation/eval_task1_ablation_smoke.sh
bash scripts/motivation/collect_and_plot_smoke.sh
```

The scripts write:

- sequential LoRA checkpoint: `checkpoints/motivation-smoke-llama3-cstance-fomc-lora-r8a16`
- ablated checkpoints: `checkpoints/motivation-smoke-llama3-cstance-fomc-lora-r8a16-ablation`
- raw eval logs: `logs/motivation-smoke-llama3-cstance-fomc-lora-r8a16/motivation_ablation`
- summary CSV/JSON/PNG/PDF/SVG: `logs/motivation-smoke-llama3-cstance-fomc-lora-r8a16/motivation_ablation_summary`

## Ablation Modes

- `seq_lora`: keep Task1 and Task2 adapters unchanged.
- `drop_first`: drop LoRA rank components only from the Task1 adapter.
- `drop_second`: drop LoRA rank components only from the Task2 adapter.
- `drop_all`: drop LoRA rank components from both adapters.

Each LoRA update is treated as a sum of rank-one components:

```text
Delta W = B @ A = sum_i B[:, i] A[i, :]
```

The ablation zeroes the matching row of `A` and column of `B` for selected components, preserving tensor shapes and PEFT compatibility. By default, selected components are the lowest-energy components according to `||A[i, :]|| * ||B[:, i]||`. Set `SELECTION=random` to run seeded random component drops.

## Useful Overrides

```bash
MAX_STEPS=20 RATIOS=0,0.1,0.2,0.3,0.4,0.5,0.6 bash scripts/motivation/run_two_task_seq_lora_smoke.sh
RATIOS=0,0.1,0.2,0.3,0.4,0.5,0.6 bash scripts/motivation/make_ablation_checkpoints.sh
RATIOS=0,0.1,0.2,0.3,0.4,0.5,0.6 bash scripts/motivation/eval_task1_ablation_smoke.sh
RATIOS=0,0.1,0.2,0.3,0.4,0.5,0.6 bash scripts/motivation/collect_and_plot_smoke.sh
```
