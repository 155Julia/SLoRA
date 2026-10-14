#!/bin/bash
set -e

PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python
CKPT=motivation-smoke-llama3-cstance-fomc-lora-r8a16
SOURCE_CHECKPOINT=checkpoints/$CKPT
OUTPUT_ROOT=checkpoints/$CKPT-ablation
RATIOS=0,0.1,0.2,0.3,0.4,0.5
MODES=seq_lora,drop_first,drop_second,drop_all
SELECTION=energy_low
SEED=2026

export PYTHONPATH="$PWD:$PYTHONPATH"

"$PYTHON_BIN" -m src.motivation.make_lora_ablation_checkpoints \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --output-root "$OUTPUT_ROOT" \
    --ratios "$RATIOS" \
    --modes "$MODES" \
    --selection "$SELECTION" \
    --seed "$SEED"

echo "[DONE] Ablation checkpoints written to $OUTPUT_ROOT"
