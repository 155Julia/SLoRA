#!/bin/bash
set -e

PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python
CKPT=motivation-smoke-llama3-cstance-fomc-lora-r8a16
LOG_ROOT=logs/$CKPT/motivation_ablation
OUTPUT_DIR=logs/$CKPT/motivation_ablation_summary
TASK1=C-STANCE
RATIOS=0,0.1,0.2,0.3,0.4,0.5
MODES=seq_lora,drop_first,drop_second,drop_all

export PYTHONPATH="$PWD:$PYTHONPATH"

"$PYTHON_BIN" -m src.motivation.collect_smoke_results \
    --log-root "$LOG_ROOT" \
    --dataset "$TASK1" \
    --ratios "$RATIOS" \
    --modes "$MODES" \
    --metric accuracy \
    --output-dir "$OUTPUT_DIR"
