#!/bin/bash
set -e

TASK_PATH=data/TRACE-Benchmark/LLM-CL-Benchmark_500
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python
CKPT=motivation-smoke-llama3-cstance-fomc-lora-r8a16
ABLATION_ROOT=checkpoints/$CKPT-ablation
LOG_ROOT=logs/$CKPT/motivation_ablation

TASK1=C-STANCE
CONV_MODE=llama3
MAX_NEW_TOKENS=16
TEST_ORDER=2
RATIOS=(0 0.1 0.2 0.3 0.4 0.5)
MODES=(seq_lora drop_first drop_second drop_all)

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

for MODE_NAME in "${MODES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        RUN_CKPT=$ABLATION_ROOT/$MODE_NAME/r$RATIO
        OUT_DIR=$LOG_ROOT/$MODE_NAME/r$RATIO/$TASK1
        ANSWERS_FILE=$OUT_DIR/infer.jsonl
        WRONG_FILE=$OUT_DIR/wrong.jsonl

        if [ ! -d "$RUN_CKPT" ]; then
            echo "[ERROR] Missing checkpoint tree: $RUN_CKPT" >&2
            exit 1
        fi

        mkdir -p "$OUT_DIR"
        echo "[RUN] mode=$MODE_NAME, ratio=$RATIO, eval=$TASK1"

        "$PYTHON_BIN" -u -m src.eval.model_diverse_gen_batch \
            --model-path "$RUN_CKPT" \
            --model-base "$MODEL_BASE" \
            --question-file "$TASK_PATH/$TASK1/test.json" \
            --answers-file "$ANSWERS_FILE" \
            --temperature 0 \
            --conv-mode "$CONV_MODE" \
            --denoised_mode "seq_lora" \
            --test_order "$TEST_ORDER" \
            --batch_size 1 \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --resume \
            > "$OUT_DIR/infer.log" 2>&1

        "$PYTHON_BIN" -u -m src.eval.eval_trace \
            --input_file "$ANSWERS_FILE" \
            --output_file "$WRONG_FILE" \
            > "$OUT_DIR/eval.log" 2>&1
    done
done

echo "[DONE] Evaluation logs written to $LOG_ROOT"
