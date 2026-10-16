#!/usr/bin/env bash
# Shared stage runner; old script names below are compatibility entry points.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

run() {
    if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

train() {
    case "$PRECISION" in
        bf16) precision_args=(--bf16 True --fp16 False) ;;
        fp16) precision_args=(--bf16 False --fp16 True) ;;
        *) echo "PRECISION must be bf16 or fp16" >&2; exit 1 ;;
    esac
    if [[ "$DRY_RUN" != 1 ]]; then
        for task in "$TASK1" "$TASK2"; do
            [[ -f "$TASK_PATH/$task/train.json" ]] || { echo "Missing data: $task" >&2; exit 1; }
        done
        # A launch can fail before any adapter is written (for example, an unsupported
        # precision). Permit that metadata-only directory to be reused, but never mix
        # with a completed or partially saved adapter.
        if [[ -e "$SOURCE_CHECKPOINT" ]] && find "$SOURCE_CHECKPOINT" -type f -name adapter_model.safetensors -print -quit | grep -q .; then
            echo "Use a fresh SOURCE_CHECKPOINT/CKPT; refusing to mix training runs." >&2
            exit 1
        fi
        mkdir -p "$SOURCE_CHECKPOINT" "$LOG_ROOT/train"
        # Save resolved settings alongside the adapters, including overrides.
        declare -p TASK_PATH TASK1 TASK2 MODEL_BASE MODEL LORA_R LORA_ALPHA TRAIN_EPOCHS MAX_STEPS LEARNING_RATE WARMUP_RATIO SEED PRECISION GPUS_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_SEQ_LENGTH > "$SOURCE_CHECKPOINT/run_config.txt"
        [[ ! -f "$TASK_PATH/manifest.json" ]] || cp "$TASK_PATH/manifest.json" "$SOURCE_CHECKPOINT/data_manifest.json"
    fi
    tasks=("$TASK1" "$TASK2")
    for idx in 0 1; do
        order=$((idx + 1))
        cmd=("$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node "$GPUS_PER_NODE"
            --standalone --nnodes 1 src/train/cl_train.py
            "${precision_args[@]}" --use_peft True --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA"
            --deepspeed scripts/zero2.json --model_name_or_path "$MODEL_BASE" --model "$MODEL"
            --dataset_name "${tasks[$idx]}" --train_data_path "$TASK_PATH/${tasks[$idx]}/train.json"
            --output_dir "$SOURCE_CHECKPOINT/order$order" --task_id "$order"
            --num_train_epochs "$TRAIN_EPOCHS" --max_steps "$MAX_STEPS"
            --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
            --evaluation_strategy no --save_strategy no --learning_rate "$LEARNING_RATE"
            --optim adamw_torch --weight_decay 0 --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type cosine
            --seed "$SEED" --max_seq_length "$MAX_SEQ_LENGTH"
            --logging_steps 10 --gradient_checkpointing True --report_to none)
        if [[ "$DRY_RUN" == 1 ]]; then run "${cmd[@]}"; else
            "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/train/order$order.log"
        fi
    done
}

ablate() {
    run "$PYTHON_BIN" -m src.motivation.make_lora_ablation_checkpoints \
        --source-checkpoint "$SOURCE_CHECKPOINT" --output-root "$ABLATION_ROOT" \
        --ratios "$RATIOS" --modes "$MODES" --selection "$SELECTION" --seed "$SEED"
}

evaluate() {
    # Manifest is the source of truth, so each unique checkpoint is evaluated once.
    runs=$("$PYTHON_BIN" -c 'import json,sys; m=json.load(open(sys.argv[1])); [print(r["mode"],format(r["ratio"],".12g"),r["path"],sep="\t") for r in m["runs"]]' "$ABLATION_ROOT/manifest.json")
    while IFS=$'\t' read -r mode ratio checkpoint; do
        out="$LOG_ROOT/$mode/r$ratio/$TASK1"
        if [[ "$DRY_RUN" != 1 ]]; then
            [[ ! -e "$out/infer.jsonl" ]] || { echo "Existing predictions: $out; choose a fresh LOG_ROOT." >&2; exit 1; }
            mkdir -p "$out"
            declare -p TASK_PATH TASK1 MODEL_BASE CONV_MODE EVAL_MODULE EVAL_BATCH_SIZE MAX_NEW_TOKENS > "$out/eval_config.txt"
        fi
        cmd=("$PYTHON_BIN" -u -m src.eval.model_diverse_gen_batch
            --model-path "$checkpoint" --model-base "$MODEL_BASE"
            --question-file "$TASK_PATH/$TASK1/test.json" --answers-file "$out/infer.jsonl"
            --temperature 0 --conv-mode "$CONV_MODE" --denoised_mode seq_lora
            --test_order 2 --batch_size "$EVAL_BATCH_SIZE" --max_new_tokens "$MAX_NEW_TOKENS")
        score=("$PYTHON_BIN" -u -m "$EVAL_MODULE" --input_file "$out/infer.jsonl" --output_file "$out/wrong.json")
        if [[ "$DRY_RUN" == 1 ]]; then run "${cmd[@]}"; run "${score[@]}"; else
            "${cmd[@]}" > "$out/infer.log" 2>&1
            "${score[@]}" > "$out/eval.log" 2>&1
        fi
    done <<< "$runs"
}

collect() {
    run "$PYTHON_BIN" -m src.motivation.collect_smoke_results --log-root "$LOG_ROOT" \
        --manifest "$ABLATION_ROOT/manifest.json" \
        --dataset "$TASK1" --ratios "$RATIOS" --modes "$MODES" --metric accuracy --output-dir "$OUTPUT_DIR"
}

case "${1:-}" in
    train) train ;; ablate) ablate ;; eval) evaluate ;; collect) collect ;;
    all) train; ablate; evaluate; collect ;;
    *) echo "Usage: bash scripts/motivation/run.sh {train|ablate|eval|collect|all}" >&2; exit 2 ;;
esac
