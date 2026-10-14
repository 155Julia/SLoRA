#!/bin/bash
set -e

TASK_PATH=data/TRACE-Benchmark/LLM-CL-Benchmark_500
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
MODEL=llama3
PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"
export LOGLEVEL=INFO

GPUS_PER_NODE=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=$((RANDOM % 101 + 20000))

LORA_R=8
LORA_ALPHA=16
MAX_STEPS=10
TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
PRECISION=fp16
CKPT=motivation-smoke-llama3-cstance-fomc-lora-r8a16

TRAINING_DATA=("C-STANCE" "FOMC")

if [ "$PRECISION" = "bf16" ]; then
    PRECISION_ARGS=(--bf16 True --fp16 False)
elif [ "$PRECISION" = "fp16" ]; then
    PRECISION_ARGS=(--bf16 False --fp16 True)
else
    echo "[ERROR] PRECISION must be either fp16 or bf16, got: $PRECISION" >&2
    exit 1
fi

OUTPUT_ROOT=$MODEL_PATH/$CKPT
LOG_DIR=$LOGS_BASE_PATH/$CKPT/train
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "[INFO] Task sequence: ${TRAINING_DATA[*]}"
echo "[INFO] MODEL_BASE=$MODEL_BASE"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[INFO] MAX_STEPS=$MAX_STEPS, LORA_R=$LORA_R, LORA_ALPHA=$LORA_ALPHA"

for idx in "${!TRAINING_DATA[@]}"; do
    ORDER=$((idx + 1))
    TASK_NAME="${TRAINING_DATA[$idx]}"
    TASK_FILE=$TASK_PATH/$TASK_NAME/train.json
    TEMP_OUTPUT_DIR=$OUTPUT_ROOT/order$ORDER
    LOG_FILE=$LOG_DIR/train_order${ORDER}_${TASK_NAME}.log

    if [ -f "$TEMP_OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "[SKIP] $TEMP_OUTPUT_DIR/adapter_model.safetensors exists."
        continue
    fi

    mkdir -p "$TEMP_OUTPUT_DIR"
    echo "[RUN] Training order$ORDER: $TASK_NAME"

    "$PYTHON_BIN" -u -m torch.distributed.run \
        --nproc_per_node "$GPUS_PER_NODE" \
        --nnodes 1 \
        --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
        --rdzv_backend c10d \
        --node_rank 0 \
        src/train/cl_train.py \
        "${PRECISION_ARGS[@]}" \
        --use_peft True \
        --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MODEL_BASE" \
        --model "$MODEL" \
        --dataset_name "$TASK_NAME" \
        --train_data_path "$TASK_FILE" \
        --output_dir "$TEMP_OUTPUT_DIR" \
        --num_train_epochs "$TRAIN_EPOCHS" \
        --max_steps "$MAX_STEPS" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps "$MAX_STEPS" \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --gradient_checkpointing True \
        --report_to none \
        --task_id "$ORDER" \
        2>&1 | tee "$LOG_FILE"
done

echo "[DONE] Sequential LoRA smoke checkpoint: $OUTPUT_ROOT"
