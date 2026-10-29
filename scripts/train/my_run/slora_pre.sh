#!/bin/bash
set -eo pipefail

TASK_PATH=${TASK_PATH:-data/TRACE-Benchmark/LLM-CL-Benchmark_5000}
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs

# Your config
MODEL_BASE=${MODEL_BASE:-LLMs/Meta-Llama-3.1-8B-Instruct}
MODEL=llama3
ORDER=ORDER1

TRAINING_DATA=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")


MODE=max

R=64
ALPHA=128
if [ -z "${CKPT:-}" ]; then
    CKPT=llama3-8b-lora-r${R}a${ALPHA}
    CKPT=${CKPT}-${MODE}
    CKPT=${ORDER}-${CKPT}
fi
JOB_NAME="${CKPT##*-}"
LOG_DIR=${LOGS_BASE_PATH}/${CKPT}
OUTPUT_DIR=${MODEL_PATH}/${CKPT}

mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
TRAIN_EPOCHS=(5 3 7 5 3 5 5 7)

for idx in "${!TRAINING_DATA[@]}"; do
    ORDER=$((idx + 1))
    TASK_NAME="${TRAINING_DATA[$idx]}"
    TASK_FILE="${TASK_PATH}/${TASK_NAME}/train.json"
    TEMP_OUTPUT_DIR=${OUTPUT_DIR}/order${ORDER}
    TRAIN_EPOCH=${TRAIN_EPOCHS[idx]}
    LOG_FILE="${LOG_DIR}/train_task${ORDER}.log"

    if [ -f "$TEMP_OUTPUT_DIR/$MODE.safetensors" ]; then
        echo "Checkpoint exists for ${TASK_NAME} (order ${ORDER}), skipping..."
        continue
    fi
    if [ -f "$TEMP_OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "Training adapter exists for ${TASK_NAME}, but ${MODE}.safetensors is missing. Finish denoising before resuming." >&2
        exit 1
    fi

    echo "Starting training for ${TASK_NAME} (order ${ORDER})..."

    if ! bash ./scripts/train/my_slurm/slora_pre.sh "$MODEL_BASE" "$MODEL" "$TASK_NAME" "$TASK_FILE" "$TEMP_OUTPUT_DIR" "$ORDER" "$TRAIN_EPOCH" "$MODE" "$R" "$ALPHA" 2>&1 | tee "$LOG_FILE"; then
        echo "ERROR: Training for ${TASK_NAME} failed (order ${ORDER}). Aborting sequence."
        exit 1
    fi
    echo "Completed training for ${TASK_NAME} (order ${ORDER}). Output in ${TEMP_OUTPUT_DIR}"
done

echo "All continuous learning tasks finished."
