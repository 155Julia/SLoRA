#!/bin/bash
set -e

# Your config
TASK_PATH=data/TRACE-Benchmark/LLM-CL-Benchmark_500
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
MODEL=llama3
DATASET=C-STANCE
PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python

export CUDA_VISIBLE_DEVICES=0
GPUS_PER_NODE=1

LORA_R=8
LORA_ALPHA=16
MAX_STEPS=10
TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
PRECISION=fp16

CKPT=smoke-${MODEL}-lora-r${LORA_R}a${LORA_ALPHA}

if [ "${PRECISION}" = "bf16" ]; then
    PRECISION_ARGS=(--bf16 True --fp16 False)
elif [ "${PRECISION}" = "fp16" ]; then
    PRECISION_ARGS=(--bf16 False --fp16 True)
else
    echo "[ERROR] PRECISION must be either fp16 or bf16, got: ${PRECISION}" >&2
    exit 1
fi

export LOGLEVEL=INFO
export PYTHONPATH="$PWD:$PYTHONPATH"

MASTER_PORT=$((RANDOM % 101 + 20000))
MASTER_ADDR=127.0.0.1

TASK_FILE="${TASK_PATH}/${DATASET}/train.json"
OUTPUT_DIR="${MODEL_PATH}/${CKPT}/order1"
LOG_DIR="${LOGS_BASE_PATH}/${CKPT}"
LOG_FILE="${LOG_DIR}/train_${DATASET}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_BASE=${MODEL_BASE}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] TASK_FILE=${TASK_FILE}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] GPUS_PER_NODE=${GPUS_PER_NODE}, MAX_STEPS=${MAX_STEPS}, PRECISION=${PRECISION}"

"${PYTHON_BIN}" -u -m torch.distributed.run \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --nnodes 1 \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend c10d \
    --node_rank 0 \
    src/train/cl_train.py \
    "${PRECISION_ARGS[@]}" \
    --use_peft True \
    --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "${MODEL_BASE}" \
    --model "${MODEL}" \
    --dataset_name "${DATASET}" \
    --train_data_path "${TASK_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${MAX_STEPS}" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --report_to none \
    --task_id 1 \
    2>&1 | tee "${LOG_FILE}"