#!/bin/bash
set -e

# Your config
TASK_PATH=data/TRACE-Benchmark/LLM-CL-Benchmark_500
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
DATASET=C-STANCE
MODEL=llama3
CONV_MODE=llama3
CKPT=smoke-${MODEL}-lora-r8a16
MAX_NEW_TOKENS=128
PYTHON_BIN=/share/home/jiayu/miniconda3/envs/slora/bin/python

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

QUESTION_FILE="${TASK_PATH}/${DATASET}/test.json"
MODEL_DIR="${MODEL_PATH}/${CKPT}"
OUT_DIR="${LOGS_BASE_PATH}/${CKPT}/eval/order1/${DATASET}"
ANSWERS_FILE="${OUT_DIR}/infer.jsonl"
WRONG_FILE="${OUT_DIR}/wrong.jsonl"

mkdir -p "${OUT_DIR}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_BASE=${MODEL_BASE}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] QUESTION_FILE=${QUESTION_FILE}"
echo "[INFO] ANSWERS_FILE=${ANSWERS_FILE}"

"${PYTHON_BIN}" -u -m src.eval.model_diverse_gen_batch \
    --model-path "${MODEL_DIR}" \
    --model-base "${MODEL_BASE}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${ANSWERS_FILE}" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}" \
    --denoised_mode "seq_lora" \
    --test_order 1 \
    --batch_size 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --resume

"${PYTHON_BIN}" -u -m src.eval.eval_trace \
    --input_file "${ANSWERS_FILE}" \
    --output_file "${WRONG_FILE}"