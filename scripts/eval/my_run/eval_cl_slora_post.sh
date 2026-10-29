#!/bin/bash
set -e

TASK_PATH=${TASK_PATH:-data/TRACE-Benchmark/LLM-CL-Benchmark_5000}
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs


# your config
domains=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")
ORDER=ORDER1

# slora_post
if [ -z "${CKPT:-}" ]; then
    CKPT=llama3-8b-lora-r64a128
    CKPT=${ORDER}-${CKPT}
fi

if [[ "$CKPT" =~ "llama2" ]]; then
    CONV_MODE=llama2
elif [[ "$CKPT" =~ "llama3" ]]; then
    CONV_MODE=llama3
elif [[ "$CKPT" =~ "qwen1.5" ]]; then
    CONV_MODE=qwen
else
    CONV_MODE=qwen
fi


MODEL_BASE=${MODEL_BASE:-LLMs/Meta-Llama-3.1-8B-Instruct}

# slora_post
DENOISED_MODE=${DENOISED_MODE:-slora_post}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MAX_TEST_ORDER=8


for domain in "${domains[@]}"; do
    TEMP_OUTPUT_DIR="$LOGS_BASE_PATH/$CKPT/$DENOISED_MODE/order${MAX_TEST_ORDER}"

    if [ -f "$TEMP_OUTPUT_DIR/$domain/eval.log" ] && grep -q "^In $domain:" "$TEMP_OUTPUT_DIR/$domain/eval.log"; then
        echo "Evaluation exists for ${domain}, skipping..."
        continue
    fi

    echo "====== Processing domain: ${domain} ======"

    if ! bash ./scripts/eval/my_slurm/eval_cl.sh \
        "$TASK_PATH" "$MODEL_BASE" "$MODEL_PATH" "$CKPT" "$LOGS_BASE_PATH" \
        "$domain" "$CONV_MODE" "$DENOISED_MODE" "$MAX_TEST_ORDER" ; then
        echo "!!!! Eval failed for ${domain}."
        exit 1
    fi

    echo "====== Done: ${domain} ======"
done



# for idx in "${!domains[@]}"; do
#     domain="${domains[$idx]}"

#     # 第1个任务测1~8，第2个任务测2~8，第3个任务测3~8，第4个任务测4~8
#     start_order=$((idx + 1))

#     for ((test_order=start_order; test_order<=MAX_TEST_ORDER; test_order++)); do
#         TEMP_OUTPUT_DIR="$LOGS_BASE_PATH/$CKPT/$DROP_MODE/order${test_order}"

#         if [ -f "$TEMP_OUTPUT_DIR/$domain/eval.log" ]; then
#             echo "Evaluation exists for ${domain}, TEST_ORDER=${test_order}, skipping..."
#             continue
#         fi

#         echo "====== Processing domain: ${domain} (TEST_ORDER=${test_order}) ======"

#         if ! bash ./scripts/eval/my_slurm/eval_cl.sh \
#             "$TASK_PATH" "$MODEL_BASE" "$MODEL_PATH" "$CKPT" "$LOGS_BASE_PATH" \
#             "$domain" "$CONV_MODE" "$DENOISED_MODE" "$test_order"; then
#             echo "!!!! Eval failed for ${domain}, TEST_ORDER=${test_order}), continuing to next."
#             continue
#         fi

#         echo "====== Done: ${domain}, TEST_ORDER=${test_order} ======"
#     done
# done

echo "All domains completed."
