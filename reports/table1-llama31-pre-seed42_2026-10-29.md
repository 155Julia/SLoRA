# TRACE evaluation summary — slora_pre/order8

Input settings: MAX_INPUT_TOKENS=8192, TRUNCATION_SIDE=left.

Scores are on the 0–100 scale used by Table 1. Accuracy and ROUGE-L are multiplied by 100; similarity and SARI are already on that scale.

只有超過 8,192 tokens 的單筆輸入才會被截斷。程式對每筆 prompt 設定 max_length=8192；長度未超過上限的內容會完整保留。TRUNCATION_SIDE=left 只決定超過上限時刪掉開頭、保留結尾。

| Task | Score | Generated / expected | Score source |
| --- | ---: | ---: | --- |
| C-STANCE | 51.75 | 2000 / 2000 | eval.log |
| FOMC | 64.52 | 496 / 496 | eval.log |
| MeetingBank | 50.05 | 692 / 692 | eval.log |
| Py150 | 53.36 | 2000 / 2000 | eval.log |
| ScienceQA | 92.15 | 2000 / 2000 | eval.log |
| NumGLUE-cm | 67.90 | 81 / 81 | eval.log |
| NumGLUE-ds | 72.92 | 325 / 325 | eval.log |
| 20Minuten | 41.54 | 200 / 200 | eval_retry.log |

**Average: 61.77**
