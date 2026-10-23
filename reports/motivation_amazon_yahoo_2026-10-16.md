# SLoRA 動機實驗：Amazon Reviews → Yahoo Answers

## 可直接用於報告的摘要

本實驗檢查 SLoRA 論文 Figure 1 的動機：第二個任務的 LoRA 更新，是否含有會干擾第一個任務的成分。使用 Llama3-8B-Instruct，依序以 Amazon Reviews（五類情緒分類）及 Yahoo Answers（十類主題分類）各訓練一個 epoch；在兩個任務的 LoRA 更新完成後，按比例移除指定任務更新中奇異值最小的成分，再用同一份 7,600 題 Amazon 測試集評分。

不刪除成分時，Amazon 準確率為 **63.053%（4,792/7,600）**。移除第二個任務 90% 的低奇異值成分後，準確率為 **64.158%（4,876/7,600）**，增加 **1.105 個百分點、淨多答對 84 題**；移除第一個任務 90% 時為 **61.829%（4,699/7,600）**，下降 **1.224 個百分點**。方向上與論文的動機觀察一致，但本次提升主要出現在高刪除比例，幅度有限，不能宣稱精確重現原圖，也不能由此單獨證明哪些成分是「雜訊」。[SLoRA 原論文，Figure 1、§5.1、附錄 B.3](https://aclanthology.org/2026.acl-long.247.pdf)

## 要報告的實驗設定

| 項目 | 本次實際設定 |
| --- | --- |
| 問題及任務順序 | Task 1：Amazon Reviews 情緒分類；Task 2：Yahoo Answers 主題分類；最後評估 Amazon Task 1。這是兩任務動機實驗，不是論文標準 CL 主實驗的四任務順序。 |
| 資料來源 | [O-LoRA 官方儲存庫](https://github.com/cmnfriend/O-LoRA) 的 `CL_Benchmark/SC/amazon`、`CL_Benchmark/TC/yahoo`，固定 revision `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`；使用原始全部資料及順序，來源 SHA-256 與輸出 SHA-256 記於 `data/standard_cl/olora/manifest.json`。 |
| 資料量 | Amazon：訓練 5,000、測試 7,600、5 類；Yahoo：訓練 10,000、測試 7,600、10 類。Yahoo test 這次沒有評分。 |
| 格式與提示 | 使用 O-LoRA 風格的任務指示及完整選項列表；訓練資料為使用者問題與標準答案的對話格式，推論用 Llama3 chat template。測試答案由標籤文字與零起算類別索引比對。 |
| 基底模型 | 本地 `LLMs/Meta-Llama-3-8B-Instruct`。 |
| 順序訓練 | 原專案 `src/train/cl_train.py`：訓練 Task 1 adapter，將其合併至基底模型，再訓練 Task 2 adapter；推論時依序載入並合併兩個 adapter。Seq-LoRA 表示完全不刪除兩個 adapter。 |
| LoRA | 每個任務 rank 64，α=128，adapter 設定檔顯示 dropout 0.05；作用於每層 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`up_proj`、`gate_proj`、`down_proj`，每個任務共 224 個 LoRA 模組。 |
| 訓練超參數 | 每任務 1 epoch、`max_steps=-1`、AdamW、學習率 2×10⁻⁴、3% warmup、cosine 排程、weight decay 0、最大序列長 1,024、gradient checkpointing、DeepSpeed ZeRO-2；seed 2026。有效 batch 約 32（單卡 micro batch 1 × accumulation 32）。 |
| 訓練實況 | Amazon 156 optimizer steps、訓練 1,474 秒、最終記錄 train loss 2.023；Yahoo 312 steps、3,381 秒、train loss 1.383。不同任務的 loss 不宜直接作性能比較。 |
| 運算環境 | 使用一張 Quadro RTX 8000（48 GB），FP16。機器上有兩張卡，但此次 `CUDA_VISIBLE_DEVICES=0`、`GPUS_PER_NODE=1`。環境記錄：Python 3.11.16、PyTorch 2.2.2+cu121、Transformers 4.40.2、PEFT 0.10.0、TRL 0.8.6、DeepSpeed 0.14.4。 |
| 移除方式 | 每個 LoRA 模組將更新矩陣 `BA` 作 SVD，按奇異值由小到大移除；刪除數量是 `floor(64 × ratio)`。例如 50% 刪 32/64，90% 刪 57/64，留下 7。保留原 adapter 形狀與 PEFT α/r 縮放。比例是成分數比例，不代表參數位元組縮減 90%。 |
| 比較條件 | Seq-LoRA：不刪；DropFirst：只刪 Amazon adapter；DropSecond：只刪 Yahoo adapter；DropAll：兩個都刪。比例 10% 至 90%，每 10% 一格。共 1 個 baseline + 27 個刪除設定；圖上 baseline 重複畫成水平線，並非訓練九次。 |
| 推論與指標 | Amazon test 全部 7,600 題；temperature 0（greedy）、generation batch 1、最多生成 32 tokens；輸出轉成類別文字或索引後，計算逐題 exact-match accuracy。全部條件共用同一組測試題。 |

本次資料與部分訓練設定依照論文 §5.1、附錄 B.3 的公開資訊；單卡 RTX 8000、FP16、seed、batch、序列長、生成長度與**固定比例低奇異值截斷**是本地具體化設定。論文的動機實驗只寫「移除較不重要的 LoRA 更新子空間成分」，沒有提供足以核對的 Figure 1 成分排序公式或資料樣本索引。因此不要把本次截斷稱為作者原始實作，也不要把它與 SLoRA 正式方法中參考基底模型子空間的自適應去噪混為一談。[SLoRA 原論文，§4.2、§5.1、附錄 B.3](https://aclanthology.org/2026.acl-long.247.pdf)

## 完整數值結果

下表單位為 **Amazon accuracy（%）**；括號是相對 Seq-LoRA 的百分點變化。每格分母都是 7,600 題。

| Drop ratio | Seq-LoRA | DropFirst | DropSecond | DropAll |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 63.053 | 62.934 (−0.118) | 63.092 (+0.039) | 62.868 (−0.184) |
| 20% | 63.053 | 62.855 (−0.197) | 63.039 (−0.013) | 62.961 (−0.092) |
| 30% | 63.053 | 62.816 (−0.237) | 63.066 (+0.013) | 62.921 (−0.132) |
| 40% | 63.053 | 62.671 (−0.382) | 63.171 (+0.118) | 62.908 (−0.145) |
| 50% | 63.053 | 62.750 (−0.303) | 63.382 (+0.329) | 63.105 (+0.053) |
| 60% | 63.053 | 62.737 (−0.316) | 63.316 (+0.263) | 63.145 (+0.092) |
| 70% | 63.053 | 62.737 (−0.316) | 63.342 (+0.289) | 62.921 (−0.132) |
| 80% | 63.053 | 62.289 (−0.763) | 63.868 (+0.816) | 62.974 (−0.079) |
| 90% | 63.053 | 61.829 (−1.224) | **64.158 (+1.105)** | 62.342 (−0.711) |

原始彙整在 `logs/motivation-llama3-amazon-yahoo-r64a128-seed2026/svd_low/summary/results.csv`；圖片與 PDF 在同資料夾。圖的縱軸約從 61.5% 開始，因此視覺上的線距會放大；報告時應同時展示百分點與答對題數。

## 如何解讀

1. **DropFirst：九個比例全部低於 baseline。** 高比例刪除較明顯，90% 時少答對 93 題。這符合第一任務更新對 Amazon 成績有用的預期；並非每增加 10% 都嚴格下降，例如 40%→50% 略回升。
2. **DropSecond：八個比例高於 baseline，只有 20% 和 baseline 幾乎持平、略低。** 50% 時 +0.329 百分點，80% 時 +0.816，90% 時 +1.105。改善集中在高比例端，故不能概括為「中等比例就顯著改善」，也不能宣稱 90% 為跨 seed 的最佳比例。
3. **DropAll：與 baseline 接近、非單調。** 50% 和 60% 略高，90% 降至 62.342%；移除第二任務的可能益處，可以被移除第一任務有效成分的損失抵銷。這是結果的解釋之一，尚非因果機制證明。
4. **改善不是每種情緒類別都一致。** DropSecond 90% 相較 baseline，在 negative 多答對 148 題、positive 多 101 題；very negative 少 80 題、very positive 少 70 題、neutral 少 15 題，淨增加 84 題。總 accuracy 上升，不等於所有類別均變好。

對同一批 7,600 題逐題配對後，DropSecond 90% 相對 baseline 有 338 題從錯變對、254 題從對變錯，淨 +84 題；DropFirst 90% 為 310 題錯變對、403 題對變錯，淨 −93 題。若把測試題視作抽樣且暫不考慮多次比較，exact McNemar 雙側 p 值分別約 0.00063 與 0.00056；用配對差的常態近似，95% 區間分別約 **+0.48 至 +1.73**、**−1.91 至 −0.54 個百分點**。這些區間只描述固定訓練結果對題目抽樣的變動，**不包含重新訓練 seed 造成的不確定性**。本次探索了 27 個刪除設定；不能把挑出最高點後的單一 p 值當作預先指定的確認性檢定。

## 結論邊界與下一步

本次觀察支持較謹慎的說法：「在這個訓練 seed、資料版本及低奇異值截斷規則下，第二任務 LoRA 更新中有些成分移除後，第一任務 Amazon accuracy 會提高；移除第一任務更新則通常會降低 Amazon accuracy。」這與論文 Figure 1 的**方向**一致。[SLoRA 原論文，Figure 1、附錄 B.3](https://aclanthology.org/2026.acl-long.247.pdf)

目前不能量化完整的 catastrophic forgetting：沒有評估 **Task 1 訓練完成、Task 2 訓練前**的 Amazon accuracy，因此不知道學習 Task 2 後究竟掉了多少；也沒有 Yahoo accuracy，無法判定 DropSecond 90% 改善 Amazon 時是否犧牲 Task 2。只有一個訓練 seed，未能呈現 run-to-run 變異。若要把這張圖用於更強的因果論證，下一輪最值得補的是 Task 1 訓練後的 Amazon 基線、各刪除設定的 Yahoo 成績，以及至少數個獨立 seed 的平均值與誤差範圍；若要精確重現作者的 Figure 1，還需作者公開的成分排序與原始資料選樣細節。

## 可重現材料

- 資料來源與雜湊：`data/standard_cl/olora/manifest.json`；轉換程式：`scripts/data/prepare_standard_cl.py`。
- 訓練的實際參數：`checkpoints/motivation-llama3-amazon-yahoo-r64a128-seed2026/run_config.txt`；兩階段訓練紀錄：`logs/motivation-llama3-amazon-yahoo-r64a128-seed2026/svd_low/train/order{1,2}.log`。
- 消融方法、比例與每個 checkpoint 路徑：`checkpoints/motivation-llama3-amazon-yahoo-r64a128-seed2026-ablation-svd_low/manifest.json`；每個修改過的 adapter 另有 `ablation_meta.json`。
- 每個條件的逐題預測 `infer.jsonl`、accuracy `eval.log`、評分錯誤題 `wrong.json`；彙整 `summary/results.csv`、`results.json`、`figure_task1_accuracy.{png,pdf,svg}`。
- 執行入口：`scripts/motivation/run.sh`；資料處理與實驗設計說明：`scripts/motivation/README.md`。

## 設定來源逐項核對（補充）

這裡的「論文採用」要分兩層：SLoRA 的動機實驗直接描述了兩任務設計與 drop 條件；SLoRA §5.1 又說標準 CL 的任務提示和評估慣例沿用 O-LoRA。O-LoRA 的資料及程式是本地具體資料來源，不能因此說 SLoRA 作者公開了 Figure 1 每筆樣本與完整執行參數。

| 設定 | 可追溯來源 | 本次與來源的關係 |
| --- | --- | --- |
| Amazon Task 1 → Yahoo Task 2、兩者用 Seq-LoRA 依序訓練，最後觀察 Task 1；DropFirst／DropSecond／DropAll，10%–90% | SLoRA Figure 1 與附錄 B.3 | **論文明寫**。這不是 O-LoRA 的四任務主實驗順序。 |
| Amazon 情緒分類、Yahoo 主題分類；測分類 accuracy | SLoRA §5.1 與 B.3 | **論文明寫任務及指標**；未公開 Figure 1 的逐題結果。 |
| 模型 Llama3-8B-Instruct、rank 64、α 128、AdamW、LR 2×10⁻⁴、3% warmup、cosine、每任務 1 epoch | SLoRA §5.1 的標準 CL 設定 | **論文明寫**。B.3 稱 Llama3-8B，§5.1 明確使用 Instruct 版本。 |
| Amazon 五類、Yahoo 十類、標籤順序、資料筆數 | O-LoRA 固定 revision 的 `CL_Benchmark/SC/amazon`、`CL_Benchmark/TC/yahoo` 原始 JSON；本地 manifest 記錄 SHA-256 與列數 | **O-LoRA 公開資料決定，不是本地猜測或重分箱**。無法確認 SLoRA Figure 1 作者選了完全相同的樣本列。 |
| SC／TC 英文任務指示、`Option:` 列表、`Answer:` 後綴 | [O-LoRA 論文提示表](https://aclanthology.org/2023.findings-emnlp.715.pdf)、[O-LoRA 資料載入程式](https://raw.githubusercontent.com/cmnfriend/O-LoRA/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/src/uie_dataset_lora.py)；SLoRA §5.1 稱沿用 O-LoRA prompt | **依 O-LoRA 慣例實作**。本地再包進 Llama3 的 system/user/assistant chat template。 |
| 使用原始 train/test 所有列、無 validation split；Amazon 5,000/7,600、Yahoo 10,000/7,600 | O-LoRA 公開檔案 + 本地轉檔程式 | **本地資料版本的可驗證事實**；原論文未附 Figure 1 抽樣索引。 |
| `q/k/v/o/up/gate/down` 七種 LoRA target modules；LoRA dropout 0.05 | 專案既有 `cl_train.py` 和實際 `adapter_config.json` | **原專案實作／本次 checkpoint 的事實**，並非 §5.1/B.3 明寫。每任務 224 個模組、448 個張量、167,772,160 個 adapter 參數。 |
| 單卡 micro batch 1、梯度累積 32、有效 batch 約 32、最大訓練序列 1,024、seed 2026、weight decay 0、gradient clipping 1.0 | 本地 `run_config.txt`、`training_args.bin` 與 runner | **本地具體設定**；不能聲稱作者 Figure 1 也用這些值。 |
| AdamW β₁=0.9、β₂=0.999、ε=10⁻⁸，tokenization 不 packing，dataloader 不 drop last，無訓練中 validation、每 10 steps 記錄一次 | 實際 `training_args.bin`／TRL 0.8.6 `SFTTrainer` 預設 | **實際程式設定**，非論文逐項給出。訓練用完整 system/user/assistant 對話計算 causal LM loss，未用 answer-only loss mask。 |
| 選最小奇異值並固定比例刪除，每模組 `floor(64 × ratio)` | 本地 `make_lora_ablation_checkpoints.py` | **本地重建假設**。SLoRA B.3 只說刪除較不重要的更新子空間成分，未給 Figure 1 的可核對排序演算法。 |
| 單張 Quadro RTX 8000、FP16、DeepSpeed ZeRO-2、gradient checkpointing | 本機 GPU 與本地 launcher | **本地環境及節省記憶體的實作**。SLoRA §5.1 報告使用四張 A100；論文沒有寫 Figure 1 用 BF16 或 FP16，故不能聲稱 FP16 與論文精度相反。 |
| greedy 解碼、最多 32 新 tokens、generation batch 1、Llama3 chat template | 本地評估 runner 和專案既有推論程式 | **本地設定**；論文未完整公開 Figure 1 的解碼參數。 |

五類與十類可由 O-LoRA 固定 revision 的 [Amazon labels.json](https://raw.githubusercontent.com/cmnfriend/O-LoRA/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/CL_Benchmark/SC/amazon/labels.json) 和 [Yahoo labels.json](https://raw.githubusercontent.com/cmnfriend/O-LoRA/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/CL_Benchmark/TC/yahoo/labels.json) 直接核對。Amazon 五類依序為 `very negative`、`negative`、`neutral`、`positive`、`very positive`；Yahoo 十類依序為 `Society & Culture`、`Science & Mathematics`、`Health`、`Education & Reference`、`Computers & Internet`、`Sports`、`Business & Finance`、`Entertainment & Music`、`Family & Relationships`、`Politics & Government`。本地轉檔逐列沿用 `label`，沒有把 Amazon 改成正負二分類。

## Baseline 及對照組完整定義

| 組別 | 已完成？ | Amazon 測試 accuracy | 實際意義 |
| --- | --- | ---: | --- |
| Base model，不訓練 | 未跑 | — | 可看預訓練模型的零樣本能力；不是 SLoRA Figure 1 必須的參考線。 |
| Task 1 only，Amazon 訓練後、Yahoo 訓練前 | 未跑 | — | 計算真正遺忘量 `Acc(Task1 only) − Acc(Seq-LoRA)` 所需。 |
| **Seq-LoRA，兩任務全部保留** | **已跑** | **63.053% = 4,792/7,600** | SLoRA Figure 1 的主要 baseline；只評一次，在圖上重複畫成水平線。 |
| DropFirst，僅刪 Task 1 更新 | 已跑 9 比例 | 61.829%–62.934% | 檢查 Task 1 更新對 Amazon 是否有益。 |
| DropSecond，僅刪 Task 2 更新 | 已跑 9 比例 | 63.039%–64.158% | 檢查 Task 2 更新中是否有干擾 Amazon 的部分。 |
| DropAll，兩任務都刪 | 已跑 9 比例 | 62.342%–63.145% | 觀察同時去除兩邊成分時的淨效果。 |
| O-LoRA、SLoRA-Pre、SLoRA-Post、random drop | 未跑 | — | 屬其他方法或控制組；不能與本張圖的結果混寫。 |

圖上任何刪除組的 0% 都等價於 Seq-LoRA，但這次 manifest 只產生一個真正的 `seq_lora/r0` checkpoint 和一份評分；其餘是 27 個不同的刪除條件。這 28 次**評估**都使用同一組已訓練的兩個任務 adapter，並非重新訓練 28 次。

## 評估流程、與 O-LoRA 評分的差異

1. `run.sh eval` 讀取消融 manifest，逐一選 checkpoint；`--test_order 2` 使模型依序載入 Amazon、Yahoo 兩個 adapter 並合併至基底模型。每次都是新的模型載入，不把上一種刪除條件接續套用到下一種。
2. 對 Amazon `test.json` 的 7,600 個問題，使用相同的任務指示、五個選項與 Llama3 chat template，附 system 訊息 `You are a helpful assistant.`；生成時 `temperature=0` 導致 `do_sample=False`、單束搜尋，最多 32 tokens，遇 EOS 或 Llama3 `<|eot_id|>` 停止。
3. 推論程式將原始 `text`、正確答案及零起算索引寫入各組的 `infer.jsonl`。`src.eval.eval_scl` 先去除可能的回答前綴、尾端標點／空白，再比對類別文字或索引；每題對為 1、錯為 0，accuracy 為正確題數除以 7,600，錯題另存 `wrong.json`。
4. SLoRA 論文 §5.1 說標準 CL 沿用 O-LoRA 的 prompts 和 evaluation protocol；本次使用的是 **SLoRA 專案原有的生成器與 `eval_scl.py`**，並沒有直接執行 O-LoRA 的 `run_uie_lora.py`。O-LoRA 的 [原評分程式](https://raw.githubusercontent.com/cmnfriend/O-LoRA/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/src/compute_metrics.py) 用小寫、移除標點與多餘空白後的 exact match，並額外計算 ROUGE；兩個 parser 的規則並非逐行相同。
5. 已稽核本次 **28 組 × 7,600 題 = 212,800 個**生成結果：每個輸出都剛好是五個標籤文字之一；以 O-LoRA 的正規化 exact match 規則重新比對，與本地 `eval_scl.py` **0 題分歧**。因此在這批輸出上，兩種評分的 accuracy 數值一致。這能確認分數比對層的一致性，不能確認作者的 Figure 1 使用完全相同的樣本、chat template、生成長度或解碼實作。

報告時可寫：「依 SLoRA 所述採用 O-LoRA 的分類資料、任務提示與 accuracy 概念；本地使用 SLoRA 原有的推論及評分程式。對全部 212,800 筆輸出，O-LoRA 正規化 exact match 與本地評分一致。Figure 1 完整推論設定未公開，故評估流程僅能稱為指標層面的相符。」[SLoRA §5.1/B.3](https://aclanthology.org/2026.acl-long.247.pdf)

## 固定比例低奇異值截斷與 SLoRA 正式方法

本次 Figure 1 **動機實驗的本地代理實作**使用固定比例低奇異值截斷（`svd_low`）；它不是 SLoRA-Pre 或 SLoRA-Post，也不能驗證為作者 Figure 1 的精確實作。兩者都會把 LoRA 更新作 SVD，並以保留的前幾個奇異方向重建；關鍵差別是「保留幾個」如何決定。

對任務 k 的單一 LoRA 模組，令 `A_k ∈ R^{r×d_in}`、`B_k ∈ R^{d_out×r}`，實際加到權重的更新為 `ΔW_k = (α/r) B_k A_k`。忽略共同常數 `α/r`，將 `B_k A_k = U diag(σ_1,…,σ_r) Vᵀ`，其中 `σ_1 ≥ ⋯ ≥ σ_r ≥ 0`。本次對固定刪除比例 `ρ` 設 `q = floor(rρ)`、`c = r−q`，重建

`ΔW_k^(ρ) = (α/r) Σ_{i=1}^{c} σ_i u_i v_iᵀ`。

例如 `r=64, ρ=0.9` 時，`q=57`、`c=7`：保留七個最大的奇異方向。這等於矩陣在 Frobenius norm 下的最佳 rank-c 近似，**不等於**「最小奇異值一定是造成遺忘的雜訊」。程式用兩次 thin QR 加一個至多 64×64 的 SVD 計算同一分解，避免直接展開巨大的 `BA`；最後轉回 adapter dtype，保持原形狀及 α/r 縮放。參見 `src/motivation/make_lora_ablation_checkpoints.py`。

SLoRA 正式方法在論文 §4.2 另外取基底模型權重 `W_0` 的左奇異子空間 `U_ref,c`，對候選保留秩 `c ∈ C` 算 `φ_c = ||U_ref,cᵀ U_c||_F²`，選 `c* = argmax_{c∈C} φ_c`，再重建 `Σ_{i=1}^{c*} σ_i u_i v_iᵀ`。因此論文的正式方法以**和基底模型子空間的相似度來選秩**，每個模組可能有不同 `c*`，不是先指定每層一律刪 10%、20%……90%。SLoRA-Pre 在每個任務後去噪，SLoRA-Post 在任務訓練後處理保存的更新。[SLoRA §4.2、§4.4](https://aclanthology.org/2026.acl-long.247.pdf)

這次先重建 Figure 1 的固定比例對照，是因為附錄 B.3 把「各比例 DropFirst／DropSecond／DropAll」當作提出方法前的診斷實驗；直接改用自適應的正式方法，會改變自變量，無法回答固定比例曲線的問題。論文沒有公開 Figure 1 對「較不重要」的可核對排序公式，所以選擇 `σ_i` 大小作為透明且可重現的代理定義。它測到的是**依奇異值大小刪除的影響**，不能推論論文的基底子空間相似度方法會產生同一張曲線。若要研究正式 SLoRA 的成效，應另設 SLoRA-Pre／Post 對照，並同時評估 Amazon 與 Yahoo。
