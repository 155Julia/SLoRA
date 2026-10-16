# Amazon → Yahoo motivation experiment

This pipeline reuses `src/train/cl_train.py`, the original model loader/inference,
and `src/eval/eval_scl.py`. It adds fixed-ratio adapter ablations and plotting.
The old `*_smoke.sh` names remain compatibility wrappers; **their defaults now
run the Amazon → Yahoo configuration below**, not the old ten-step TRACE test.
Old smoke checkpoints/results are left untouched.

## Data and provenance

SLoRA §5.1 follows [O-LoRA](https://github.com/cmnfriend/O-LoRA) for standard CL
prompts and evaluation. Use its five-class Amazon Reviews, not the two-class
`SetFit/amazon_polarity`. The pinned source revision is
`07117e1fc4a5f5ad9308a815a42cee8f46502dc8`:

| Task | Source directory | Classes | Train | Test |
| --- | --- | ---: | ---: | ---: |
| amazon | CL_Benchmark/SC/amazon | 5 | 5,000 | 7,600 |
| yahoo | CL_Benchmark/TC/yahoo | 10 | 10,000 | 7,600 |

```bash
conda activate slora
python scripts/data/prepare_standard_cl.py --download
```

Already downloaded files are verified and reused. The converter verifies pinned
SHA-256 hashes, preserves every source row and its order, uses O-LoRA's SC/TC
instruction and option-list template, and writes `conversations` plus `eval`
metadata. Labels use their original ordered list (zero-based `answer_idx`).
It writes `data/standard_cl/olora/{amazon,yahoo}/{train,test}.json` and a provenance
manifest with counts, labels and input/output hashes. No validation split is
fabricated from the test set. Training does not use test data.

These are the upstream benchmark files, not a claim that the SLoRA authors used
exactly these rows for Figure 1: B.3 does not publish a sample-index manifest.
The previously downloaded Yahoo Topics dataset is valid but not used by default;
the official O-LoRA Yahoo subset avoids another sampling decision.

## Configuration and limits of paper alignment

Sources: [SLoRA §5.1 and Appendix B.3](https://aclanthology.org/2026.acl-long.247.pdf),
[O-LoRA data loader](https://github.com/cmnfriend/O-LoRA/blob/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/src/uie_dataset_lora.py),
and [instructions](https://github.com/cmnfriend/O-LoRA/blob/07117e1fc4a5f5ad9308a815a42cee8f46502dc8/configs/instruction_config.json).

| Setting | Default | Evidence |
| --- | --- | --- |
| Model / sequence | Llama3-8B-Instruct; Amazon → Yahoo | §5.1 / B.3 |
| LoRA r / alpha | 64 / 128 | §5.1 general defaults |
| Epochs / max steps | 1 / -1 (no step cap) | Standard CL setting, §5.1 |
| Optimizer / LR | AdamW / 2e-4 | §5.1 |
| Warmup / scheduler | 0.03 / cosine | §5.1 |
| Drop ratios | 0.1 through 0.9 | B.3 |
| GPU / batch / accumulation | 1 / 1 / 32 | Local choice: effective batch 32, matching checked-in main launcher (2×2×8), not a published Figure 1 batch size |
| Precision / seed | fp16 / 2026 | RTX 8000 supports FP16, not BF16; local, configurable choices |
| Sequence limit / generation limit | 1024 / 32 | Explicit local choices; Figure 1 limits unspecified |
| Target modules | q,k,v,o,up,gate,down projections | Existing trainer |

All shell settings in `config.sh` honor environment overrides. `PRECISION=bf16`
is appropriate on Ampere-or-newer GPUs such as A100; the local Quadro RTX 8000
defaults to `fp16` because it does not support BF16. The paper reports
four A100 GPUs; the local single-GPU launch is not a hardware-identical replay.
Training writes resolved settings and the data manifest beside its adapters.
Evaluation writes its settings beside predictions.

### What “less important components” means here

B.3 does not specify an exact scoring formula. Default `SELECTION=svd_low` is an
explicit reconstruction choice: truncate the smallest singular values of `B @ A`.
It uses thin QR and a rank-sized SVD, preserving tensor shapes and original PEFT
alpha/r scaling. It is **not verified as the authors' Figure 1 implementation**.

`SELECTION=energy_low` retains the previous rank-one heuristic, now corrected to
`||A[i,:]|| * ||B[:,i]||`. `energy_high` and seeded `random` are also available.
These A/B-coordinate heuristics are not equivalent to SVD truncation.
The number removed is `floor(rank * ratio)`; a tiny positive ratio may remove
zero components. Actual removed indices are recorded in `ablation_meta.json`.

## Run

From the repository root, with the `slora` environment activated:

```bash
bash scripts/motivation/run.sh train
bash scripts/motivation/run.sh ablate
bash scripts/motivation/run.sh eval
bash scripts/motivation/run.sh collect
# Or run all four stages:
# bash scripts/motivation/run.sh all
```

Inspect commands without loading a model:

```bash
DRY_RUN=1 bash scripts/motivation/run.sh train
MAX_STEPS=20 LORA_R=8 LORA_ALPHA=16 CKPT=amazon-yahoo-debug \
  DRY_RUN=1 bash scripts/motivation/run.sh train
```

Set `PYTHON_BIN` if the desired Python is not on PATH. Keep the same `CKPT`,
`SELECTION` and relevant overrides for each stage, or export them once before
running `all`. A short `MAX_STEPS` run is only a smoke test, not paper alignment.
For TRACE smoke experiments explicitly set `TASK_PATH`, `TASK1`, `TASK2`, and
`EVAL_MODULE=src.eval.eval_trace` as well as a separate `CKPT`.

Each grid has one Seq-LoRA baseline plus the three drop modes. Baseline and
zero-drop results are reused in plots. Unmodified adapters are symlinks to the
source, so keep source checkpoints available and immutable. No optimizer states,
tokenizers or intermediate checkpoints are duplicated for ablations. Evaluation
reuses the same modified adapter files for DropAll and DropFirst/DropSecond, and
uses the generated manifest, and collection reads that manifest's ratios/modes.

Existing training/output roots or predictions cause an error instead of silently
reusing another configuration. For a new experiment choose a new `CKPT`; for
another selection use `SELECTION`. Failed runs need a fresh output path (there
is no automatic resume). Training uses `pipefail` so failures cannot be hidden
by `tee`. Missing evaluation results fail collection rather than produce a
misleading partial plot.

## Verification

```bash
python -m unittest discover -s tests -v
```

CPU tests cover rank-one energy, SVD against dense truncated SVD, drop endpoints,
random nested masks, unique checkpoint grids, environment overrides, conversion
through the SCL metric, and manifest-based evaluation/plot collection. They do
not establish that full GPU training reproduces the paper's accuracy curve.
