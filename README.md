# GRADE — Generalizable Reasoning-Aware Dialogue Evaluation for AI Tutors

<!--**Authors:** Parth Bhalerao, Jeromy Chang, David Chou
**Affiliation:** Santa Clara University — CSEN-346, Spring 2026-->

Fine-tuning and evaluation pipeline for assessing how well LLMs perform as **math tutors**, built around the BEA 2025 shared task on pedagogical ability assessment. Each (conversation, tutor-response) pair is rated on four dimensions:

1. **Mistake Identification (MI)** — Does the response catch the student's error?
2. **Mistake Location (ML)** — Does it pinpoint *where* the error occurred?
3. **Providing Guidance (PG)** — Does it steer the student in the right direction?
4. **Actionability (Act)** — Does it give a clear next step?

Labels are 3-way: `Yes` / `No` / `To some extent`. We additionally train a **Multitask (MT)** head that predicts all four jointly.

This repo covers data preparation, prompt construction, zero-shot inference, LoRA fine-tuning, augmented-data generation (Gen and Gen+Verify), chain-of-thought reasoning variants, model scaling, evaluation, and carbon-emissions tracking across Llama-3.1-8B, Mistral-7B, Qwen3-14B, Gemma3-12B, and Gemma3-27B.

---

## Base Paper

This work builds on:

> **TutorMind at BEA 2025 Shared Task: Leveraging Fine-Tuned LLMs and Data Augmentation for Mistake Identification.** Dekmak et al., BEA 2025.

```bibtex
@inproceedings{dekmak-etal-2025-tutormind,
  title     = {TutorMind at {BEA} 2025 Shared Task: Leveraging Fine-Tuned {LLM}s and Data Augmentation for Mistake Identification},
  author    = {Dekmak and others},
  booktitle = {Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)},
  year      = {2025}
}
```

## Dataset

Available in the data folder, with separate non-augmented, and augmented variants in the jsonl formats.
<!--The processed dev/train/val splits used in this project are released on the Hugging Face Hub:

**[nlpscu/TutorMind on Hugging Face](https://huggingface.co/datasets/nlpscu/TutorMind)** — `https://huggingface.co/datasets/nlpscu/TutorMind`

```python
from datasets import load_dataset
ds = load_dataset("nlpscu/TutorMind")
```

The raw labeled file (`data/augmented_full_devset.json`) and the per-task chat-format `.jsonl` splits under `data/train/` and `data/val/` are the local materializations used by the training scripts.

----->

## Headline Results

Best **strict F1** per (model × task) across all 184 logged runs in [master_metrics.csv](master_metrics.csv). Strict F1 is the primary metric and treats `To some extent` as its own class; lenient F1 (in parentheses) collapses it into the majority class. Best per task in **bold**.

| Model | MI | ML | PG | Act | MT |
|---|---:|---:|---:|---:|---:|
| Mistral-7B    | 0.517 (0.762) | 0.369 (0.635) | 0.399 (0.610) | 0.473 (0.695) | 0.500 (0.733) |
| LLaMA-3.1-8B  | 0.660 (0.869) | 0.448 (0.724) | 0.485 (0.737) | 0.588 (0.781) | 0.567 (0.846) |
| Qwen3-14B     | 0.722 (0.861) | 0.503 (0.748) | 0.530 (0.741) | 0.637 (0.873) | 0.622 (0.857) |
| Gemma3-12B    | 0.750 (0.895) | **0.526 (0.764)** | **0.576 (0.789)** | **0.692 (0.883)** | 0.600 (0.879) |
| Gemma3-27B    | **0.771 (0.883)** | 0.513 (0.763) | 0.572 (0.787) | 0.645 (0.869) | **0.644 (0.888)** |

Key takeaways (by strict F1):

- **Gemma3 leads across the board** — Gemma3-27B wins MI and MT; Gemma3-12B wins ML, PG, and Act. The 27B is *not* a universal winner, suggesting a sweet spot around 12B for the 3-way decision.
- **LoRA beats zero-shot** on every model × task pair except Mistral-7B MI, where the zero-shot baseline (0.517) edges out LoRA.
- **Augmentation helps the harder labels.** Qwen3-Gen / Qwen3-Gen+Verify training data wins ML on every model, PG on Qwen3 and Gemma3-27B, and MT on Mistral-7B, LLaMA-3.1-8B, and Qwen3-14B. MI and Act are usually best with the original splits only.
- **Qwen3 "thinking" is task-dependent** — think-on wins MI, ML, and Act; think-off wins PG and MT.

Carbon-cost-adjusted rankings live in [master_metrics_with_carbon.csv](master_metrics_with_carbon.csv) — see the *Carbon tracking* notes below.

---

## Repository Structure

```
TutorMind/
├── README.md                              # This file
├── .gitignore
│
├── cleandata.py                           # Dedup + label sanity on augmented_full_devset.json
├── newdataset-preparation.py              # Build per-task train/val .jsonl splits from the devset
├── generate_prompts.py                    # Generate prompts.json (all task/CoT/MT prompt templates)
├── prompts.json                           # Materialized prompt templates used by every experiment
├── generate_augmented_data.py             # Qwen3-14B-based Gen / Gen+Verify augmentation generator
├── evaluate_run.py                        # Score a single prediction run; append row to master_metrics.csv
├── smoke_test_evaluate_run.py             # Smoke tests for evaluate_run.py
├── tryeval.py                             # Ad-hoc evaluation / inspection utility
├── merge_master_with_carbon.py            # Join master_metrics.csv with CarbonCalibration emissions
├── master_metrics.csv                     # Authoritative results table (all runs, all tasks, all methods)
├── master_metrics_with_carbon.csv         # master_metrics.csv + per-run kWh / CO₂ from CodeCarbon
├── ListExperiments.txt                    # Human-readable index of run IDs ↔ configs
│
├── data/                                  # All datasets used by training / inference
│   ├── augmented_full_devset.json         # Raw labeled (conversation, tutor-response) records
│   ├── Mistake_identification_*_chat_format.jsonl   # Legacy MI-only chat-format splits
│   ├── train/                             # Per-task training splits (chat format, used by trainers)
│   │   ├── mistake_identification_train.jsonl
│   │   ├── mistake_location_train.jsonl
│   │   ├── providing_guidance_train.jsonl
│   │   ├── actionability_train.jsonl
│   │   ├── multitask_train.jsonl
│   │   ├── Gen/                           # Augmented train sets (Qwen3 Gen-only, ~500 extra/class)
│   │   │   └── *_train_aug_qwen3_gen500.jsonl
│   │   ├── Gen+Verify/                    # Augmented train sets (Qwen3 Gen + LLM verifier filter)
│   │   │   └── *_train_aug_qwen3_genverify500.jsonl
│   │   └── augmentation_audit/            # Audit + smoke-test scripts for augmentation pipeline
│   └── val/                               # Per-task validation splits (golden eval set)
│       └── {mistake_identification,mistake_location,providing_guidance,actionability,multitask}_val.jsonl
│
├── utils/
│   └── codecarbon_helper.py               # `track_emissions()` context manager; routes all runs
│                                          # through repo-central emissions/ CSV files
│
├── Baseline-Experiments/                  # Llama-3.1-8B, Mistral-7B on the ORIGINAL train splits
│   ├── ZeroShot/Llama-3.1-8B/             # zero_shot_infer.py + per-task prediction CSVs
│   ├── LORA/Llama-3.1-8B/                 # train_lora.py + infer_lora.py + run-NN adapters & preds
│   ├── LORA/Mistral-7B/                   # (parallel layout)
│   ├── EVAL/                              # Evaluation outputs for baseline runs
│   ├── logs/                              # Slurm + script logs (per model, per run)
│   └── run_llama_baseline.sh              # Driver: zero-shot + LoRA × {MI,ML,PG,Act,MT}
│
├── Scaling-Experiments/                   # Same pipeline scaled to larger models
│   ├── ZeroShot/{Mistral-7B,Qwen3-14B,Gemma3-12B,Gemma3-27B}/   # zeroshot_*.py + preds
│   ├── LORA/{Mistral-7B,Qwen3-14B,Gemma3-12B,Gemma3-27B}/       # train / infer LoRA scripts
│   ├── Qwen3-14B/ThinkAug/                # Qwen3 "thinking-mode" augmentation outputs
│   ├── EVAL/                              # Per-model evaluation outputs
│   ├── logs/
│   ├── run_mistral_baseline.sh            # Mistral scaling driver
│   └── run_qwen_scaling.sh                # Qwen3 / Gemma scaling driver
│
├── DataAugmentation-Experiments/          # Train on ORIGINAL + Gen / Gen+Verify augmented data
│   ├── Generate/ZeroShot/Llama/           # zero_shot_infer.py used to GENERATE augmented responses
│   ├── Verify/Zeroshot/Llama/             # zero_shot_infer.py used to VERIFY generated responses
│   ├── LORA/{Llama,Mistral,Qwen3-14B,Gemma3-12B,Gemma3-27B}/    # LoRA train+infer on augmented data
│   ├── FullFT/                            # (reserved) full fine-tuning experiments
│   └── EVAL/                              # Augmented-data run evaluations
│
├── CoT-Experiments/                       # Chain-of-thought / Qwen3 "thinking" runs
│   ├── ZeroShot/Qwen3-14B/                # zeroshot_qwen_cot.py — CoT zero-shot
│   ├── ZeroShot/Qwen3_14B_Gen/, Qwen3_14B_GenVerify/   # CoT zero-shot on augmented test conditions
│   ├── LORA/Qwen3-14B/                    # train_qwen_lora_cot.py + infer_qwen_lora_cot.py
│   ├── LORA/Qwen3_14B_Gen/, Qwen3_14B_GenVerify/        # CoT LoRA + augmented data
│   ├── archive/                           # Older CoT driver scripts
│   ├── logs/
│   ├── run_qwen_cot.sh                    # CoT driver (MI/ML/PG/Act/MT × zero-shot/LoRA)
│   └── run_qwen3_think_aug_111_120.sh     # CoT × augmentation combined driver
│
├── MultiTask-Experiments/                 # MT-specific scaffolding (LORA / FullFT / EVAL)
│
├── AllCombined-Experiments/               # End-to-end "best recipe" runs (LORA / FullFT / EVAL)
│
├── CarbonCalibration-Temp/                # CodeCarbon recalibration (isolated from main runs)
│   ├── scripts/                           # *_zeroshot_calibration.py and *_lora_calibration.py
│   │                                      # for Llama / Mistral / Qwen3 (think on+off) / Gemma 12B+27B
│   ├── adapters/                          # Calibration-only LoRA adapters (calib_runNNN_*)
│   ├── outputs/                           # Calibration prediction CSVs
│   ├── emissions/                         # Per-run CodeCarbon emissions CSVs (kWh, CO₂eq)
│   ├── carbon_calibration_summary.csv     # Aggregated calibration table
│   └── logs/, notes/
│
├── emissions/                             # Repo-central CodeCarbon outputs from production runs
│                                          # (written via utils/codecarbon_helper.py)
│
├── logs/                                  # Top-level Slurm + Python logs
├── slurm_logs/                            # Slurm stdout/stderr from cluster jobs
│
├── run_emissions_mt_part1.sh              # Drivers that re-run MT with CodeCarbon tracking
├── run_emissions_mt_part2.sh
├── run_aug_gen_qwen3.sh                   # Driver for generate_augmented_data.py
├── run_genverify_debug_mi_no.sh           # Debug driver for Gen+Verify MI=No edge case
├── rerun_runs_116_120_genverify_inference_*.sh    # Re-inference driver for runs 116–120
├── submit_*.sbatch                        # Slurm submission scripts for Qwen3 retrain / re-eval
```

---

## What Each Script Does

### Data preparation
- **`cleandata.py`** — In-place clean of `data/augmented_full_devset.json`: removes (conversation_id, model) duplicates, flags Command R+ rows as MI-only (their ML/PG/Act labels are N/A). Run **before** `newdataset-preparation.py`.
- **`newdataset-preparation.py`** — Splits the cleaned devset 80/20 into per-task chat-format `.jsonl` files under `data/train/` and `data/val/`. Deduplicates at the JSONL level and skips N/A labels for non-MI tasks. Reports class distributions.
- **`generate_prompts.py`** → **`prompts.json`** — Single source of truth for every prompt used in the project (zero-shot, CoT/thinking, single-task, multitask, and augmentation prompts).
- **`generate_augmented_data.py`** — Runs Qwen3-14B locally to generate synthetic tutor responses for under-represented labels. Writes `data/train/Gen/` (generation only) and feeds `data/train/Gen+Verify/` (LLM-verified subset).

### Training & inference
Each experiment family follows the same internal layout:

| Folder | Train script | Infer script | Trained on |
|---|---|---|---|
| `Baseline-Experiments/LORA/<Model>/` | `train_lora.py` (or `train_mistral_lora.py`) | `infer_lora.py` | Original splits |
| `Scaling-Experiments/LORA/<Model>/` | `train_*.py` / `lora_*.py` | `infer_*.py` | Original splits, larger models |
| `DataAugmentation-Experiments/LORA/<Model>/` | `train_*.py` | `infer_*.py` | Original + Gen / Gen+Verify |
| `CoT-Experiments/LORA/Qwen3-14B/` | `train_qwen_lora_cot.py` | `infer_qwen_lora_cot.py` | Original ± augmented, CoT format |

Zero-shot variants live under each family's `ZeroShot/<Model>/` and run a single `zero_shot_infer.py` / `zeroshot_*.py` against the validation splits.

### Evaluation
- **`evaluate_run.py`** — Canonical scorer. Takes one prediction CSV (single-task: `pred_label`; MT: `pred_mi/ml/pg/act`), validates against the matching `*_val.jsonl`, and appends a row to **`master_metrics.csv`** with accuracy / F1 / class-level stats.
- **`smoke_test_evaluate_run.py`** — Self-tests for `evaluate_run.py`.
- **`tryeval.py`** — Ad-hoc inspection / one-off evaluations during development.

### Carbon tracking
- **`utils/codecarbon_helper.py`** — `track_emissions(...)` context manager wrapped around every training / inference loop. Routes per-run kWh and CO₂eq into `emissions/` (production) or `CarbonCalibration-Temp/emissions/` (calibration).
- **`CarbonCalibration-Temp/scripts/*_calibration.py`** — Isolated re-runs of a representative subset of MI/MT × {zero-shot, LoRA} × {original, Gen, Gen+Verify} jobs to calibrate the CodeCarbon tracker without polluting `master_metrics.csv`.
- **`merge_master_with_carbon.py`** — Joins `master_metrics.csv` against `CarbonCalibration-Temp/carbon_calibration_summary.csv` to produce **`master_metrics_with_carbon.csv`**, the table used for accuracy-vs-emissions analysis.

### Cluster drivers
- `run_llama_baseline.sh`, `run_mistral_baseline.sh`, `run_qwen_scaling.sh`, `run_qwen_cot.sh`, `run_qwen3_think_aug_111_120.sh`, `run_aug_gen_qwen3.sh`, `run_emissions_mt_part1.sh`, `run_emissions_mt_part2.sh` — Bash drivers that loop over tasks / models / methods and `srun` the matching Python script.
- `submit_*.sbatch`, `rerun_*.sh` — Slurm submission and re-run scripts for specific run-ID ranges.

---

## Reproducing a Run

```bash
# 1. Build splits from the labeled devset
python cleandata.py
python newdataset-preparation.py

# 2. (Optional) regenerate prompt templates
python generate_prompts.py

# 3. Train + infer a LoRA baseline (example: Llama-3.1-8B on MI)
bash Baseline-Experiments/run_llama_baseline.sh

# 4. Score the run and append to master_metrics.csv
python evaluate_run.py --run-id 006 --task mi \
    --pred-csv Baseline-Experiments/LORA/Llama-3.1-8B/outputs/<run>/preds.csv

# 5. Join with carbon data
python merge_master_with_carbon.py
```

---

## Annotation Schema

| Dimension | Description | Values |
|---|---|---|
| `Mistake_Identification` | Does the response catch the student's error? | `Yes` / `No` / `To some extent` |
| `Mistake_Location` | Does it pinpoint where the error occurred? | `Yes` / `No` / `To some extent` |
| `Providing_Guidance` | Does it steer the student in the right direction? | `Yes` / `No` / `To some extent` |
| `Actionability` | Does it give a clear next step to the student? | `Yes` / `No` / `To some extent` |

Annotated tutor models in the source devset include **Expert** (human), **GPT-4**, **Claude Sonnet**, **Gemini**, **Llama-3.1-405B**, **Llama-3.1-8B**, **Mistral**, **Phi-3**, and **Command R+** (MI-only).

---

## License

MIT — see [LICENSE](LICENSE).
