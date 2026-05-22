#!/usr/bin/env python3
"""CodeCarbon calibration for Gemma3-12B LoRA MI and MT.

Supports 6 calibration runs:
- 036 = Original MI
- 040 = Original MT
- 076 = Gen MI
- 080 = Gen MT
- 101 = Gen+Verify MI
- 105 = Gen+Verify MT

All outputs go under CarbonCalibration-Temp.
This does not touch master_metrics.csv or original experiment folders.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer


REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
ADAPTERS_DIR = TEMP_ROOT / "adapters"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

MODEL_NAME = "Gemma3-12B"
MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3"
PROMPTS_PATH = REPO_ROOT / "prompts.json"

METHOD = "LoRA"
THINK = "N/A"
UNKNOWN_LABEL = "Unknown"

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
SEED = 3407
MAX_NEW_TOKENS = 64

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


RUNS: Dict[str, dict] = {
    "036": {
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "None",
        "train_jsonl": str(REPO_ROOT / "data/train/mistake_identification_train.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "040": {
        "task": "MT",
        "task_group": "MT",
        "aug": "None",
        "train_jsonl": str(REPO_ROOT / "data/train/multitask_train.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/multitask_val.jsonl"),
    },
    "076": {
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "080": {
        "task": "MT",
        "task_group": "MT",
        "aug": "Qwen3-Gen",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/multitask_val.jsonl"),
    },
    "101": {
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "105": {
        "task": "MT",
        "task_group": "MT",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/multitask_val.jsonl"),
    },
}

TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def slugify_aug(aug: str) -> str:
    if aug == "None":
        return "orig"
    return aug.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")


def build_project_base(run_id: str, task: str, aug: str) -> str:
    return f"calib_run{run_id}_gemma3_12b_lora_{slugify_aug(aug)}_{task.lower()}"


def load_jsonl(path: Path) -> List[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return examples


def choose_sample_indices(
    full_size: int,
    sample_fraction: float,
    min_samples: int,
    max_samples: int,
    seed: int,
) -> list[int]:
    if full_size <= 0:
        raise ValueError("Input file is empty.")

    requested = math.ceil(full_size * sample_fraction)
    sample_size = max(min_samples, requested)
    sample_size = min(sample_size, max_samples, full_size)

    rng = random.Random(seed)
    return sorted(rng.sample(range(full_size), sample_size))


def estimate_update_steps(num_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    return max(1, math.ceil(steps_per_epoch * epochs))


def normalize_label(text: str) -> str:
    if not text:
        return UNKNOWN_LABEL

    candidates = [text.strip()]
    for line in text.splitlines():
        line = line.strip()
        if line:
            candidates.append(line)
        if ":" in line:
            candidates.append(line.split(":", 1)[1].strip())

    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'`.,!?:;").lower()
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)

        if cleaned == "yes":
            return "Yes"
        if cleaned == "no":
            return "No"
        if cleaned in {"to some extent", "to some extend"}:
            return "To some extent"

    return UNKNOWN_LABEL


def parse_multitask_output(text: str) -> dict:
    field_map = {
        "mistakeidentification": "pred_mi",
        "mistakelocation": "pred_ml",
        "providingguidance": "pred_pg",
        "actionability": "pred_act",
        "mi": "pred_mi",
        "ml": "pred_ml",
        "pg": "pred_pg",
        "act": "pred_act",
    }

    result = {
        "pred_mi": UNKNOWN_LABEL,
        "pred_ml": UNKNOWN_LABEL,
        "pred_pg": UNKNOWN_LABEL,
        "pred_act": UNKNOWN_LABEL,
    }

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue

        field, value = line.split(":", 1)
        key = field.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        col = field_map.get(key)

        if col:
            result[col] = normalize_label(value.strip())

    regex_patterns = {
        "pred_mi": r"(mistake[\s_-]*identification|mi)\s*:\s*(yes|no|to some extent)",
        "pred_ml": r"(mistake[\s_-]*location|ml)\s*:\s*(yes|no|to some extent)",
        "pred_pg": r"(providing[\s_-]*guidance|pg)\s*:\s*(yes|no|to some extent)",
        "pred_act": r"(actionability|act)\s*:\s*(yes|no|to some extent)",
    }

    for col, pattern in regex_patterns.items():
        if result[col] != UNKNOWN_LABEL:
            continue
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[col] = normalize_label(match.group(2))

    return result


def extract_user_content(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found.")


def format_chat_for_training(example: dict, processor) -> dict:
    converted = []
    for msg in example["messages"]:
        converted.append(
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
        )

    text = processor.apply_chat_template(
        converted,
        add_generation_prompt=False,
        tokenize=False,
    )

    return {"text": text}


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_PATH)


def load_base_model():
    print(f"Loading base model from {MODEL_PATH} ...")
    print("Mode: full bfloat16, no quantization")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {allocated:.1f}GB / {total:.1f}GB")

    return model


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_trainer(model, processor, train_dataset: Dataset, run_id: str, output_dir: Path) -> SFTTrainer:
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        fp16=False,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",
        save_strategy="no",
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    return trainer


@torch.inference_mode()
def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    new_tokens = output_ids[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def cleanup_memory(*objects) -> None:
    for obj in objects:
        if obj is not None:
            del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def read_last_codecarbon_row(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    return rows[-1]


def get_codecarbon_metrics(project_name: str) -> dict[str, float | str]:
    csv_path = EMISSIONS_DIR / f"{project_name}.csv"
    row = read_last_codecarbon_row(csv_path)

    return {
        "codecarbon_csv_path": str(csv_path),
        "codecarbon_duration_sec": safe_float(row.get("duration")),
        "energy_consumed_kwh": safe_float(row.get("energy_consumed")),
        "emissions_kg_co2eq": safe_float(row.get("emissions")),
        "gpu_energy_kwh": safe_float(row.get("gpu_energy")),
        "cpu_energy_kwh": safe_float(row.get("cpu_energy")),
        "ram_energy_kwh": safe_float(row.get("ram_energy")),
        "gpu_model": row.get("gpu_model", ""),
        "cpu_model": row.get("cpu_model", ""),
        "ram_total_size": row.get("ram_total_size", ""),
        "codecarbon_version": row.get("codecarbon_version", ""),
    }


def write_predictions_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preferred_summary_fieldnames() -> list[str]:
    return [
        "timestamp_utc",
        "status",
        "original_run_id",
        "calibration_project_base",
        "model",
        "method",
        "task",
        "task_group",
        "think",
        "aug",
        "train_file",
        "val_file",
        "full_train_size",
        "train_sample_size",
        "train_sample_fraction_actual",
        "train_sample_seed",
        "train_sample_indices_preview",
        "full_val_size",
        "val_sample_size",
        "val_sample_fraction_actual",
        "val_sample_seed",
        "val_sample_indices_preview",
        "epochs",
        "learning_rate",
        "batch_size",
        "grad_accum",
        "effective_batch_size",
        "max_new_tokens",
        "estimated_full_train_steps",
        "measured_train_steps",
        "prediction_output_path",
        "unknown_count",
        "retry_count",
        "processor_load_project_name",
        "processor_load_codecarbon_csv_path",
        "processor_load_wall_duration_sec",
        "processor_load_energy_consumed_kwh",
        "processor_load_emissions_kg_co2eq",
        "train_prep_project_name",
        "train_prep_codecarbon_csv_path",
        "train_prep_wall_duration_sec",
        "train_prep_energy_consumed_kwh",
        "train_prep_emissions_kg_co2eq",
        "train_load_project_name",
        "train_load_codecarbon_csv_path",
        "train_load_wall_duration_sec",
        "train_load_energy_consumed_kwh",
        "train_load_emissions_kg_co2eq",
        "train_load_gpu_energy_kwh",
        "train_fit_project_name",
        "train_fit_codecarbon_csv_path",
        "train_fit_wall_duration_sec",
        "train_fit_energy_consumed_kwh",
        "train_fit_emissions_kg_co2eq",
        "train_fit_gpu_energy_kwh",
        "infer_generate_project_name",
        "infer_generate_codecarbon_csv_path",
        "infer_generate_wall_duration_sec",
        "infer_generate_energy_consumed_kwh",
        "infer_generate_emissions_kg_co2eq",
        "infer_generate_gpu_energy_kwh",
        "measured_total_energy_kwh",
        "measured_total_emissions_kg_co2eq",
        "estimated_full_train_fit_energy_kwh",
        "estimated_full_train_fit_emissions_kg_co2eq",
        "estimated_full_infer_generate_energy_kwh",
        "estimated_full_infer_generate_emissions_kg_co2eq",
        "estimated_full_energy_kwh",
        "estimated_full_emissions_kg_co2eq",
        "estimation_formula",
        "gpu_model",
        "cpu_model",
        "ram_total_size",
        "codecarbon_version",
        "notes",
    ]


def upsert_summary_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    existing_fieldnames = []

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    key = row["original_run_id"]
    existing_rows = [existing for existing in existing_rows if existing.get("original_run_id") != key]
    existing_rows.append(row)

    fieldnames = []

    for name in preferred_summary_fieldnames():
        if name not in fieldnames:
            fieldnames.append(name)

    for name in existing_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)

    for record in existing_rows:
        for name in record.keys():
            if name not in fieldnames:
                fieldnames.append(name)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeCarbon calibration for Gemma3-12B LoRA MI/MT."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=sorted(RUNS.keys()),
        help="036/040 original, 076/080 Gen, 101/105 Gen+Verify.",
    )

    parser.add_argument("--train-sample-fraction", type=float, default=0.25)
    parser.add_argument("--min-train-samples", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--train-sample-seed", type=int, default=42)

    parser.add_argument("--val-sample-fraction", type=float, default=0.10)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--max-val-samples", type=int, default=100)
    parser.add_argument("--val-sample-seed", type=int, default=42)

    args = parser.parse_args()

    if args.train_sample_fraction <= 0 or args.train_sample_fraction > 1:
        raise ValueError("--train-sample-fraction must be in (0, 1].")
    if args.val_sample_fraction <= 0 or args.val_sample_fraction > 1:
        raise ValueError("--val-sample-fraction must be in (0, 1].")

    ensure_dirs()

    run = RUNS[args.run_id]
    run_id = args.run_id
    task = run["task"]
    aug = run["aug"]

    train_path = Path(run["train_jsonl"])
    val_path = Path(run["val_jsonl"])

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing val file: {val_path}")
    if not PROMPTS_PATH.is_file():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_PATH}")

    project_base = build_project_base(run_id, task, aug)
    prediction_output_path = OUTPUTS_DIR / f"{project_base}_sample_predictions.csv"
    tmp_output_dir = ADAPTERS_DIR / project_base / "tmp_trainer"

    processor_project = f"{project_base}_processor_load"
    train_prep_project = f"{project_base}_train_prep"
    train_load_project = f"{project_base}_train_load_lora"
    train_fit_project = f"{project_base}_train_fit"
    infer_generate_project = f"{project_base}_infer_generate"

    print("=" * 72)
    print(f"Carbon calibration run: {project_base}")
    print(f"Original run id: {run_id}")
    print(f"Task: {task}")
    print(f"Aug: {aug}")
    print(f"Model: {MODEL_NAME}")
    print(f"Method: {METHOD}")
    print("Mode: full bfloat16, no quantization")
    print(f"Train file: {train_path}")
    print(f"Val file: {val_path}")
    print(f"Prediction output: {prediction_output_path}")
    print(f"Emissions dir: {EMISSIONS_DIR}")
    print("=" * 72)

    with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    single_task_prompts = prompts["single_task_training"]["prompts"]
    multitask_prompt = prompts["multitask_training"]["prompt"]
    retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask = prompts["retry_prompts"]["prompts"]["multitask"]

    full_train_examples = load_jsonl(train_path)
    full_val_examples = load_jsonl(val_path)

    full_train_size = len(full_train_examples)
    full_val_size = len(full_val_examples)

    train_indices = choose_sample_indices(
        full_size=full_train_size,
        sample_fraction=args.train_sample_fraction,
        min_samples=args.min_train_samples,
        max_samples=args.max_train_samples,
        seed=args.train_sample_seed + int(run_id),
    )

    val_indices = choose_sample_indices(
        full_size=full_val_size,
        sample_fraction=args.val_sample_fraction,
        min_samples=args.min_val_samples,
        max_samples=args.max_val_samples,
        seed=args.val_sample_seed + int(run_id),
    )

    train_examples = [full_train_examples[index] for index in train_indices]
    val_examples = [full_val_examples[index] for index in val_indices]

    train_sample_size = len(train_examples)
    val_sample_size = len(val_examples)

    train_sample_fraction_actual = train_sample_size / full_train_size
    val_sample_fraction_actual = val_sample_size / full_val_size

    estimated_full_train_steps = estimate_update_steps(
        full_train_size,
        PER_DEVICE_BATCH_SIZE,
        GRADIENT_ACCUM_STEPS,
        NUM_EPOCHS,
    )

    print(f"Full train size: {full_train_size}")
    print(f"Train sample size: {train_sample_size}")
    print(f"Full validation size: {full_val_size}")
    print(f"Validation sample size: {val_sample_size}")
    print(f"Estimated full train optimizer steps: {estimated_full_train_steps}")

    processor_start = time.time()
    with track_emissions(processor_project, output_dir=EMISSIONS_DIR):
        processor = load_processor()
    processor_wall = time.time() - processor_start

    prep_start = time.time()
    with track_emissions(train_prep_project, output_dir=EMISSIONS_DIR):
        formatted = [format_chat_for_training(ex, processor) for ex in train_examples]
        train_dataset = Dataset.from_list(formatted)
    train_prep_wall = time.time() - prep_start

    train_load_start = time.time()
    with track_emissions(train_load_project, output_dir=EMISSIONS_DIR):
        model = load_base_model()
        model = apply_lora(model)
    train_load_wall = time.time() - train_load_start

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        run_id=run_id,
        output_dir=tmp_output_dir,
    )

    train_fit_start = time.time()
    with track_emissions(train_fit_project, output_dir=EMISSIONS_DIR):
        trainer.train()
    train_fit_wall = time.time() - train_fit_start

    measured_train_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    if measured_train_steps <= 0:
        measured_train_steps = estimate_update_steps(
            train_sample_size,
            PER_DEVICE_BATCH_SIZE,
            GRADIENT_ACCUM_STEPS,
            NUM_EPOCHS,
        )

    print(f"Measured train optimizer steps: {measured_train_steps}")

    model.eval()

    if task == "MT":
        system_prompt = multitask_prompt
        retry_prompt = retry_multitask
        output_columns = ["row_index", "task", "pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"]
    else:
        system_prompt = single_task_prompts[TASK_TO_PROMPT_KEY[task]]
        retry_prompt = retry_single
        output_columns = ["row_index", "task", "pred_label", "raw_output"]

    rows = []
    unknown_count = 0
    retry_count = 0

    infer_start = time.time()
    with track_emissions(infer_generate_project, output_dir=EMISSIONS_DIR):
        for local_i, example in enumerate(val_examples, start=1):
            source_index = val_indices[local_i - 1]
            user_content = extract_user_content(example)
            raw_output = run_inference(model, processor, system_prompt, user_content)

            if task == "MT":
                parsed = parse_multitask_output(raw_output)

                if UNKNOWN_LABEL in parsed.values():
                    retry_count += 1
                    raw_output2 = run_inference(model, processor, retry_prompt, user_content)
                    parsed2 = parse_multitask_output(raw_output2)

                    for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
                        if parsed[col] == UNKNOWN_LABEL:
                            parsed[col] = parsed2[col]

                    raw_output = raw_output2

                if UNKNOWN_LABEL in parsed.values():
                    unknown_count += 1

                rows.append(
                    {
                        "row_index": source_index,
                        "task": task,
                        "pred_mi": parsed["pred_mi"],
                        "pred_ml": parsed["pred_ml"],
                        "pred_pg": parsed["pred_pg"],
                        "pred_act": parsed["pred_act"],
                        "raw_output": raw_output,
                    }
                )
            else:
                pred_label = normalize_label(raw_output)

                if pred_label == UNKNOWN_LABEL:
                    retry_count += 1
                    raw_output2 = run_inference(model, processor, retry_prompt, user_content)
                    pred_label = normalize_label(raw_output2)
                    raw_output = raw_output2

                if pred_label == UNKNOWN_LABEL:
                    unknown_count += 1

                rows.append(
                    {
                        "row_index": source_index,
                        "task": task,
                        "pred_label": pred_label,
                        "raw_output": raw_output,
                    }
                )

            if local_i == 1 or local_i % 10 == 0 or local_i == val_sample_size:
                elapsed = time.time() - infer_start
                print(
                    f"Progress: {local_i}/{val_sample_size} | "
                    f"Unknowns: {unknown_count} | Retries: {retry_count} | "
                    f"Inference time: {elapsed / 60:.2f} min",
                    flush=True,
                )

    infer_wall = time.time() - infer_start

    write_predictions_csv(prediction_output_path, output_columns, rows)

    processor_metrics = get_codecarbon_metrics(processor_project)
    train_prep_metrics = get_codecarbon_metrics(train_prep_project)
    train_load_metrics = get_codecarbon_metrics(train_load_project)
    train_fit_metrics = get_codecarbon_metrics(train_fit_project)
    infer_metrics = get_codecarbon_metrics(infer_generate_project)

    processor_energy = float(processor_metrics["energy_consumed_kwh"])
    processor_emissions = float(processor_metrics["emissions_kg_co2eq"])

    train_prep_energy = float(train_prep_metrics["energy_consumed_kwh"])
    train_prep_emissions = float(train_prep_metrics["emissions_kg_co2eq"])

    train_load_energy = float(train_load_metrics["energy_consumed_kwh"])
    train_load_emissions = float(train_load_metrics["emissions_kg_co2eq"])

    train_fit_energy = float(train_fit_metrics["energy_consumed_kwh"])
    train_fit_emissions = float(train_fit_metrics["emissions_kg_co2eq"])

    infer_energy = float(infer_metrics["energy_consumed_kwh"])
    infer_emissions = float(infer_metrics["emissions_kg_co2eq"])

    train_fit_energy_per_step = train_fit_energy / measured_train_steps
    train_fit_emissions_per_step = train_fit_emissions / measured_train_steps

    infer_energy_per_example = infer_energy / val_sample_size
    infer_emissions_per_example = infer_emissions / val_sample_size

    estimated_full_train_fit_energy = train_fit_energy_per_step * estimated_full_train_steps
    estimated_full_train_fit_emissions = train_fit_emissions_per_step * estimated_full_train_steps

    estimated_full_infer_energy = infer_energy_per_example * full_val_size
    estimated_full_infer_emissions = infer_emissions_per_example * full_val_size

    measured_total_energy = (
        processor_energy + train_prep_energy + train_load_energy + train_fit_energy + infer_energy
    )
    measured_total_emissions = (
        processor_emissions + train_prep_emissions + train_load_emissions + train_fit_emissions + infer_emissions
    )

    estimated_full_energy = (
        processor_energy
        + train_prep_energy
        + train_load_energy
        + estimated_full_train_fit_energy
        + estimated_full_infer_energy
    )

    estimated_full_emissions = (
        processor_emissions
        + train_prep_emissions
        + train_load_emissions
        + estimated_full_train_fit_emissions
        + estimated_full_infer_emissions
    )

    hardware_source = infer_metrics or train_fit_metrics or train_load_metrics

    summary_row = {
        "timestamp_utc": utc_now(),
        "status": "success",
        "original_run_id": run_id,
        "calibration_project_base": project_base,
        "model": MODEL_NAME,
        "method": METHOD,
        "task": task,
        "task_group": run["task_group"],
        "think": THINK,
        "aug": aug,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "full_train_size": full_train_size,
        "train_sample_size": train_sample_size,
        "train_sample_fraction_actual": f"{train_sample_fraction_actual:.8f}",
        "train_sample_seed": args.train_sample_seed,
        "train_sample_indices_preview": json.dumps(train_indices[:25]),
        "full_val_size": full_val_size,
        "val_sample_size": val_sample_size,
        "val_sample_fraction_actual": f"{val_sample_fraction_actual:.8f}",
        "val_sample_seed": args.val_sample_seed,
        "val_sample_indices_preview": json.dumps(val_indices[:25]),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "grad_accum": GRADIENT_ACCUM_STEPS,
        "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUM_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "estimated_full_train_steps": estimated_full_train_steps,
        "measured_train_steps": measured_train_steps,
        "prediction_output_path": str(prediction_output_path),
        "unknown_count": unknown_count,
        "retry_count": retry_count,
        "processor_load_project_name": processor_project,
        "processor_load_codecarbon_csv_path": processor_metrics["codecarbon_csv_path"],
        "processor_load_wall_duration_sec": f"{processor_wall:.6f}",
        "processor_load_energy_consumed_kwh": f"{processor_energy:.12f}",
        "processor_load_emissions_kg_co2eq": f"{processor_emissions:.12f}",
        "train_prep_project_name": train_prep_project,
        "train_prep_codecarbon_csv_path": train_prep_metrics["codecarbon_csv_path"],
        "train_prep_wall_duration_sec": f"{train_prep_wall:.6f}",
        "train_prep_energy_consumed_kwh": f"{train_prep_energy:.12f}",
        "train_prep_emissions_kg_co2eq": f"{train_prep_emissions:.12f}",
        "train_load_project_name": train_load_project,
        "train_load_codecarbon_csv_path": train_load_metrics["codecarbon_csv_path"],
        "train_load_wall_duration_sec": f"{train_load_wall:.6f}",
        "train_load_energy_consumed_kwh": f"{train_load_energy:.12f}",
        "train_load_emissions_kg_co2eq": f"{train_load_emissions:.12f}",
        "train_load_gpu_energy_kwh": f"{float(train_load_metrics['gpu_energy_kwh']):.12f}",
        "train_fit_project_name": train_fit_project,
        "train_fit_codecarbon_csv_path": train_fit_metrics["codecarbon_csv_path"],
        "train_fit_wall_duration_sec": f"{train_fit_wall:.6f}",
        "train_fit_energy_consumed_kwh": f"{train_fit_energy:.12f}",
        "train_fit_emissions_kg_co2eq": f"{train_fit_emissions:.12f}",
        "train_fit_gpu_energy_kwh": f"{float(train_fit_metrics['gpu_energy_kwh']):.12f}",
        "infer_generate_project_name": infer_generate_project,
        "infer_generate_codecarbon_csv_path": infer_metrics["codecarbon_csv_path"],
        "infer_generate_wall_duration_sec": f"{infer_wall:.6f}",
        "infer_generate_energy_consumed_kwh": f"{infer_energy:.12f}",
        "infer_generate_emissions_kg_co2eq": f"{infer_emissions:.12f}",
        "infer_generate_gpu_energy_kwh": f"{float(infer_metrics['gpu_energy_kwh']):.12f}",
        "measured_total_energy_kwh": f"{measured_total_energy:.12f}",
        "measured_total_emissions_kg_co2eq": f"{measured_total_emissions:.12f}",
        "estimated_full_train_fit_energy_kwh": f"{estimated_full_train_fit_energy:.12f}",
        "estimated_full_train_fit_emissions_kg_co2eq": f"{estimated_full_train_fit_emissions:.12f}",
        "estimated_full_infer_generate_energy_kwh": f"{estimated_full_infer_energy:.12f}",
        "estimated_full_infer_generate_emissions_kg_co2eq": f"{estimated_full_infer_emissions:.12f}",
        "estimated_full_energy_kwh": f"{estimated_full_energy:.12f}",
        "estimated_full_emissions_kg_co2eq": f"{estimated_full_emissions:.12f}",
        "estimation_formula": (
            "processor_once + train_prep_sample + train_load_once + "
            "train_fit_per_step*estimated_full_train_steps + "
            "infer_generate_per_example*full_val_size"
        ),
        "gpu_model": hardware_source.get("gpu_model", ""),
        "cpu_model": hardware_source.get("cpu_model", ""),
        "ram_total_size": hardware_source.get("ram_total_size", ""),
        "codecarbon_version": hardware_source.get("codecarbon_version", ""),
        "notes": (
            "Retrospective Gemma3-12B LoRA calibration. "
            "Uses full bfloat16, no quantization, Gemma3ForConditionalGeneration, "
            "AutoProcessor, attn_implementation=eager, LoRA r=16 alpha=16 dropout=0.0, "
            "batch=2 grad_accum=8 adamw_8bit. Training fit scales by optimizer steps; "
            "inference scales by validation examples."
        ),
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    cleanup_memory(model, trainer)

    print()
    print("=" * 72)
    print("Gemma3-12B LoRA calibration complete")
    print("=" * 72)
    print(f"Predictions saved: {prediction_output_path}")
    print(f"Summary CSV updated: {SUMMARY_CSV}")
    print(f"Measured total calibration emissions: {measured_total_emissions:.12f} kg CO2eq")
    print(f"Estimated full-run emissions: {estimated_full_emissions:.12f} kg CO2eq")
    print("=" * 72)


if __name__ == "__main__":
    main()