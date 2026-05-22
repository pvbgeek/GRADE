#!/usr/bin/env python3
"""CodeCarbon calibration for Gemma3-27B LoRA MI and MT.

Supports 6 calibration runs:
- 046 = Original MI
- 050 = Original MT
- 081 = Gen MI
- 085 = Gen MT
- 106 = Gen+Verify MI
- 110 = Gen+Verify MT

Important:
- Same model/training settings as the old Gemma3-27B LoRA script.
- No tryeval.
- No master metrics update.
- All outputs go under CarbonCalibration-Temp.
"""

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
ADAPTERS_DIR = TEMP_ROOT / "adapters"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

MODEL_NAME = "Gemma3-27B"
MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3-27b"
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
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
SEED = 3407
LOGGING_STEPS = 10
MAX_SEQ_LENGTH = 768
MAX_NEW_TOKENS = 64

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


RUNS: Dict[str, dict] = {
    "046": {
        "run_id": "046",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "None",
        "train_jsonl": str(REPO_ROOT / "data/train/mistake_identification_train.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "050": {
        "run_id": "050",
        "task": "MT",
        "task_group": "MT",
        "aug": "None",
        "train_jsonl": str(REPO_ROOT / "data/train/multitask_train.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/multitask_val.jsonl"),
    },
    "081": {
        "run_id": "081",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "085": {
        "run_id": "085",
        "task": "MT",
        "task_group": "MT",
        "aug": "Qwen3-Gen",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/multitask_val.jsonl"),
    },
    "106": {
        "run_id": "106",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": str(REPO_ROOT / "data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl"),
        "val_jsonl": str(REPO_ROOT / "data/val/mistake_identification_val.jsonl"),
    },
    "110": {
        "run_id": "110",
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
    return f"calib_run{run_id}_gemma3_27b_lora_{slugify_aug(aug)}_{task.lower()}"


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


def canonicalize_text(text: str) -> str:
    text = str(text).strip().strip("\"'`")
    text = text.strip(" .,!?:;")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def normalize_label(text: str) -> str:
    if not text:
        return UNKNOWN_LABEL

    text = str(text).replace("<end_of_turn>", "").replace("</s>", "").strip()

    candidates = [text]

    for line in text.splitlines():
        line = line.strip()

        if line:
            candidates.append(line)

        if ":" in line:
            candidates.append(line.split(":", 1)[1].strip())

    for candidate in candidates:
        cleaned = canonicalize_text(candidate)

        if "to some extent" in cleaned or "to some extend" in cleaned:
            return "To some extent"

        if cleaned == "no" or cleaned.startswith("no "):
            return "No"

        if cleaned == "yes" or cleaned.startswith("yes "):
            return "Yes"

    return UNKNOWN_LABEL


def parse_multitask_output(text: str) -> dict:
    text = str(text).replace("<end_of_turn>", "").replace("</s>", "").strip()

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

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        field, value = line.split(":", 1)
        key = (
            field.strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace(".", "")
            .replace("1", "")
            .replace("2", "")
            .replace("3", "")
            .replace("4", "")
        )

        col = field_map.get(key)

        if col:
            result[col] = normalize_label(value.strip())

    regex_patterns = {
        "pred_mi": r"(mistake[\s_-]*identification|mi)\s*[:\-]\s*(yes|no|to some extent)",
        "pred_ml": r"(mistake[\s_-]*location|ml)\s*[:\-]\s*(yes|no|to some extent)",
        "pred_pg": r"(providing[\s_-]*guidance|pg)\s*[:\-]\s*(yes|no|to some extent)",
        "pred_act": r"(actionability|act)\s*[:\-]\s*(yes|no|to some extent)",
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


def print_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[GPU Memory] {label} | "
        f"allocated={allocated:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"max={max_allocated:.2f}GB",
        flush=True,
    )


def force_text_lora_trainable(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()

    text_tensors = 0
    text_params = 0
    vision_tensors = 0
    vision_params = 0

    for name, param in model.named_parameters():
        lname = name.lower()

        is_lora = "lora" in lname
        is_vision = "vision_tower" in lname or "vision_model" in lname
        is_text = "language_model" in lname or "text_model" in lname

        if is_lora and is_vision:
            param.requires_grad = False
            vision_tensors += 1
            vision_params += param.numel()

        elif is_lora and is_text:
            param.requires_grad = True
            text_tensors += 1
            text_params += param.numel()

    if text_params == 0:
        print("\n[Warning] No params matched language_model/text_model. Falling back to non-vision LoRA params.")

        for name, param in model.named_parameters():
            lname = name.lower()

            is_lora = "lora" in lname
            is_vision = "vision_tower" in lname or "vision_model" in lname

            if is_lora and not is_vision:
                param.requires_grad = True
                text_tensors += 1
                text_params += param.numel()

    print("\n[Text LoRA Trainable]")
    print(f"  text LoRA tensors trainable: {text_tensors}")
    print(f"  text LoRA params trainable:  {text_params:,}")
    print(f"  vision LoRA tensors frozen:  {vision_tensors}")
    print(f"  vision LoRA params frozen:   {vision_params:,}")

    if text_params == 0:
        raise RuntimeError("No text LoRA params found. Check Gemma3 module names.")

    return model


class TrainDebugCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps
        self.tracked_name = None
        self.initial_tensor = None

    def _find_trainable_param(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                return name, param

        return None, None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainable_count = 0
        trainable_tensors = 0
        vision_trainable = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_count += param.numel()
                trainable_tensors += 1

                if "vision_tower" in name.lower() or "vision_model" in name.lower():
                    vision_trainable += param.numel()

        print("\n[Trainable Check]")
        print(f"  trainable tensors: {trainable_tensors}")
        print(f"  trainable params:  {trainable_count:,}")
        print(f"  trainable vision params: {vision_trainable:,}")

        name, param = self._find_trainable_param(model)

        if param is None:
            print("[Trainable Check] ERROR: No trainable parameter found.")
            return

        self.tracked_name = name
        self.initial_tensor = param.detach().float().cpu().clone()

        print("\n[Weight Tracking]")
        print(f"  tracking: {self.tracked_name}")
        print(f"  shape: {tuple(param.shape)}")
        print(f"  requires_grad: {param.requires_grad}")

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return

        total_sq = 0.0
        max_abs = 0.0
        tensors_with_grad = 0
        trainable_tensors = 0

        for _, param in model.named_parameters():
            if param.requires_grad:
                trainable_tensors += 1

                if param.grad is not None:
                    grad = param.grad.detach().float()
                    norm = grad.norm().item()
                    total_sq += norm * norm
                    max_abs = max(max_abs, grad.abs().max().item())
                    tensors_with_grad += 1

        grad_norm = total_sq ** 0.5

        print(
            "\n[Real Grad Check]"
            f" step={state.global_step}"
            f" | trainable_grad_norm={grad_norm:.8f}"
            f" | max_abs_grad={max_abs:.8e}"
            f" | tensors_with_grad={tensors_with_grad}/{trainable_tensors}"
        )

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if self.tracked_name is None or self.initial_tensor is None:
            return

        if state.global_step % self.every_n_steps != 0:
            return

        current_param = None

        for name, param in model.named_parameters():
            if name == self.tracked_name:
                current_param = param
                break

        if current_param is None:
            return

        current = current_param.detach().float().cpu()
        delta = (current - self.initial_tensor).abs().mean().item()

        print(f"[Weight Change] step={state.global_step} | mean_abs_delta={delta:.8e}")
        print_gpu_memory(f"after log step {state.global_step}")


def load_processor():
    print(f"\nLoading processor from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    print("Processor loaded.")
    return processor


def load_model():
    print("\nLoading Gemma3-27B in 8-bit...")
    print(f"Model path: {MODEL_PATH}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    elapsed = time.time() - start

    print(f"Base model loaded in {elapsed:.1f}s")
    print_gpu_memory("after base model load")

    return model


def apply_lora(model):
    print("\nApplying text attention + MLP LoRA adapters...")

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

    model = force_text_lora_trainable(model)

    trainable = 0
    total = 0
    vision_trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()

        if param.requires_grad:
            trainable += param.numel()

            if "vision_tower" in name.lower() or "vision_model" in name.lower():
                vision_trainable += param.numel()

    print(f"Final trainable parameters check: {trainable:,} / {total:,}")
    print(f"Final trainable vision params: {vision_trainable:,}")

    if trainable == 0:
        raise RuntimeError("No trainable parameters found after text LoRA filtering.")

    if vision_trainable != 0:
        raise RuntimeError("Vision parameters are still trainable. Stop and fix filtering.")

    print_gpu_memory("after text attention + MLP LoRA attach")

    return model


def cleanup_memory(model=None, trainer=None):
    print("\nCleaning memory...")

    if trainer is not None:
        del trainer

    if model is not None:
        del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print_gpu_memory("after cleanup")


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
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }

    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

    new_tokens = output_ids[0][input_len:]

    return processor.decode(new_tokens, skip_special_tokens=True).strip()


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
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",
        save_strategy="no",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        callbacks=[TrainDebugCallback(every_n_steps=LOGGING_STEPS)],
    )

    return trainer


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
        "max_seq_length",
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
        "estimated_full_train_prep_energy_kwh",
        "estimated_full_train_prep_emissions_kg_co2eq",
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
        description="CodeCarbon calibration for Gemma3-27B LoRA MI/MT."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=sorted(RUNS.keys()),
        help="046/050 original, 081/085 Gen, 106/110 Gen+Verify.",
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
    if args.min_train_samples <= 0 or args.max_train_samples < args.min_train_samples:
        raise ValueError("Invalid train sample bounds.")
    if args.min_val_samples <= 0 or args.max_val_samples < args.min_val_samples:
        raise ValueError("Invalid validation sample bounds.")

    ensure_dirs()

    run = RUNS[args.run_id]
    run_id = run["run_id"]
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
    print("Mode: 8-bit BitsAndBytes, text attention + MLP LoRA")
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
    print(f"Train sample fraction: {train_sample_fraction_actual:.6f}")
    print(f"Full validation size: {full_val_size}")
    print(f"Validation sample size: {val_sample_size}")
    print(f"Validation sample fraction: {val_sample_fraction_actual:.6f}")
    print(f"Estimated full train optimizer steps: {estimated_full_train_steps}")

    processor_start = time.time()
    with track_emissions(processor_project, output_dir=EMISSIONS_DIR):
        processor = load_processor()
    processor_wall = time.time() - processor_start

    prep_start = time.time()
    with track_emissions(train_prep_project, output_dir=EMISSIONS_DIR):
        formatted_train = [format_chat_for_training(ex, processor) for ex in train_examples]
        train_dataset = Dataset.from_list(formatted_train)
    train_prep_wall = time.time() - prep_start

    train_load_start = time.time()
    with track_emissions(train_load_project, output_dir=EMISSIONS_DIR):
        model = load_model()
        model = apply_lora(model)
    train_load_wall = time.time() - train_load_start

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        run_id=run_id,
        output_dir=tmp_output_dir,
    )

    trainer.model = force_text_lora_trainable(trainer.model)

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
    model.config.use_cache = True

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

    train_prep_energy_per_example = train_prep_energy / train_sample_size
    train_prep_emissions_per_example = train_prep_emissions / train_sample_size

    train_fit_energy_per_step = train_fit_energy / measured_train_steps
    train_fit_emissions_per_step = train_fit_emissions / measured_train_steps

    infer_energy_per_example = infer_energy / val_sample_size
    infer_emissions_per_example = infer_emissions / val_sample_size

    estimated_full_train_prep_energy = train_prep_energy_per_example * full_train_size
    estimated_full_train_prep_emissions = train_prep_emissions_per_example * full_train_size

    estimated_full_train_fit_energy = train_fit_energy_per_step * estimated_full_train_steps
    estimated_full_train_fit_emissions = train_fit_emissions_per_step * estimated_full_train_steps

    estimated_full_infer_energy = infer_energy_per_example * full_val_size
    estimated_full_infer_emissions = infer_emissions_per_example * full_val_size

    measured_total_energy = (
        processor_energy
        + train_prep_energy
        + train_load_energy
        + train_fit_energy
        + infer_energy
    )

    measured_total_emissions = (
        processor_emissions
        + train_prep_emissions
        + train_load_emissions
        + train_fit_emissions
        + infer_emissions
    )

    estimated_full_energy = (
        processor_energy
        + estimated_full_train_prep_energy
        + train_load_energy
        + estimated_full_train_fit_energy
        + estimated_full_infer_energy
    )

    estimated_full_emissions = (
        processor_emissions
        + estimated_full_train_prep_emissions
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
        "max_seq_length": MAX_SEQ_LENGTH,
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
        "estimated_full_train_prep_energy_kwh": f"{estimated_full_train_prep_energy:.12f}",
        "estimated_full_train_prep_emissions_kg_co2eq": f"{estimated_full_train_prep_emissions:.12f}",
        "estimated_full_train_fit_energy_kwh": f"{estimated_full_train_fit_energy:.12f}",
        "estimated_full_train_fit_emissions_kg_co2eq": f"{estimated_full_train_fit_emissions:.12f}",
        "estimated_full_infer_generate_energy_kwh": f"{estimated_full_infer_energy:.12f}",
        "estimated_full_infer_generate_emissions_kg_co2eq": f"{estimated_full_infer_emissions:.12f}",
        "estimated_full_energy_kwh": f"{estimated_full_energy:.12f}",
        "estimated_full_emissions_kg_co2eq": f"{estimated_full_emissions:.12f}",
        "estimation_formula": (
            "processor_once + train_prep_per_example*full_train_size + "
            "train_load_once + train_fit_per_step*estimated_full_train_steps + "
            "infer_generate_per_example*full_val_size"
        ),
        "gpu_model": hardware_source.get("gpu_model", ""),
        "cpu_model": hardware_source.get("cpu_model", ""),
        "ram_total_size": hardware_source.get("ram_total_size", ""),
        "codecarbon_version": hardware_source.get("codecarbon_version", ""),
        "notes": (
            "Retrospective Gemma3-27B LoRA calibration. Same model settings as old script: "
            "BitsAndBytes 8-bit, Gemma3ForConditionalGeneration, AutoProcessor use_fast=False, "
            "attn_implementation=eager, dtype=bfloat16, use_cache false during training, "
            "gradient checkpointing enabled before training, LoRA r=16 alpha=16 dropout=0.0, "
            "text attention plus MLP target modules, text-only LoRA trainable filtering, "
            "batch=1 grad_accum=8 max_length=768 adamw_8bit."
        ),
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    cleanup_memory(model=model, trainer=trainer)

    print()
    print("=" * 72)
    print("Gemma3-27B LoRA calibration complete")
    print("=" * 72)
    print(f"Predictions saved: {prediction_output_path}")
    print(f"Summary CSV updated: {SUMMARY_CSV}")
    print(f"Measured total calibration emissions: {measured_total_emissions:.12f} kg CO2eq")
    print(f"Estimated full-run emissions: {estimated_full_emissions:.12f} kg CO2eq")
    print("=" * 72)


if __name__ == "__main__":
    main()