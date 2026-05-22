#!/usr/bin/env python3
"""CodeCarbon calibration for Qwen3-14B Think ON LoRA MI and MT.

Supports 6 Qwen3-14B Think ON LoRA calibration runs:
- 056 = Original MI
- 060 = Original MT
- 111 = Gen MI
- 115 = Gen MT
- 116 = Gen+Verify MI
- 120 = Gen+Verify MT

Important:
- This script does NOT touch master_metrics.csv.
- This script does NOT write to original experiment output folders.
- All outputs go under CarbonCalibration-Temp.
- Keeps old Think ON LoRA settings.
- Fixes MT parsing/output columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
ADAPTERS_DIR = TEMP_ROOT / "adapters"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"
PROMPTS_JSON = REPO_ROOT / "prompts.json"
VAL_DIR = REPO_ROOT / "data" / "val"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Qwen3-14B"
METHOD = "LoRA"
THINK = "ON"
UNKNOWN_LABEL = "Unknown"

SINGLE_TASKS = ["MI", "ML", "PG", "Act"]

RUNS = [
    {
        "run_id": "056",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "None",
        "train_file": "data/train/mistake_identification_train.jsonl",
        "val_file": "mistake_identification_val.jsonl",
    },
    {
        "run_id": "060",
        "task": "MT",
        "task_group": "MT",
        "aug": "None",
        "train_file": "data/train/multitask_train.jsonl",
        "val_file": "multitask_val.jsonl",
    },
    {
        "run_id": "111",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen",
        "train_file": "data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl",
        "val_file": "mistake_identification_val.jsonl",
    },
    {
        "run_id": "115",
        "task": "MT",
        "task_group": "MT",
        "aug": "Qwen3-Gen",
        "train_file": "data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl",
        "val_file": "multitask_val.jsonl",
    },
    {
        "run_id": "116",
        "task": "MI",
        "task_group": "SingleTask",
        "aug": "Qwen3-Gen+Verify",
        "train_file": "data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl",
        "val_file": "mistake_identification_val.jsonl",
    },
    {
        "run_id": "120",
        "task": "MT",
        "task_group": "MT",
        "aug": "Qwen3-Gen+Verify",
        "train_file": "data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl",
        "val_file": "multitask_val.jsonl",
    },
]

SINGLE_TASK_PROMPT_KEYS = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}

MT_FIELD_ALIASES = {
    "mistakeidentification": "MI",
    "mistakelocation": "ML",
    "providingguidance": "PG",
    "actionability": "Act",
    "mi": "MI",
    "ml": "ML",
    "pg": "PG",
    "act": "Act",
}

MT_REGEX_ALIASES = {
    "MI": ["Mistake_Identification", "Mistake Identification", "MI"],
    "ML": ["Mistake_Location", "Mistake Location", "ML"],
    "PG": ["Providing_Guidance", "Providing Guidance", "PG"],
    "Act": ["Actionability", "Act"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

    return records


def write_predictions_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "emissions_rate_kg_per_sec": safe_float(row.get("emissions_rate")),
        "cpu_energy_kwh": safe_float(row.get("cpu_energy")),
        "gpu_energy_kwh": safe_float(row.get("gpu_energy")),
        "ram_energy_kwh": safe_float(row.get("ram_energy")),
        "cpu_power_w": safe_float(row.get("cpu_power")),
        "gpu_power_w": safe_float(row.get("gpu_power")),
        "ram_power_w": safe_float(row.get("ram_power")),
        "gpu_model": row.get("gpu_model", ""),
        "cpu_model": row.get("cpu_model", ""),
        "ram_total_size": row.get("ram_total_size", ""),
        "codecarbon_version": row.get("codecarbon_version", ""),
    }


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


def get_special_token_content(value: object, default: str | None = None) -> str | None:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content

    return default


def load_tokenizer(model_path: str) -> AutoTokenizer | PreTrainedTokenizerFast:
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (AttributeError, TypeError, ValueError):
        base_model_path = Path(model_path)
        tokenizer_config = load_json(base_model_path / "tokenizer_config.json")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(base_model_path / "tokenizer.json"),
            bos_token=get_special_token_content(tokenizer_config.get("bos_token"), "<s>"),
            eos_token=get_special_token_content(tokenizer_config.get("eos_token"), "</s>"),
            unk_token=get_special_token_content(tokenizer_config.get("unk_token"), "<unk>"),
            pad_token=get_special_token_content(tokenizer_config.get("pad_token"), "<pad>"),
        )

        tokenizer.chat_template = tokenizer_config.get("chat_template")
        return tokenizer


def slugify_aug(aug: str) -> str:
    if aug == "None":
        return "orig"
    return aug.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")


def apply_chat_template_think_on(tokenizer: AutoTokenizer, messages, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def select_prompt(prompts: dict, task: str) -> str:
    if task == "MT":
        prompt = prompts["multitask_thinking"]["prompt"]
    else:
        prompt_key = SINGLE_TASK_PROMPT_KEYS[task]
        prompt = prompts["single_task_thinking"]["prompts"][prompt_key]

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt registry did not contain a usable prompt for task {task}.")

    return prompt


def extract_last_role_content(record: dict, role: str, source_path: Path, row_index: int) -> str:
    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError(f"{source_path} row {row_index} is missing a valid messages list.")

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{source_path} row {row_index} has invalid {role} content.")

            return content

    raise ValueError(f"{source_path} row {row_index} has no {role} message.")


def build_training_messages(record: dict, prompt: str, source_path: Path, row_index: int) -> list[dict]:
    user_content = extract_last_role_content(record, "user", source_path, row_index)
    assistant_content = extract_last_role_content(record, "assistant", source_path, row_index)

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def build_training_dataset_from_records(
    tokenizer: AutoTokenizer,
    records: Sequence[dict],
    source_path: Path,
    prompt: str,
    source_indices: Sequence[int],
) -> Dataset:
    formatted_rows = []

    for local_index, record in enumerate(records):
        source_row_index = source_indices[local_index]
        messages = build_training_messages(record, prompt, source_path, source_row_index)

        formatted_rows.append(
            {
                "text": apply_chat_template_think_on(
                    tokenizer=tokenizer,
                    messages=messages,
                    add_generation_prompt=False,
                )
            }
        )

    return Dataset.from_list(formatted_rows)


def canonicalize_text(text: str) -> str:
    stripped = text.strip().strip("\"'`")
    stripped = stripped.strip(" .,!?:;")
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


def normalize_label(value: object) -> str:
    if value is None:
        return UNKNOWN_LABEL

    text = str(value).strip()

    if not text:
        return UNKNOWN_LABEL

    candidates = [text]
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")

    if first_line:
        candidates.append(first_line)

    for candidate in list(candidates):
        if ":" in candidate:
            _, tail = candidate.split(":", 1)
            candidates.append(tail.strip())

    seen = set()

    for candidate in candidates:
        cleaned = canonicalize_text(candidate)

        if cleaned in seen:
            continue

        seen.add(cleaned)

        if cleaned == "yes":
            return "Yes"

        if cleaned == "no":
            return "No"

        if cleaned == "to some extent":
            return "To some extent"

    return UNKNOWN_LABEL


def strip_think_blocks(raw_output: str) -> str:
    if not raw_output:
        return raw_output

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^\s*<think>.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return cleaned.strip()


def recover_single_task_label(raw_output: str) -> str:
    cleaned = strip_think_blocks(raw_output)
    parsed = normalize_label(cleaned)

    if parsed != UNKNOWN_LABEL:
        return parsed

    evaluation_matches = list(
        re.finditer(
            r"evaluation\s*:\s*(yes|no|to some extent)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )

    if evaluation_matches:
        return normalize_label(evaluation_matches[-1].group(1))

    trailing_match = re.search(
        r"(yes|no|to some extent)\s*$",
        cleaned,
        flags=re.IGNORECASE,
    )

    if trailing_match:
        return normalize_label(trailing_match.group(1))

    return UNKNOWN_LABEL


def regex_for_alias(alias: str) -> str:
    parts = re.split(r"[\s_-]+", alias.strip())
    return r"[\s_-]*".join(re.escape(part) for part in parts if part)


def parse_single_task_output(raw_output: str) -> str:
    return recover_single_task_label(raw_output)


def parse_multitask_output(raw_output: str) -> Dict[str, str]:
    cleaned = strip_think_blocks(raw_output)
    parsed = {dimension: UNKNOWN_LABEL for dimension in SINGLE_TASKS}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        field_name, raw_value = line.split(":", 1)
        field_key = canonicalize_text(field_name).replace(" ", "")
        dimension = MT_FIELD_ALIASES.get(field_key)

        if dimension and parsed[dimension] == UNKNOWN_LABEL:
            parsed[dimension] = normalize_label(raw_value)

    for dimension in SINGLE_TASKS:
        if parsed[dimension] != UNKNOWN_LABEL:
            continue

        alias_regex = "|".join(regex_for_alias(alias) for alias in MT_REGEX_ALIASES[dimension])

        match = re.search(
            rf"(?:{alias_regex})\s*:\s*(Yes|No|To some extent)",
            cleaned,
            flags=re.IGNORECASE,
        )

        if match:
            parsed[dimension] = normalize_label(match.group(1))

    return parsed


def extract_last_user_content(record: dict, source_path: Path, row_index: int) -> str:
    return extract_last_role_content(record, "user", source_path, row_index)


def build_messages(record: dict, prompt: str, source_path: Path, row_index: int) -> List[dict]:
    user_content = extract_last_user_content(record, source_path, row_index)

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_retry_messages(original_messages: List[dict], retry_prompt: str, prior_output: str) -> List[dict]:
    return original_messages + [
        {"role": "assistant", "content": prior_output},
        {"role": "user", "content": retry_prompt},
    ]


def render_prompt_texts(
    tokenizer: AutoTokenizer,
    records: Sequence[dict],
    prompt: str,
    source_path: Path,
    source_indices: Sequence[int],
) -> List[str]:
    rendered = []

    for offset, record in enumerate(records):
        row_index = source_indices[offset]
        messages = build_messages(record, prompt, source_path, row_index)

        rendered.append(
            apply_chat_template_think_on(
                tokenizer=tokenizer,
                messages=messages,
                add_generation_prompt=True,
            )
        )

    return rendered


def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: Sequence[str],
    max_new_tokens: int,
) -> List[str]:
    encoded = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_device = get_model_input_device(model)
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    decoded = []

    for batch_index, prompt_length in enumerate(prompt_lengths):
        completion_tokens = outputs[batch_index, int(prompt_length):]
        decoded.append(tokenizer.decode(completion_tokens, skip_special_tokens=True).strip())

    return decoded


def iter_batches_with_indices(
    records: Sequence[dict],
    source_indices: Sequence[int],
    batch_size: int,
) -> Iterable[tuple[int, Sequence[dict], Sequence[int]]]:
    for start_index in range(0, len(records), batch_size):
        yield (
            start_index,
            records[start_index:start_index + batch_size],
            source_indices[start_index:start_index + batch_size],
        )


def get_output_columns(task: str) -> List[str]:
    if task == "MT":
        return ["row_index", "task", "pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"]

    return ["row_index", "task", "pred_label", "raw_output"]


def load_train_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer(BASE_MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(f"Loading {MODEL_NAME} in full bfloat16 from {BASE_MODEL_PATH} ...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_infer_model_and_tokenizer(adapter_path: Path) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer(BASE_MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(f"Loading {MODEL_NAME} in full bfloat16 from {BASE_MODEL_PATH} ...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    return model, tokenizer


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
        "inference_batch_size",
        "estimated_full_train_steps",
        "measured_train_steps",
        "adapter_output_path",
        "prediction_output_path",
        "unknown_count",
        "retry_count",
        "train_prep_project_name",
        "train_prep_codecarbon_csv_path",
        "train_prep_wall_duration_sec",
        "train_prep_codecarbon_duration_sec",
        "train_prep_energy_consumed_kwh",
        "train_prep_emissions_kg_co2eq",
        "train_prep_energy_per_example_kwh",
        "train_prep_emissions_per_example_kg",
        "train_load_project_name",
        "train_load_codecarbon_csv_path",
        "train_load_wall_duration_sec",
        "train_load_codecarbon_duration_sec",
        "train_load_energy_consumed_kwh",
        "train_load_emissions_kg_co2eq",
        "train_load_gpu_energy_kwh",
        "train_fit_project_name",
        "train_fit_codecarbon_csv_path",
        "train_fit_wall_duration_sec",
        "train_fit_codecarbon_duration_sec",
        "train_fit_energy_consumed_kwh",
        "train_fit_emissions_kg_co2eq",
        "train_fit_gpu_energy_kwh",
        "train_fit_energy_per_step_kwh",
        "train_fit_emissions_per_step_kg",
        "infer_load_project_name",
        "infer_load_codecarbon_csv_path",
        "infer_load_wall_duration_sec",
        "infer_load_codecarbon_duration_sec",
        "infer_load_energy_consumed_kwh",
        "infer_load_emissions_kg_co2eq",
        "infer_load_gpu_energy_kwh",
        "infer_generate_project_name",
        "infer_generate_codecarbon_csv_path",
        "infer_generate_wall_duration_sec",
        "infer_generate_codecarbon_duration_sec",
        "infer_generate_energy_consumed_kwh",
        "infer_generate_emissions_kg_co2eq",
        "infer_generate_gpu_energy_kwh",
        "infer_generate_energy_per_example_kwh",
        "infer_generate_emissions_per_example_kg",
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


def build_project_base(run_id: str, task: str, aug: str) -> str:
    return f"calib_run{run_id}_qwen14b_thinkon_lora_{slugify_aug(aug)}_{task.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeCarbon calibration for Qwen3-14B Think ON LoRA MI/MT."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=[run["run_id"] for run in RUNS],
        help="056/060 original, 111/115 Gen, 116/120 Gen+Verify.",
    )

    parser.add_argument("--train-sample-fraction", type=float, default=0.25)
    parser.add_argument("--min-train-samples", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--train-sample-seed", type=int, default=42)

    parser.add_argument("--val-sample-fraction", type=float, default=0.10)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--max-val-samples", type=int, default=100)
    parser.add_argument("--val-sample-seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=1)

    args = parser.parse_args()

    if args.train_sample_fraction <= 0 or args.train_sample_fraction > 1:
        raise ValueError("--train-sample-fraction must be in (0, 1].")
    if args.val_sample_fraction <= 0 or args.val_sample_fraction > 1:
        raise ValueError("--val-sample-fraction must be in (0, 1].")
    if args.min_train_samples <= 0 or args.max_train_samples < args.min_train_samples:
        raise ValueError("Invalid train sample bounds.")
    if args.min_val_samples <= 0 or args.max_val_samples < args.min_val_samples:
        raise ValueError("Invalid validation sample bounds.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be greater than zero.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be greater than zero.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be greater than zero.")

    ensure_dirs()
    set_seed(args.seed)

    run = next(item for item in RUNS if item["run_id"] == args.run_id)

    run_id = run["run_id"]
    task = run["task"]
    aug = run["aug"]
    project_base = build_project_base(run_id, task, aug)

    train_path = REPO_ROOT / run["train_file"]
    val_path = VAL_DIR / run["val_file"]

    adapter_root = ADAPTERS_DIR / project_base
    adapter_dir = adapter_root / task
    prediction_output_path = OUTPUTS_DIR / f"{project_base}_sample_predictions.csv"

    train_prep_project = f"{project_base}_train_prep"
    train_load_project = f"{project_base}_train_load"
    train_fit_project = f"{project_base}_train_fit_save"
    infer_load_project = f"{project_base}_infer_load"
    infer_generate_project = f"{project_base}_infer_generate"

    print("=" * 72)
    print(f"Carbon calibration run: {project_base}")
    print(f"Original run id: {run_id}")
    print(f"Task: {task}")
    print(f"Aug: {aug}")
    print(f"Think: {THINK}")
    print(f"Model: {MODEL_NAME}")
    print(f"Method: {METHOD}")
    print("Mode: full bfloat16, no quantization")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Train file: {train_path}")
    print(f"Val file: {val_path}")
    print(f"Adapter output: {adapter_dir}")
    print(f"Prediction output: {prediction_output_path}")
    print(f"Emissions dir: {EMISSIONS_DIR}")
    print("=" * 72)

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing val file: {val_path}")
    if not PROMPTS_JSON.is_file():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_JSON}")

    prompts = load_json(PROMPTS_JSON)
    prompt = select_prompt(prompts, task)
    retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multi = prompts["retry_prompts"]["prompts"]["multitask"]

    full_train_records = load_jsonl_records(train_path)
    full_val_records = load_jsonl_records(val_path)

    full_train_size = len(full_train_records)
    full_val_size = len(full_val_records)

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

    train_records = [full_train_records[index] for index in train_indices]
    val_records = [full_val_records[index] for index in val_indices]

    train_sample_size = len(train_records)
    val_sample_size = len(val_records)

    train_sample_fraction_actual = train_sample_size / full_train_size
    val_sample_fraction_actual = val_sample_size / full_val_size

    estimated_full_train_steps = estimate_update_steps(
        num_examples=full_train_size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
    )

    print(f"Full train size: {full_train_size}")
    print(f"Train sample size: {train_sample_size}")
    print(f"Train sample fraction: {train_sample_fraction_actual:.6f}")
    print(f"Full validation size: {full_val_size}")
    print(f"Validation sample size: {val_sample_size}")
    print(f"Validation sample fraction: {val_sample_fraction_actual:.6f}")
    print(f"Estimated full train optimizer steps: {estimated_full_train_steps}")

    prep_start = time.time()
    with track_emissions(train_prep_project, output_dir=EMISSIONS_DIR):
        train_tokenizer = load_tokenizer(BASE_MODEL_PATH)

        if train_tokenizer.pad_token_id is None:
            train_tokenizer.pad_token = train_tokenizer.eos_token

        train_tokenizer.padding_side = "right"

        train_dataset = build_training_dataset_from_records(
            tokenizer=train_tokenizer,
            records=train_records,
            source_path=train_path,
            prompt=prompt,
            source_indices=train_indices,
        )
    train_prep_wall = time.time() - prep_start

    train_load_start = time.time()
    with track_emissions(train_load_project, output_dir=EMISSIONS_DIR):
        train_model, train_tokenizer = load_train_model_and_tokenizer()
    train_load_wall = time.time() - train_load_start

    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=train_tokenizer,
    )

    train_fit_start = time.time()
    with track_emissions(train_fit_project, output_dir=EMISSIONS_DIR):
        trainer.train()
        trainer.save_model(str(adapter_dir))
        train_tokenizer.save_pretrained(str(adapter_dir))

        manifest = {
            "model": MODEL_NAME,
            "base_model_path": BASE_MODEL_PATH,
            "method": METHOD,
            "task": task,
            "original_run_id": run_id,
            "aug": aug,
            "think": THINK,
            "train_jsonl": str(train_path),
            "full_train_size": full_train_size,
            "train_sample_size": train_sample_size,
            "train_sample_fraction_actual": train_sample_fraction_actual,
            "adapter_out": str(adapter_dir),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "timestamp_utc": utc_now(),
        }

        with (adapter_dir / "train_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    train_fit_wall = time.time() - train_fit_start

    measured_train_steps = int(getattr(trainer.state, "global_step", 0) or 0)

    if measured_train_steps <= 0:
        measured_train_steps = estimate_update_steps(
            num_examples=train_sample_size,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
        )

    print(f"Measured train optimizer steps: {measured_train_steps}")

    del trainer
    del train_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    infer_load_start = time.time()
    with track_emissions(infer_load_project, output_dir=EMISSIONS_DIR):
        infer_model, infer_tokenizer = load_infer_model_and_tokenizer(adapter_dir)
    infer_load_wall = time.time() - infer_load_start

    output_columns = get_output_columns(task)
    all_rows = []
    unknown_count = 0
    retry_count = 0

    infer_generate_start = time.time()
    with track_emissions(infer_generate_project, output_dir=EMISSIONS_DIR):
        for batch_start, batch_records, batch_source_indices in iter_batches_with_indices(
            val_records,
            val_indices,
            args.inference_batch_size,
        ):
            prompt_texts = render_prompt_texts(
                tokenizer=infer_tokenizer,
                records=batch_records,
                prompt=prompt,
                source_path=val_path,
                source_indices=batch_source_indices,
            )

            raw_outputs = generate_batch(
                model=infer_model,
                tokenizer=infer_tokenizer,
                prompt_texts=prompt_texts,
                max_new_tokens=args.max_new_tokens,
            )

            for offset, raw_output in enumerate(raw_outputs):
                source_row_index = batch_source_indices[offset]
                record = batch_records[offset]
                base_messages = build_messages(record, prompt, val_path, source_row_index)

                if task == "MT":
                    parsed = parse_multitask_output(raw_output)

                    if UNKNOWN_LABEL in parsed.values():
                        retry_count += 1
                        retry_messages = build_retry_messages(base_messages, retry_multi, raw_output)
                        retry_prompt_text = apply_chat_template_think_on(
                            tokenizer=infer_tokenizer,
                            messages=retry_messages,
                            add_generation_prompt=True,
                        )

                        retry_output = generate_batch(
                            model=infer_model,
                            tokenizer=infer_tokenizer,
                            prompt_texts=[retry_prompt_text],
                            max_new_tokens=args.max_new_tokens,
                        )[0]

                        retry_parsed = parse_multitask_output(retry_output)

                        for dim in parsed:
                            if parsed[dim] == UNKNOWN_LABEL:
                                parsed[dim] = retry_parsed[dim]

                        raw_output = retry_output

                    if UNKNOWN_LABEL in parsed.values():
                        unknown_count += 1

                    all_rows.append(
                        {
                            "row_index": source_row_index,
                            "task": task,
                            "pred_mi": parsed["MI"],
                            "pred_ml": parsed["ML"],
                            "pred_pg": parsed["PG"],
                            "pred_act": parsed["Act"],
                            "raw_output": raw_output,
                        }
                    )

                else:
                    parsed = parse_single_task_output(raw_output)

                    if parsed == UNKNOWN_LABEL:
                        retry_count += 1
                        retry_messages = build_retry_messages(base_messages, retry_single, raw_output)
                        retry_prompt_text = apply_chat_template_think_on(
                            tokenizer=infer_tokenizer,
                            messages=retry_messages,
                            add_generation_prompt=True,
                        )

                        retry_output = generate_batch(
                            model=infer_model,
                            tokenizer=infer_tokenizer,
                            prompt_texts=[retry_prompt_text],
                            max_new_tokens=args.max_new_tokens,
                        )[0]

                        retry_parsed = parse_single_task_output(retry_output)

                        if retry_parsed != UNKNOWN_LABEL:
                            parsed = retry_parsed

                        raw_output = retry_output

                    if parsed == UNKNOWN_LABEL:
                        unknown_count += 1

                    all_rows.append(
                        {
                            "row_index": source_row_index,
                            "task": task,
                            "pred_label": parsed,
                            "raw_output": raw_output,
                        }
                    )

            if len(all_rows) == 1 or len(all_rows) % 10 == 0 or len(all_rows) == val_sample_size:
                elapsed = time.time() - infer_generate_start
                print(
                    f"Progress: {len(all_rows)}/{val_sample_size} | "
                    f"Unknowns: {unknown_count} | Retries: {retry_count} | "
                    f"Inference time: {elapsed / 60:.2f} min",
                    flush=True,
                )

    infer_generate_wall = time.time() - infer_generate_start

    write_predictions_csv(prediction_output_path, output_columns, all_rows)

    train_prep = get_codecarbon_metrics(train_prep_project)
    train_load = get_codecarbon_metrics(train_load_project)
    train_fit = get_codecarbon_metrics(train_fit_project)
    infer_load = get_codecarbon_metrics(infer_load_project)
    infer_generate = get_codecarbon_metrics(infer_generate_project)

    train_prep_energy = float(train_prep["energy_consumed_kwh"])
    train_prep_emissions = float(train_prep["emissions_kg_co2eq"])

    train_load_energy = float(train_load["energy_consumed_kwh"])
    train_load_emissions = float(train_load["emissions_kg_co2eq"])

    train_fit_energy = float(train_fit["energy_consumed_kwh"])
    train_fit_emissions = float(train_fit["emissions_kg_co2eq"])

    infer_load_energy = float(infer_load["energy_consumed_kwh"])
    infer_load_emissions = float(infer_load["emissions_kg_co2eq"])

    infer_generate_energy = float(infer_generate["energy_consumed_kwh"])
    infer_generate_emissions = float(infer_generate["emissions_kg_co2eq"])

    train_prep_energy_per_example = train_prep_energy / train_sample_size
    train_prep_emissions_per_example = train_prep_emissions / train_sample_size

    train_fit_energy_per_step = train_fit_energy / measured_train_steps
    train_fit_emissions_per_step = train_fit_emissions / measured_train_steps

    infer_generate_energy_per_example = infer_generate_energy / val_sample_size
    infer_generate_emissions_per_example = infer_generate_emissions / val_sample_size

    measured_total_energy = (
        train_prep_energy
        + train_load_energy
        + train_fit_energy
        + infer_load_energy
        + infer_generate_energy
    )

    measured_total_emissions = (
        train_prep_emissions
        + train_load_emissions
        + train_fit_emissions
        + infer_load_emissions
        + infer_generate_emissions
    )

    estimated_full_train_prep_energy = train_prep_energy_per_example * full_train_size
    estimated_full_train_prep_emissions = train_prep_emissions_per_example * full_train_size

    estimated_full_train_fit_energy = train_fit_energy_per_step * estimated_full_train_steps
    estimated_full_train_fit_emissions = train_fit_emissions_per_step * estimated_full_train_steps

    estimated_full_infer_generate_energy = infer_generate_energy_per_example * full_val_size
    estimated_full_infer_generate_emissions = infer_generate_emissions_per_example * full_val_size

    estimated_full_energy = (
        estimated_full_train_prep_energy
        + train_load_energy
        + estimated_full_train_fit_energy
        + infer_load_energy
        + estimated_full_infer_generate_energy
    )

    estimated_full_emissions = (
        estimated_full_train_prep_emissions
        + train_load_emissions
        + estimated_full_train_fit_emissions
        + infer_load_emissions
        + estimated_full_infer_generate_emissions
    )

    hardware_source = infer_generate or train_fit or train_load

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
        "train_file": run["train_file"],
        "val_file": run["val_file"],
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
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "inference_batch_size": args.inference_batch_size,
        "estimated_full_train_steps": estimated_full_train_steps,
        "measured_train_steps": measured_train_steps,
        "adapter_output_path": str(adapter_dir),
        "prediction_output_path": str(prediction_output_path),
        "unknown_count": unknown_count,
        "retry_count": retry_count,
        "train_prep_project_name": train_prep_project,
        "train_prep_codecarbon_csv_path": train_prep["codecarbon_csv_path"],
        "train_prep_wall_duration_sec": f"{train_prep_wall:.6f}",
        "train_prep_codecarbon_duration_sec": f"{float(train_prep['codecarbon_duration_sec']):.6f}",
        "train_prep_energy_consumed_kwh": f"{train_prep_energy:.12f}",
        "train_prep_emissions_kg_co2eq": f"{train_prep_emissions:.12f}",
        "train_prep_energy_per_example_kwh": f"{train_prep_energy_per_example:.12f}",
        "train_prep_emissions_per_example_kg": f"{train_prep_emissions_per_example:.12f}",
        "train_load_project_name": train_load_project,
        "train_load_codecarbon_csv_path": train_load["codecarbon_csv_path"],
        "train_load_wall_duration_sec": f"{train_load_wall:.6f}",
        "train_load_codecarbon_duration_sec": f"{float(train_load['codecarbon_duration_sec']):.6f}",
        "train_load_energy_consumed_kwh": f"{train_load_energy:.12f}",
        "train_load_emissions_kg_co2eq": f"{train_load_emissions:.12f}",
        "train_load_gpu_energy_kwh": f"{float(train_load['gpu_energy_kwh']):.12f}",
        "train_fit_project_name": train_fit_project,
        "train_fit_codecarbon_csv_path": train_fit["codecarbon_csv_path"],
        "train_fit_wall_duration_sec": f"{train_fit_wall:.6f}",
        "train_fit_codecarbon_duration_sec": f"{float(train_fit['codecarbon_duration_sec']):.6f}",
        "train_fit_energy_consumed_kwh": f"{train_fit_energy:.12f}",
        "train_fit_emissions_kg_co2eq": f"{train_fit_emissions:.12f}",
        "train_fit_gpu_energy_kwh": f"{float(train_fit['gpu_energy_kwh']):.12f}",
        "train_fit_energy_per_step_kwh": f"{train_fit_energy_per_step:.12f}",
        "train_fit_emissions_per_step_kg": f"{train_fit_emissions_per_step:.12f}",
        "infer_load_project_name": infer_load_project,
        "infer_load_codecarbon_csv_path": infer_load["codecarbon_csv_path"],
        "infer_load_wall_duration_sec": f"{infer_load_wall:.6f}",
        "infer_load_codecarbon_duration_sec": f"{float(infer_load['codecarbon_duration_sec']):.6f}",
        "infer_load_energy_consumed_kwh": f"{infer_load_energy:.12f}",
        "infer_load_emissions_kg_co2eq": f"{infer_load_emissions:.12f}",
        "infer_load_gpu_energy_kwh": f"{float(infer_load['gpu_energy_kwh']):.12f}",
        "infer_generate_project_name": infer_generate_project,
        "infer_generate_codecarbon_csv_path": infer_generate["codecarbon_csv_path"],
        "infer_generate_wall_duration_sec": f"{infer_generate_wall:.6f}",
        "infer_generate_codecarbon_duration_sec": f"{float(infer_generate['codecarbon_duration_sec']):.6f}",
        "infer_generate_energy_consumed_kwh": f"{infer_generate_energy:.12f}",
        "infer_generate_emissions_kg_co2eq": f"{infer_generate_emissions:.12f}",
        "infer_generate_gpu_energy_kwh": f"{float(infer_generate['gpu_energy_kwh']):.12f}",
        "infer_generate_energy_per_example_kwh": f"{infer_generate_energy_per_example:.12f}",
        "infer_generate_emissions_per_example_kg": f"{infer_generate_emissions_per_example:.12f}",
        "measured_total_energy_kwh": f"{measured_total_energy:.12f}",
        "measured_total_emissions_kg_co2eq": f"{measured_total_emissions:.12f}",
        "estimated_full_train_prep_energy_kwh": f"{estimated_full_train_prep_energy:.12f}",
        "estimated_full_train_prep_emissions_kg_co2eq": f"{estimated_full_train_prep_emissions:.12f}",
        "estimated_full_train_fit_energy_kwh": f"{estimated_full_train_fit_energy:.12f}",
        "estimated_full_train_fit_emissions_kg_co2eq": f"{estimated_full_train_fit_emissions:.12f}",
        "estimated_full_infer_generate_energy_kwh": f"{estimated_full_infer_generate_energy:.12f}",
        "estimated_full_infer_generate_emissions_kg_co2eq": f"{estimated_full_infer_generate_emissions:.12f}",
        "estimated_full_energy_kwh": f"{estimated_full_energy:.12f}",
        "estimated_full_emissions_kg_co2eq": f"{estimated_full_emissions:.12f}",
        "estimation_formula": (
            "train_prep_per_example*full_train_size + train_load_once + "
            "train_fit_per_step*estimated_full_train_steps + infer_load_once + "
            "infer_generate_per_example*full_val_size"
        ),
        "gpu_model": hardware_source.get("gpu_model", ""),
        "cpu_model": hardware_source.get("cpu_model", ""),
        "ram_total_size": hardware_source.get("ram_total_size", ""),
        "codecarbon_version": hardware_source.get("codecarbon_version", ""),
        "notes": (
            "Retrospective Qwen3-14B Think ON LoRA calibration. "
            "Full bfloat16, no quantization, device_map=auto, enable_thinking=True. "
            "Training uses thinking prompts and CoT chat template. "
            "LoRA r=16 alpha=32 dropout=0.05 targeting q/k/v/o/gate/up/down. "
            "batch=1 grad_accum=16 epochs=3 lr=2e-4 max_length=2048. "
            "Inference uses max_new_tokens=1024 and fixed MT parsing to output pred_mi/pred_ml/pred_pg/pred_act."
        ),
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    print()
    print("=" * 72)
    print("Qwen3-14B Think ON LoRA calibration complete")
    print("=" * 72)
    print(f"Adapter saved: {adapter_dir}")
    print(f"Predictions saved: {prediction_output_path}")
    print(f"Summary CSV updated: {SUMMARY_CSV}")
    print()
    print(f"Measured train prep emissions: {train_prep_emissions:.12f} kg CO2eq")
    print(f"Measured train load emissions: {train_load_emissions:.12f} kg CO2eq")
    print(f"Measured train fit/save emissions: {train_fit_emissions:.12f} kg CO2eq")
    print(f"Measured inference load emissions: {infer_load_emissions:.12f} kg CO2eq")
    print(f"Measured inference generate emissions: {infer_generate_emissions:.12f} kg CO2eq")
    print(f"Measured total calibration emissions: {measured_total_emissions:.12f} kg CO2eq")
    print()
    print(f"Estimated full-run emissions: {estimated_full_emissions:.12f} kg CO2eq")
    print()
    print("Formula used:")
    print("  train_prep_per_example * full_train_size")
    print("  + train_load_once")
    print("  + train_fit_per_step * estimated_full_train_steps")
    print("  + infer_load_once")
    print("  + infer_generate_per_example * full_val_size")
    print("=" * 72)


if __name__ == "__main__":
    main()