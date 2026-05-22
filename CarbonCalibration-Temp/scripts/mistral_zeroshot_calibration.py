#!/usr/bin/env python3
"""CodeCarbon calibration for Mistral-7B Zero-shot MI and MT.

Temporary calibration-only script.

Supported runs:
- 011 = Mistral-7B Zero-shot MI
- 015 = Mistral-7B Zero-shot MT

Important:
- This script does NOT touch master_metrics.csv.
- This script does NOT write to original experiment output folders.
- All outputs go under CarbonCalibration-Temp.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/Mistral"
PROMPTS_JSON = REPO_ROOT / "prompts.json"
VAL_DIR = REPO_ROOT / "data" / "val"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MODEL_NAME = "Mistral-7B"
METHOD = "Zero-shot"
AUG = "None"
THINK = "N/A"
UNKNOWN_LABEL = "Unknown"

SINGLE_TASKS = ["MI", "ML", "PG", "Act"]

RUNS = [
    {
        "run_id": "011",
        "task": "MI",
        "task_group": "SingleTask",
        "val_file": "mistake_identification_val.jsonl",
        "model": MODEL_NAME,
        "method": METHOD,
        "think": THINK,
        "aug": AUG,
    },
    {
        "run_id": "015",
        "task": "MT",
        "task_group": "MT",
        "val_file": "multitask_val.jsonl",
        "model": MODEL_NAME,
        "method": METHOD,
        "think": THINK,
        "aug": AUG,
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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
        for row in rows:
            writer.writerow(row)


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
        raise ValueError("Validation file is empty.")

    requested = math.ceil(full_size * sample_fraction)
    sample_size = max(min_samples, requested)
    sample_size = min(sample_size, max_samples, full_size)

    rng = random.Random(seed)
    return sorted(rng.sample(range(full_size), sample_size))


def get_special_token_content(value: object, default: str | None = None) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return default


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


def recover_single_task_label(raw_output: str) -> str:
    parsed = normalize_label(raw_output)
    if parsed != UNKNOWN_LABEL:
        return parsed

    evaluation_matches = list(
        re.finditer(
            r"evaluation\s*:\s*(yes|no|to some extent)",
            raw_output,
            flags=re.IGNORECASE,
        )
    )
    if evaluation_matches:
        return normalize_label(evaluation_matches[-1].group(1))

    trailing_match = re.search(
        r"(yes|no|to some extent)\s*$",
        raw_output,
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
    parsed = {dimension: UNKNOWN_LABEL for dimension in SINGLE_TASKS}

    for raw_line in raw_output.splitlines():
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
            raw_output,
            flags=re.IGNORECASE,
        )
        if match:
            parsed[dimension] = normalize_label(match.group(1))

    return parsed


def select_prompt(prompts: dict, task: str) -> str:
    if task == "MT":
        prompt = prompts["multitask_training"]["prompt"]
    else:
        prompt_key = SINGLE_TASK_PROMPT_KEYS[task]
        prompt = prompts["single_task_zero_shot"]["prompts"][prompt_key]

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt registry did not contain a usable prompt for task {task}.")

    return prompt


def extract_last_user_content(record: dict, source_path: Path, row_index: int) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_path} row {row_index} is missing a valid messages list.")

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{source_path} row {row_index} has invalid user content.")
            return content

    raise ValueError(f"{source_path} row {row_index} has no user message.")


def build_messages(record: dict, prompt: str, source_path: Path, row_index: int) -> List[dict]:
    user_content = extract_last_user_content(record, source_path, row_index)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def render_prompt_texts(
    tokenizer: AutoTokenizer,
    records: Sequence[dict],
    prompt: str,
    source_path: Path,
    source_indices: Sequence[int],
) -> List[str]:
    rendered: List[str] = []

    for offset, record in enumerate(records):
        row_index = source_indices[offset]
        messages = build_messages(record, prompt, source_path, row_index)
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
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
    temperature: float,
    top_p: float,
) -> List[str]:
    encoded = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_device = get_model_input_device(model)
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

    generation_kwargs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**generation_kwargs)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    decoded: List[str] = []
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


def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    except (AttributeError, TypeError, ValueError):
        base_model_path = Path(BASE_MODEL_PATH)
        tokenizer_config = load_json(base_model_path / "tokenizer_config.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(base_model_path / "tokenizer.json"),
            bos_token=get_special_token_content(tokenizer_config.get("bos_token"), "<s>"),
            eos_token=get_special_token_content(tokenizer_config.get("eos_token"), "</s>"),
            unk_token=get_special_token_content(tokenizer_config.get("unk_token"), "<unk>"),
            pad_token=get_special_token_content(tokenizer_config.get("pad_token"), "<pad>"),
        )
        tokenizer.chat_template = tokenizer_config.get("chat_template")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------

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
        "val_file",
        "full_val_size",
        "val_sample_size",
        "val_sample_fraction_actual",
        "val_sample_seed",
        "val_sample_indices_preview",
        "max_new_tokens",
        "inference_batch_size",
        "temperature",
        "top_p",
        "unknown_count",
        "prediction_output_path",
        "load_project_name",
        "load_codecarbon_csv_path",
        "load_wall_duration_sec",
        "load_codecarbon_duration_sec",
        "load_energy_consumed_kwh",
        "load_emissions_kg_co2eq",
        "load_gpu_energy_kwh",
        "inference_project_name",
        "inference_codecarbon_csv_path",
        "inference_wall_duration_sec",
        "inference_codecarbon_duration_sec",
        "inference_energy_consumed_kwh",
        "inference_emissions_kg_co2eq",
        "inference_gpu_energy_kwh",
        "inference_energy_per_example_kwh",
        "inference_emissions_per_example_kg",
        "measured_total_energy_kwh",
        "measured_total_emissions_kg_co2eq",
        "estimated_full_energy_kwh_by_examples",
        "estimated_full_emissions_kg_by_examples",
        "estimation_formula",
        "gpu_model",
        "cpu_model",
        "ram_total_size",
        "codecarbon_version",
        "notes",
    ]


def upsert_summary_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    existing_fieldnames: list[str] = []

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    key = row["original_run_id"]
    existing_rows = [existing for existing in existing_rows if existing.get("original_run_id") != key]
    existing_rows.append(row)

    fieldnames: list[str] = []

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


def build_project_base(run_id: str, task: str) -> str:
    return f"calib_run{run_id}_mistral7b_zeroshot_{task.lower()}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeCarbon calibration for Mistral-7B Zero-shot MI/MT."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=["011", "015"],
        help="011 = Mistral Zero-shot MI, 015 = Mistral Zero-shot MT.",
    )

    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    if args.sample_fraction <= 0 or args.sample_fraction > 1:
        raise ValueError("--sample-fraction must be in (0, 1].")
    if args.min_samples <= 0 or args.max_samples < args.min_samples:
        raise ValueError("Invalid sample bounds.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    ensure_dirs()

    run = next(item for item in RUNS if item["run_id"] == args.run_id)

    run_id = run["run_id"]
    task = run["task"]
    project_base = build_project_base(run_id, task)

    val_path = VAL_DIR / run["val_file"]
    prediction_output_path = OUTPUTS_DIR / f"{project_base}_sample_predictions.csv"

    load_project_name = f"{project_base}_load"
    inference_project_name = f"{project_base}_inference"

    print("=" * 72)
    print(f"Carbon calibration run: {project_base}")
    print(f"Original run id: {run_id}")
    print(f"Task: {task}")
    print(f"Model: {run['model']}")
    print(f"Method: {run['method']}")
    print(f"Val file: {val_path}")
    print(f"Prediction output: {prediction_output_path}")
    print(f"Emissions dir: {EMISSIONS_DIR}")
    print("=" * 72)

    if not val_path.is_file():
        raise FileNotFoundError(f"Missing val file: {val_path}")
    if not PROMPTS_JSON.is_file():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_JSON}")

    prompts = load_json(PROMPTS_JSON)
    prompt = select_prompt(prompts, task)

    full_val_records = load_jsonl_records(val_path)
    full_val_size = len(full_val_records)

    val_indices = choose_sample_indices(
        full_size=full_val_size,
        sample_fraction=args.sample_fraction,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        seed=args.sample_seed + int(run_id),
    )

    val_records = [full_val_records[index] for index in val_indices]
    val_sample_size = len(val_records)
    val_sample_fraction_actual = val_sample_size / full_val_size

    print(f"Full validation size: {full_val_size}")
    print(f"Validation sample size: {val_sample_size}")
    print(f"Validation sample fraction: {val_sample_fraction_actual:.6f}")
    print(f"Validation sample index preview: {val_indices[:10]}")

    # ------------------------------------------------------------------
    # 1. Model load, tracked separately.
    # ------------------------------------------------------------------

    load_start = time.time()
    with track_emissions(load_project_name, output_dir=EMISSIONS_DIR):
        model, tokenizer = load_model_and_tokenizer()
    load_wall_duration_sec = time.time() - load_start

    # ------------------------------------------------------------------
    # 2. Inference, tracked separately.
    # ------------------------------------------------------------------

    output_columns = get_output_columns(task)
    all_rows: List[dict] = []
    unknown_count = 0

    inference_start = time.time()
    with track_emissions(inference_project_name, output_dir=EMISSIONS_DIR):
        for batch_start, batch_records, batch_source_indices in iter_batches_with_indices(
            val_records,
            val_indices,
            args.batch_size,
        ):
            prompt_texts = render_prompt_texts(
                tokenizer=tokenizer,
                records=batch_records,
                prompt=prompt,
                source_path=val_path,
                source_indices=batch_source_indices,
            )

            raw_outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for offset, raw_output in enumerate(raw_outputs):
                source_row_index = batch_source_indices[offset]

                if task == "MT":
                    parsed = parse_multitask_output(raw_output)

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
                        unknown_count += 1

                    all_rows.append(
                        {
                            "row_index": source_row_index,
                            "task": task,
                            "pred_label": parsed,
                            "raw_output": raw_output,
                        }
                    )

            if len(all_rows) % 25 == 0 or len(all_rows) == val_sample_size:
                elapsed = time.time() - inference_start
                print(
                    f"Progress: {len(all_rows)}/{val_sample_size} | "
                    f"Unknowns: {unknown_count} | "
                    f"Inference time: {elapsed / 60:.2f} min",
                    flush=True,
                )

    inference_wall_duration_sec = time.time() - inference_start

    write_predictions_csv(prediction_output_path, output_columns, all_rows)

    # ------------------------------------------------------------------
    # CodeCarbon metrics + careful full-run estimate.
    # ------------------------------------------------------------------

    load_metrics = get_codecarbon_metrics(load_project_name)
    inference_metrics = get_codecarbon_metrics(inference_project_name)

    load_energy = float(load_metrics["energy_consumed_kwh"])
    load_emissions = float(load_metrics["emissions_kg_co2eq"])
    load_gpu_energy = float(load_metrics["gpu_energy_kwh"])

    inference_energy = float(inference_metrics["energy_consumed_kwh"])
    inference_emissions = float(inference_metrics["emissions_kg_co2eq"])
    inference_gpu_energy = float(inference_metrics["gpu_energy_kwh"])

    inference_energy_per_example = inference_energy / val_sample_size
    inference_emissions_per_example = inference_emissions / val_sample_size

    measured_total_energy = load_energy + inference_energy
    measured_total_emissions = load_emissions + inference_emissions

    estimated_full_energy = load_energy + (inference_energy_per_example * full_val_size)
    estimated_full_emissions = load_emissions + (inference_emissions_per_example * full_val_size)

    summary_row = {
        "timestamp_utc": utc_now(),
        "status": "success",
        "original_run_id": run_id,
        "calibration_project_base": project_base,
        "model": run["model"],
        "method": run["method"],
        "task": task,
        "task_group": run["task_group"],
        "think": run["think"],
        "aug": run["aug"],
        "val_file": run["val_file"],
        "full_val_size": full_val_size,
        "val_sample_size": val_sample_size,
        "val_sample_fraction_actual": f"{val_sample_fraction_actual:.8f}",
        "val_sample_seed": args.sample_seed,
        "val_sample_indices_preview": json.dumps(val_indices[:25]),
        "max_new_tokens": args.max_new_tokens,
        "inference_batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "unknown_count": unknown_count,
        "prediction_output_path": str(prediction_output_path),

        "load_project_name": load_project_name,
        "load_codecarbon_csv_path": load_metrics["codecarbon_csv_path"],
        "load_wall_duration_sec": f"{load_wall_duration_sec:.6f}",
        "load_codecarbon_duration_sec": f"{float(load_metrics['codecarbon_duration_sec']):.6f}",
        "load_energy_consumed_kwh": f"{load_energy:.12f}",
        "load_emissions_kg_co2eq": f"{load_emissions:.12f}",
        "load_gpu_energy_kwh": f"{load_gpu_energy:.12f}",

        "inference_project_name": inference_project_name,
        "inference_codecarbon_csv_path": inference_metrics["codecarbon_csv_path"],
        "inference_wall_duration_sec": f"{inference_wall_duration_sec:.6f}",
        "inference_codecarbon_duration_sec": f"{float(inference_metrics['codecarbon_duration_sec']):.6f}",
        "inference_energy_consumed_kwh": f"{inference_energy:.12f}",
        "inference_emissions_kg_co2eq": f"{inference_emissions:.12f}",
        "inference_gpu_energy_kwh": f"{inference_gpu_energy:.12f}",
        "inference_energy_per_example_kwh": f"{inference_energy_per_example:.12f}",
        "inference_emissions_per_example_kg": f"{inference_emissions_per_example:.12f}",

        "measured_total_energy_kwh": f"{measured_total_energy:.12f}",
        "measured_total_emissions_kg_co2eq": f"{measured_total_emissions:.12f}",
        "estimated_full_energy_kwh_by_examples": f"{estimated_full_energy:.12f}",
        "estimated_full_emissions_kg_by_examples": f"{estimated_full_emissions:.12f}",
        "estimation_formula": "load_once + inference_per_example * full_val_size",

        "gpu_model": inference_metrics["gpu_model"] or load_metrics["gpu_model"],
        "cpu_model": inference_metrics["cpu_model"] or load_metrics["cpu_model"],
        "ram_total_size": inference_metrics["ram_total_size"] or load_metrics["ram_total_size"],
        "codecarbon_version": inference_metrics["codecarbon_version"] or load_metrics["codecarbon_version"],
        "notes": "Retrospective zero-shot calibration. Load and inference tracked separately; model settings preserved from original Mistral zero-shot script.",
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    print()
    print("=" * 72)
    print("Mistral zero-shot calibration complete")
    print("=" * 72)
    print(f"Predictions saved: {prediction_output_path}")
    print(f"Load emissions CSV: {load_metrics['codecarbon_csv_path']}")
    print(f"Inference emissions CSV: {inference_metrics['codecarbon_csv_path']}")
    print(f"Summary CSV updated: {SUMMARY_CSV}")
    print()
    print(f"Measured load emissions: {load_emissions:.12f} kg CO2eq")
    print(f"Measured inference emissions: {inference_emissions:.12f} kg CO2eq")
    print(f"Measured total calibration emissions: {measured_total_emissions:.12f} kg CO2eq")
    print(f"Estimated full-run emissions: {estimated_full_emissions:.12f} kg CO2eq")
    print()
    print("Formula used:")
    print("  estimated_full = load_once + inference_per_example * full_val_size")
    print("=" * 72)


if __name__ == "__main__":
    main()