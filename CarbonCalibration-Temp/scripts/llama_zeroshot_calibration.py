#!/usr/bin/env python3

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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------
# Repo / path setup
# ---------------------------------------------------------------------

REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

MODEL_PATH = "/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct"
PROMPTS_PATH = REPO_ROOT / "prompts.json"
VAL_DIR = REPO_ROOT / "data" / "val"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


# ---------------------------------------------------------------------
# Calibration run definitions
# ---------------------------------------------------------------------

RUNS = [
    {
        "run_id": "001",
        "task": "MI",
        "task_group": "SingleTask",
        "val_file": "mistake_identification_val.jsonl",
        "model": "LLaMA-3.1-8B",
        "method": "Zero-shot",
        "think": "N/A",
    },
    {
        "run_id": "005",
        "task": "MT",
        "task_group": "MT",
        "val_file": "multitask_val.jsonl",
        "model": "LLaMA-3.1-8B",
        "method": "Zero-shot",
        "think": "N/A",
    },
]

TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
}

UNKNOWN_LABEL = "Unknown"
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------
# Label parsing helpers copied from original logic
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_text(text: str) -> str:
    text = text.strip().strip("\"'`")
    text = text.strip(" .,!?:;")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


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


def recover_single_label(raw_output: str) -> str:
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


def parse_multitask_output(raw_output: str) -> dict[str, str]:
    parsed = {
        "pred_mi": UNKNOWN_LABEL,
        "pred_ml": UNKNOWN_LABEL,
        "pred_pg": UNKNOWN_LABEL,
        "pred_act": UNKNOWN_LABEL,
    }

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

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        field_name, raw_value = line.split(":", 1)
        field_key = canonicalize_text(field_name).replace(" ", "")
        out_col = field_map.get(field_key)

        if out_col and parsed[out_col] == UNKNOWN_LABEL:
            parsed[out_col] = normalize_label(raw_value)

    regex_patterns = {
        "pred_mi": r"(mistake[\s_-]*identification|mi)\s*:\s*(yes|no|to some extent)",
        "pred_ml": r"(mistake[\s_-]*location|ml)\s*:\s*(yes|no|to some extent)",
        "pred_pg": r"(providing[\s_-]*guidance|pg)\s*:\s*(yes|no|to some extent)",
        "pred_act": r"(actionability|act)\s*:\s*(yes|no|to some extent)",
    }

    for out_col, pattern in regex_patterns.items():
        if parsed[out_col] != UNKNOWN_LABEL:
            continue
        match = re.search(pattern, raw_output, flags=re.IGNORECASE)
        if match:
            parsed[out_col] = normalize_label(match.group(2))

    return parsed


# ---------------------------------------------------------------------
# Data/model helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_user(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found in example.")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    return model, tokenizer


@torch.inference_mode()
def run_inference(model, tokenizer, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    outputs = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    indices = sorted(rng.sample(range(full_size), sample_size))
    return indices


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


def write_predictions(out_path: Path, preds: list[dict]) -> None:
    if not preds:
        raise ValueError("No predictions were produced.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)


def upsert_summary_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "status",
        "original_run_id",
        "calibration_project_base",
        "model",
        "method",
        "task",
        "task_group",
        "think",
        "val_file",
        "full_val_size",
        "sample_size",
        "sample_fraction_actual",
        "sample_seed",
        "sample_indices_preview",
        "max_new_tokens",
        "unknown_count",
        "retry_count",
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

    rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    key = row["original_run_id"]
    rows = [existing for existing in rows if existing.get("original_run_id") != key]
    rows.append(row)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_project_base(run_id: str, task: str) -> str:
    task_slug = task.lower()
    return f"calib_run{run_id}_llama31_8b_zeroshot_{task_slug}"


# ---------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeCarbon calibration script for LLaMA-3.1-8B zero-shot MI/MT."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        choices=["001", "005"],
        help="001 = MI calibration, 005 = MT calibration.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction of validation examples to run for calibration.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum number of validation examples to sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of validation examples to sample.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic representative sampling.",
    )

    args = parser.parse_args()

    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be in the range (0, 1].")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive.")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.max_samples < args.min_samples:
        raise ValueError("--max-samples must be greater than or equal to --min-samples.")

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    run = next(item for item in RUNS if item["run_id"] == args.run_id)

    run_id = run["run_id"]
    task = run["task"]
    val_file = run["val_file"]
    project_base = build_project_base(run_id, task)

    load_project_name = f"{project_base}_load"
    inference_project_name = f"{project_base}_inference"

    prediction_output_path = OUTPUTS_DIR / f"{project_base}_sample_predictions.csv"

    print("=" * 72)
    print(f"Carbon calibration run: {project_base}")
    print(f"Original run id: {run_id}")
    print(f"Task: {task}")
    print(f"Model: {run['model']}")
    print(f"Method: {run['method']}")
    print(f"Temp emissions dir: {EMISSIONS_DIR}")
    print(f"Temp output path: {prediction_output_path}")
    print("=" * 72)

    with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    single_prompts = prompts["single_task_zero_shot"]["prompts"]
    multi_prompt = prompts["multitask_training"]["prompt"]
    retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multi = prompts["retry_prompts"]["prompts"]["multitask"]

    data_path = VAL_DIR / val_file
    data = load_jsonl(data_path)
    full_val_size = len(data)

    sample_indices = choose_sample_indices(
        full_size=full_val_size,
        sample_fraction=args.sample_fraction,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        seed=args.sample_seed + int(run_id),
    )

    sampled_data = [data[index] for index in sample_indices]
    sample_size = len(sampled_data)
    sample_fraction_actual = sample_size / full_val_size

    print(f"Full validation size: {full_val_size}")
    print(f"Sample size: {sample_size}")
    print(f"Actual sample fraction: {sample_fraction_actual:.6f}")
    print(f"Sample index preview: {sample_indices[:10]}")

    # -------------------------------------------------------------
    # Track model loading separately.
    # This avoids wrongly scaling one-time loading cost by examples.
    # -------------------------------------------------------------

    load_start = time.time()
    with track_emissions(load_project_name, output_dir=EMISSIONS_DIR):
        model, tokenizer = load_model()
    load_wall_duration_sec = time.time() - load_start

    print(f"Model load wall time: {load_wall_duration_sec:.2f} sec")

    # -------------------------------------------------------------
    # Track inference separately.
    # This part is scaled by number of examples.
    # -------------------------------------------------------------

    preds = []
    unknown_count = 0
    retry_count = 0

    inference_start = time.time()
    with track_emissions(inference_project_name, output_dir=EMISSIONS_DIR):
        for local_i, ex in enumerate(sampled_data, start=1):
            user = extract_user(ex)

            if task == "MT":
                raw = run_inference(model, tokenizer, multi_prompt, user)
                parsed = parse_multitask_output(raw)

                if UNKNOWN_LABEL in parsed.values():
                    retry_count += 1
                    raw2 = run_inference(model, tokenizer, retry_multi, user)
                    parsed2 = parse_multitask_output(raw2)

                    for key in parsed:
                        if parsed[key] == UNKNOWN_LABEL:
                            parsed[key] = parsed2[key]

                    raw = raw2

                if UNKNOWN_LABEL in parsed.values():
                    unknown_count += 1

                preds.append(
                    {
                        "sample_index": sample_indices[local_i - 1],
                        "pred_mi": parsed["pred_mi"],
                        "pred_ml": parsed["pred_ml"],
                        "pred_pg": parsed["pred_pg"],
                        "pred_act": parsed["pred_act"],
                        "raw_output": raw,
                    }
                )

            else:
                prompt = single_prompts[TASK_TO_PROMPT_KEY[task]]
                raw = run_inference(model, tokenizer, prompt, user)
                label = recover_single_label(raw)

                if label == UNKNOWN_LABEL:
                    retry_count += 1
                    raw2 = run_inference(model, tokenizer, retry_single, user)
                    label = recover_single_label(raw2)
                    raw = raw2

                if label == UNKNOWN_LABEL:
                    unknown_count += 1

                preds.append(
                    {
                        "sample_index": sample_indices[local_i - 1],
                        "pred_label": label,
                        "raw_output": raw,
                    }
                )

            if local_i == 1 or local_i % 10 == 0 or local_i == sample_size:
                elapsed = time.time() - inference_start
                print(
                    f"Progress: {local_i}/{sample_size} | "
                    f"Unknowns: {unknown_count} | Retries: {retry_count} | "
                    f"Inference time: {elapsed / 60:.2f} min",
                    flush=True,
                )

    inference_wall_duration_sec = time.time() - inference_start

    write_predictions(prediction_output_path, preds)

    # -------------------------------------------------------------
    # Read CodeCarbon CSVs and calculate careful extrapolation.
    # Formula:
    # estimated full = one-time load + inference_per_example * full_val_size
    # -------------------------------------------------------------

    load_metrics = get_codecarbon_metrics(load_project_name)
    inference_metrics = get_codecarbon_metrics(inference_project_name)

    load_energy = float(load_metrics["energy_consumed_kwh"])
    load_emissions = float(load_metrics["emissions_kg_co2eq"])
    load_gpu_energy = float(load_metrics["gpu_energy_kwh"])

    inference_energy = float(inference_metrics["energy_consumed_kwh"])
    inference_emissions = float(inference_metrics["emissions_kg_co2eq"])
    inference_gpu_energy = float(inference_metrics["gpu_energy_kwh"])

    inference_energy_per_example = inference_energy / sample_size
    inference_emissions_per_example = inference_emissions / sample_size

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
        "val_file": val_file,
        "full_val_size": full_val_size,
        "sample_size": sample_size,
        "sample_fraction_actual": f"{sample_fraction_actual:.8f}",
        "sample_seed": args.sample_seed,
        "sample_indices_preview": json.dumps(sample_indices[:25]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "unknown_count": unknown_count,
        "retry_count": retry_count,
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
        "notes": "Retrospective calibration. Load and inference tracked separately to avoid scaling one-time model loading cost.",
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    print()
    print("=" * 72)
    print("Calibration complete")
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