#!/usr/bin/env python3
"""CodeCarbon calibration for Gemma3-27B Zero-shot MI and MT.

Supported runs:
- 041 = Gemma3-27B Zero-shot MI
- 045 = Gemma3-27B Zero-shot MT

Important:
- Uses the same vLLM quantized setup as the original Gemma3-27B zero-shot script.
- Does NOT touch master_metrics.csv.
- Does NOT write to original experiment folders.
- All outputs go under CarbonCalibration-Temp.
"""

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import gc
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


REPO_ROOT = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind")
TEMP_ROOT = REPO_ROOT / "CarbonCalibration-Temp"

EMISSIONS_DIR = TEMP_ROOT / "emissions"
OUTPUTS_DIR = TEMP_ROOT / "outputs"
LOGS_DIR = TEMP_ROOT / "logs"
SUMMARY_CSV = TEMP_ROOT / "carbon_calibration_summary.csv"

MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3-27b"
PROMPTS_PATH = REPO_ROOT / "prompts.json"
VAL_DIR = REPO_ROOT / "data" / "val"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Gemma3-27B"
METHOD = "Zero-shot"
AUG = "None"
THINK = "N/A"
UNKNOWN_LABEL = "Unknown"
MAX_NEW_TOKENS = 64

RUNS = {
    "041": {
        "run_id": "041",
        "task": "MI",
        "task_group": "SingleTask",
        "val_file": "mistake_identification_val.jsonl",
    },
    "045": {
        "run_id": "045",
        "task": "MT",
        "task_group": "MT",
        "val_file": "multitask_val.jsonl",
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
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


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
    raise ValueError("No user message found in example.")


def loading_ticker(stop_event: threading.Event) -> None:
    start_time = time.time()

    while not stop_event.is_set():
        time.sleep(30)

        if stop_event.is_set():
            break

        elapsed = int(time.time() - start_time)

        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            ).strip().split("\n")

            vram_info = f"GPU 0: {smi_output[0]} MB"
        except Exception:
            vram_info = "VRAM query failed"

        print(
            f"\n[Status Update] 8-bit/vLLM model loading... "
            f"{elapsed}s elapsed. Current VRAM: [{vram_info}]",
            flush=True,
        )


def load_model():
    print(f"\nLoading {MODEL_NAME} from {MODEL_PATH} via vLLM...")
    print("Model mode: vLLM single-GPU quantized setup")
    print("Settings: tensor_parallel_size=1, quantization='fp8', enforce_eager=True, max_model_len=16384")

    stop_event = threading.Event()
    ticker_thread = threading.Thread(target=loading_ticker, args=(stop_event,))
    ticker_thread.start()

    try:
        model = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            quantization="fp8",
            enforce_eager=True,
            disable_log_stats=True,
            max_model_len=16384,
        )

        processor = AutoProcessor.from_pretrained(MODEL_PATH)

    finally:
        stop_event.set()
        ticker_thread.join()

    print("\n--- SUCCESS: Model loaded ---")
    return model, processor


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

    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )

    outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


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
        "val_file",
        "full_val_size",
        "val_sample_size",
        "val_sample_fraction_actual",
        "val_sample_seed",
        "val_sample_indices_preview",
        "max_new_tokens",
        "vllm_tensor_parallel_size",
        "vllm_quantization",
        "vllm_enforce_eager",
        "vllm_max_model_len",
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


def build_project_base(run_id: str, task: str) -> str:
    return f"calib_run{run_id}_gemma3_27b_zeroshot_vllm_fp8_{task.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeCarbon calibration for Gemma3-27B Zero-shot MI/MT."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=["041", "045"],
        help="041 = Gemma3-27B Zero-shot MI, 045 = Gemma3-27B Zero-shot MT.",
    )

    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)

    args = parser.parse_args()

    if args.sample_fraction <= 0 or args.sample_fraction > 1:
        raise ValueError("--sample-fraction must be in (0, 1].")
    if args.min_samples <= 0 or args.max_samples < args.min_samples:
        raise ValueError("Invalid sample bounds.")

    ensure_dirs()

    run = RUNS[args.run_id]
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
    print(f"Model: {MODEL_NAME}")
    print(f"Method: {METHOD}")
    print("Mode: vLLM single-GPU quantized setup")
    print("vLLM settings: tensor_parallel_size=1, quantization='fp8', enforce_eager=True, max_model_len=16384")
    print(f"Val file: {val_path}")
    print(f"Prediction output: {prediction_output_path}")
    print(f"Emissions dir: {EMISSIONS_DIR}")
    print("=" * 72)

    if not val_path.is_file():
        raise FileNotFoundError(f"Missing val file: {val_path}")
    if not PROMPTS_PATH.is_file():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_PATH}")

    with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    single_task_prompts = prompts["single_task_zero_shot"]["prompts"]
    multitask_prompt = prompts["multitask_training"]["prompt"]
    retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask = prompts["retry_prompts"]["prompts"]["multitask"]

    examples = load_jsonl(val_path)
    full_val_size = len(examples)

    sample_indices = choose_sample_indices(
        full_size=full_val_size,
        sample_fraction=args.sample_fraction,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        seed=args.sample_seed + int(run_id),
    )

    sampled_examples = [examples[index] for index in sample_indices]
    sample_size = len(sampled_examples)
    sample_fraction_actual = sample_size / full_val_size

    print(f"Full validation size: {full_val_size}")
    print(f"Sample size: {sample_size}")
    print(f"Actual sample fraction: {sample_fraction_actual:.6f}")
    print(f"Sample index preview: {sample_indices[:10]}")

    load_start = time.time()
    with track_emissions(load_project_name, output_dir=EMISSIONS_DIR):
        model, processor = load_model()
    load_wall_duration_sec = time.time() - load_start

    if task == "MT":
        system_prompt = multitask_prompt
        retry_prompt = retry_multitask
        output_columns = ["row_index", "task", "pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"]
    else:
        system_prompt = single_task_prompts[TASK_TO_PROMPT_KEY[task]]
        retry_prompt = retry_single
        output_columns = ["row_index", "task", "pred_label", "raw_output"]

    predictions: List[dict] = []
    unknown_count = 0
    retry_count = 0

    inference_start = time.time()
    with track_emissions(inference_project_name, output_dir=EMISSIONS_DIR):
        for local_i, example in enumerate(sampled_examples, start=1):
            source_index = sample_indices[local_i - 1]
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

                predictions.append(
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

                predictions.append(
                    {
                        "row_index": source_index,
                        "task": task,
                        "pred_label": pred_label,
                        "raw_output": raw_output,
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

    write_predictions_csv(prediction_output_path, output_columns, predictions)

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
        "model": MODEL_NAME,
        "method": METHOD,
        "task": task,
        "task_group": run["task_group"],
        "think": THINK,
        "aug": AUG,
        "val_file": run["val_file"],
        "full_val_size": full_val_size,
        "val_sample_size": sample_size,
        "val_sample_fraction_actual": f"{sample_fraction_actual:.8f}",
        "val_sample_seed": args.sample_seed,
        "val_sample_indices_preview": json.dumps(sample_indices[:25]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "vllm_tensor_parallel_size": 1,
        "vllm_quantization": "fp8",
        "vllm_enforce_eager": True,
        "vllm_max_model_len": 16384,
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
        "notes": (
            "Retrospective Gemma3-27B zero-shot calibration. "
            "Uses original vLLM setup: tensor_parallel_size=1, quantization=fp8, "
            "enforce_eager=True, max_model_len=16384, AutoProcessor chat template, "
            "SamplingParams max_tokens=64 temperature=0.0. Load and inference tracked separately."
        ),
    }

    upsert_summary_row(SUMMARY_CSV, summary_row)

    del model
    gc.collect()

    print()
    print("=" * 72)
    print("Gemma3-27B zero-shot calibration complete")
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