#!/usr/bin/env python3
"""
zeroshot_gemma12b.py
Zero-shot inference for Gemma3-12B on all 5 tasks.
Scaling Experiments — Runs 031-035.

Loads the model ONCE and runs inference on all 5 val files sequentially.
Produces 5 prediction CSVs ready for evaluate_run.py.

Uses Gemma3ForConditionalGeneration + AutoProcessor (official Google approach).
Text-only messages (no images).

Changes:
  - attn_implementation="eager" to avoid SDPA alignment error
  - torch.cuda.empty_cache() between runs to clear KV cache
  - RESUME_FROM config to skip already completed runs

Runs:
  031 | Gemma3-12B | Zero-shot | MI  | Aug: None | Think: N/A
  032 | Gemma3-12B | Zero-shot | ML  | Aug: None | Think: N/A
  033 | Gemma3-12B | Zero-shot | PG  | Aug: None | Think: N/A
  034 | Gemma3-12B | Zero-shot | Act | Aug: None | Think: N/A
  035 | Gemma3-12B | Zero-shot | MT  | Aug: None | Think: N/A
"""

import json
import re
import csv
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/WAVE/datasets/oignat_lab/Gemma3"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
VAL_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
OUT_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling-Experiments/ZeroShot/Gemma3-12B")

# ── Run config ────────────────────────────────────────────────────────────────
RUNS = [
    {"run_id": "031", "task": "MI",  "val_file": "mistake_identification_val.jsonl"},
    {"run_id": "032", "task": "ML",  "val_file": "mistake_location_val.jsonl"},
    {"run_id": "033", "task": "PG",  "val_file": "providing_guidance_val.jsonl"},
    {"run_id": "034", "task": "Act", "val_file": "actionability_val.jsonl"},
    {"run_id": "035", "task": "MT",  "val_file": "multitask_val.jsonl"},
]

# ── Resume config ─────────────────────────────────────────────────────────────
# Set to True to skip runs that already have a completed CSV file
SKIP_COMPLETED = True

# Task to prompt key mapping
TASK_TO_PROMPT_KEY = {
    "MI":  "Mistake_Identification",
    "ML":  "Mistake_Location",
    "PG":  "Providing_Guidance",
    "Act": "Actionability",
}

# ── Generation config ─────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 64

# ── Label normalization ───────────────────────────────────────────────────────
def normalize_label(text: str) -> str:
    """Extract and normalize label from model output."""
    if not text:
        return "Unknown"

    candidates = [text.strip()]
    for line in text.splitlines():
        line = line.strip()
        if line:
            candidates.append(line)
        if ":" in line:
            candidates.append(line.split(":", 1)[1].strip())

    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'`.,!?:;").lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned == "yes":
            return "Yes"
        if cleaned == "no":
            return "No"
        if cleaned in {"to some extent", "to some extend"}:
            return "To some extent"

    return "Unknown"


def parse_multitask_output(text: str) -> dict:
    """Parse multitask model output into 4 labels."""
    field_map = {
        "mistakeidentification": "pred_mi",
        "mistakelocation":       "pred_ml",
        "providingguidance":     "pred_pg",
        "actionability":         "pred_act",
    }

    result = {"pred_mi": "Unknown", "pred_ml": "Unknown",
              "pred_pg": "Unknown", "pred_act": "Unknown"}

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        field, value = line.split(":", 1)
        key = field.strip().lower().replace(" ", "").replace("_", "")
        col = field_map.get(key)
        if col:
            result[col] = normalize_label(value.strip())

    return result


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    """Load Gemma3-12B using official transformers approach."""
    print(f"\nLoading Gemma3-12B from {MODEL_PATH} ...")
    start = time.time()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",   # Avoids SDPA alignment error
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")

    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
    """Run one inference call and return raw model output text."""
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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = output_ids[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_val_examples(val_path: Path) -> list:
    """Load validation examples from JSONL file."""
    examples = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_user_content(example: dict) -> str:
    """Extract user message content from a chat-format example."""
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found in example")


def get_output_path(run_id: str, task: str) -> Path:
    """Return the output CSV path for a given run."""
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt_predictions.csv"
    return OUT_DIR / f"run{run_id}_{task.lower()}_predictions.csv"


# ── Single-task run ───────────────────────────────────────────────────────────
def run_single_task(model, processor, run_config: dict,
                    system_prompt: str, retry_prompt: str) -> None:
    """Run zero-shot inference for one single-task run and save predictions CSV."""
    run_id   = run_config["run_id"]
    task     = run_config["task"]
    val_path = VAL_DIR / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | Zero-shot | Gemma3-12B")
    print(f"Val file: {val_path.name} | Examples: ", end="")

    examples = load_val_examples(val_path)
    print(len(examples))

    predictions = []
    unknowns    = 0

    for i, example in enumerate(examples):
        user_content = extract_user_content(example)
        raw_output   = run_inference(model, processor, system_prompt, user_content)
        pred_label   = normalize_label(raw_output)

        if pred_label == "Unknown":
            print(f"  [Row {i}] Unknown — retrying...")
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            pred_label  = normalize_label(raw_output2)
            raw_output  = raw_output2
            if pred_label == "Unknown":
                unknowns += 1
                print(f"  [Row {i}] Still Unknown after retry")

        predictions.append({
            "pred_label": pred_label,
            "raw_output": raw_output,
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(examples)} | Unknowns so far: {unknowns}")

    # Clear GPU cache after run
    torch.cuda.empty_cache()

    # Save CSV
    out_path = get_output_path(run_id, task)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_label", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved → {out_path.name}")
    print(f"Unknowns: {unknowns}/{len(predictions)} ({unknowns/len(predictions)*100:.1f}%)")


# ── Multitask run ─────────────────────────────────────────────────────────────
def run_multitask(model, processor, run_config: dict,
                  system_prompt: str, retry_prompt: str) -> None:
    """Run zero-shot inference for the multitask run and save predictions CSV."""
    run_id   = run_config["run_id"]
    task     = run_config["task"]
    val_path = VAL_DIR / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | Zero-shot | Gemma3-12B")
    print(f"Val file: {val_path.name} | Examples: ", end="")

    examples = load_val_examples(val_path)
    print(len(examples))

    predictions = []
    unknowns    = 0

    for i, example in enumerate(examples):
        user_content = extract_user_content(example)
        raw_output   = run_inference(model, processor, system_prompt, user_content)
        parsed       = parse_multitask_output(raw_output)

        if "Unknown" in parsed.values():
            print(f"  [Row {i}] Unknown dimension(s) — retrying...")
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            parsed2     = parse_multitask_output(raw_output2)
            for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
                if parsed[col] == "Unknown":
                    parsed[col] = parsed2[col]
            raw_output = raw_output2

        if "Unknown" in parsed.values():
            unknowns += 1
            print(f"  [Row {i}] Still has Unknown after retry")

        predictions.append({
            "pred_mi":    parsed["pred_mi"],
            "pred_ml":    parsed["pred_ml"],
            "pred_pg":    parsed["pred_pg"],
            "pred_act":   parsed["pred_act"],
            "raw_output": raw_output,
        })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(examples)} | Rows with Unknown: {unknowns}")

    # Clear GPU cache after run
    torch.cuda.empty_cache()

    # Save CSV
    out_path = get_output_path(run_id, task)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved → {out_path.name}")
    print(f"Rows with Unknown: {unknowns}/{len(predictions)} ({unknowns/len(predictions)*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("TutorMind — Zero-shot Inference")
    print("Model: Gemma3-12B")
    print("Experiment: Scaling — Runs 031-035")
    print(f"Output dir: {OUT_DIR}")
    print(f"Skip completed runs: {SKIP_COMPLETED}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"\nLoading prompts from {PROMPTS_PATH} ...")
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    single_task_prompts = prompts["single_task_zero_shot"]["prompts"]
    multitask_prompt    = prompts["multitask_training"]["prompt"]
    retry_single        = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask     = prompts["retry_prompts"]["prompts"]["multitask"]

    # Load model ONCE
    model, processor = load_model()

    # Run all 5 tasks
    total_start = time.time()

    for run_config in RUNS:
        run_id = run_config["run_id"]
        task   = run_config["task"]

        # Skip if already completed
        if SKIP_COMPLETED and get_output_path(run_id, task).exists():
            print(f"\nSkipping Run {run_id} | Task: {task} — CSV already exists ✅")
            continue

        run_start = time.time()

        if task == "MT":
            run_multitask(
                model, processor, run_config,
                system_prompt=multitask_prompt,
                retry_prompt=retry_multitask,
            )
        else:
            prompt_key    = TASK_TO_PROMPT_KEY[task]
            system_prompt = single_task_prompts[prompt_key]
            run_single_task(
                model, processor, run_config,
                system_prompt=system_prompt,
                retry_prompt=retry_single,
            )

        run_elapsed = time.time() - run_start
        print(f"Run {run_id} done in {run_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All runs completed in {total_elapsed/60:.1f} min")
    print(f"\nNext — run evaluate_run.py for each run:")

    val_flag_map = {"MI": "val-mi", "ML": "val-ml", "PG": "val-pg", "Act": "val-act", "MT": "val-mt"}
    for run_config in RUNS:
        run_id   = run_config["run_id"]
        task     = run_config["task"]
        fname    = get_output_path(run_id, task).name
        val_flag = val_flag_map[task]
        val_file = run_config["val_file"]
        print(
            f"  python evaluate_run.py "
            f"--predictions {OUT_DIR}/{fname} "
            f"--task {task} --run-id {run_id} "
            f"--model Gemma3-12B --method Zero-shot --aug None --think N/A "
            f"--{val_flag} {VAL_DIR}/{val_file} "
            f"--out {OUT_DIR}/run{run_id}_metrics.csv"
        )


if __name__ == "__main__":
    main()