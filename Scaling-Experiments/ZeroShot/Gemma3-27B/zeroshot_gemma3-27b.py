#!/usr/bin/env python3
"""
zeroshot_gemma27b.py
Zero-shot inference for Gemma3-27B — one run at a time.
Scaling Experiments — Runs 041-045.

Usage:
  python zeroshot_gemma27b.py --run-id 041   # MI
  python zeroshot_gemma27b.py --run-id 042   # ML
  python zeroshot_gemma27b.py --run-id 043   # PG
  python zeroshot_gemma27b.py --run-id 044   # Act
  python zeroshot_gemma27b.py --run-id 045   # MT

Gemma3-27B (~14GB 8-bit) — uses 1 L40S GPU via vLLM bitsandbytes quantization.
Fix: Switched to 8-bit single-GPU to bypass multi-GPU network/NCCL deadlocks.
"""

import os

# 1. STOP HUGGING FACE FROM WAKING UP TENSORFLOW
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Avoid tokenizer deadlock warnings when vLLM forks worker processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import re
import csv
import time
import threading
from pathlib import Path
import subprocess

# 2. IMPORT VLLM BEFORE TRANSFORMERS
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/WAVE/datasets/oignat_lab/Gemma3-27b"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
VAL_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
OUT_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling-Experiments/ZeroShot/Gemma3-27B")

# ── Run config ────────────────────────────────────────────────────────────────
RUNS = {
    "041": {"run_id": "041", "task": "MI",  "val_file": "mistake_identification_val.jsonl"},
    "042": {"run_id": "042", "task": "ML",  "val_file": "mistake_location_val.jsonl"},
    "043": {"run_id": "043", "task": "PG",  "val_file": "providing_guidance_val.jsonl"},
    "044": {"run_id": "044", "task": "Act", "val_file": "actionability_val.jsonl"},
    "045": {"run_id": "045", "task": "MT",  "val_file": "multitask_val.jsonl"},
}

# ── Resume config ─────────────────────────────────────────────────────────────
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


# ── Loading Ticker ────────────────────────────────────────────────────────────
def loading_ticker(stop_event):
    """Background thread to print updates and live VRAM usage while vLLM loads."""
    start_time = time.time()
    while not stop_event.is_set():
        time.sleep(30) # Prints an update every 30 seconds
        if stop_event.is_set():
            break
        elapsed = int(time.time() - start_time)
        
        # Ask the system exactly how much VRAM is currently filled
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip().split('\n')
            
            # Format the output nicely (focusing on GPU 0 since we moved to 1 GPU)
            vram_info = f"GPU 0: {smi_output[0]} MB"
        except Exception:
            vram_info = "VRAM query failed"

        print(f"\n[Status Update] 8-bit model loading... {elapsed}s elapsed. Current VRAM: [{vram_info}]")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"\nLoading Gemma3-27B from {MODEL_PATH} via vLLM...")
    print(f"Model size: ~28GB (8-bit bitsandbytes) — using 1 GPU")
    start = time.time()

    # Start the background status ticker
    stop_event = threading.Event()
    ticker_thread = threading.Thread(target=loading_ticker, args=(stop_event,))
    ticker_thread.start()

    try:
        model = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,           # Still fits on 1 of your 48GB GPUs!
            quantization="fp8",               # Turns on 8-bit compression
            enforce_eager=True,
            disable_log_stats=True,
            max_model_len=16384,
        )

        # We keep AutoProcessor just to apply the exact same chat template
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
    finally:
        # Guarantee the ticker stops printing whether it succeeds or crashes
        stop_event.set()
        ticker_thread.join()

    elapsed = time.time() - start
    print(f"\n--- SUCCESS: Model loaded in {elapsed:.1f}s ---")

    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_content}]},
    ]

    # Convert the messages list into a single raw prompt string using the official template
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False, # We want text, vLLM handles its own internal tokenization
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0, # Equivalent to do_sample=False
    )

    # Generate with vLLM (use_tqdm=False prevents progress bars for single inferences)
    outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)
    
    return outputs[0].outputs[0].text.strip()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_val_examples(val_path: Path) -> list:
    examples = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_user_content(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found in example")


def get_output_path(run_id: str, task: str) -> Path:
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt_predictions.csv"
    return OUT_DIR / f"run{run_id}_{task.lower()}_predictions.csv"


# ── Single-task run ───────────────────────────────────────────────────────────
def run_single_task(model, processor, run_config: dict,
                    system_prompt: str, retry_prompt: str) -> None:
    run_id   = run_config["run_id"]
    task     = run_config["task"]
    val_path = VAL_DIR / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | Zero-shot | Gemma3-27B (8-bit)")
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

        predictions.append({"pred_label": pred_label, "raw_output": raw_output})

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(examples)} | Unknowns so far: {unknowns}")

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
    run_id   = run_config["run_id"]
    task     = run_config["task"]
    val_path = VAL_DIR / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | Zero-shot | Gemma3-27B (8-bit)")
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

    out_path = get_output_path(run_id, task)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved → {out_path.name}")
    print(f"Rows with Unknown: {unknowns}/{len(predictions)} ({unknowns/len(predictions)*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Zero-shot inference for Gemma3-27B")
    parser.add_argument(
        "--run-id", required=True,
        choices=["041", "042", "043", "044", "045"],
        help="Run ID: 041=MI, 042=ML, 043=PG, 044=Act, 045=MT"
    )
    args = parser.parse_args()

    run_config = RUNS[args.run_id]
    task       = run_config["task"]

    print("TutorMind — Zero-shot Inference")
    print("Model: Gemma3-27B (8-bit Quantized)")
    print(f"Run: {args.run_id} | Task: {task}")
    print(f"Output dir: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SKIP_COMPLETED and get_output_path(args.run_id, task).exists():
        print(f"Run {args.run_id} already completed — CSV exists ✅")
        return

    print(f"\nLoading prompts from {PROMPTS_PATH} ...")
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    single_task_prompts = prompts["single_task_zero_shot"]["prompts"]
    multitask_prompt    = prompts["multitask_training"]["prompt"]
    retry_single        = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask     = prompts["retry_prompts"]["prompts"]["multitask"]

    model, processor = load_model()

    start = time.time()

    if task == "MT":
        run_multitask(model, processor, run_config,
                      system_prompt=multitask_prompt,
                      retry_prompt=retry_multitask)
    else:
        prompt_key    = TASK_TO_PROMPT_KEY[task]
        system_prompt = single_task_prompts[prompt_key]
        run_single_task(model, processor, run_config,
                        system_prompt=system_prompt,
                        retry_prompt=retry_single)

    elapsed = time.time() - start
    print(f"\nRun {args.run_id} completed in {elapsed/60:.1f} min")

    fname    = get_output_path(args.run_id, task).name
    val_flag = {"MI": "val-mi", "ML": "val-ml", "PG": "val-pg", "Act": "val-act", "MT": "val-mt"}[task]
    val_file = run_config["val_file"]
    print(f"\nNext — run eval:")
    print(
        f"  python tryeval.py "
        f"--predictions {OUT_DIR}/{fname} "
        f"--task {task} --run-id {args.run_id} "
        f"--model Gemma3-27B --method Zero-shot --aug None --think N/A "
        f"--{val_flag} {VAL_DIR}/{val_file} "
        f"--out {OUT_DIR}/run{args.run_id}_metrics.csv"
    )

    # Clean shutdown
    del model
    import gc
    gc.collect()

if __name__ == "__main__":
    main()