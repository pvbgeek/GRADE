#!/usr/bin/env python3
"""
lora_gemma12b_aug.py

Gemma3-12B LoRA fine-tuning + inference for augmented TutorMind runs.

Runs supported:
  076 | Gemma3-12B | LoRA | MI  | Aug: Qwen3-Gen        | Think: N/A
  077 | Gemma3-12B | LoRA | ML  | Aug: Qwen3-Gen        | Think: N/A
  078 | Gemma3-12B | LoRA | PG  | Aug: Qwen3-Gen        | Think: N/A
  079 | Gemma3-12B | LoRA | Act | Aug: Qwen3-Gen        | Think: N/A
  080 | Gemma3-12B | LoRA | MT  | Aug: Qwen3-Gen        | Think: N/A

  101 | Gemma3-12B | LoRA | MI  | Aug: Qwen3-Gen+Verify | Think: N/A
  102 | Gemma3-12B | LoRA | ML  | Aug: Qwen3-Gen+Verify | Think: N/A
  103 | Gemma3-12B | LoRA | PG  | Aug: Qwen3-Gen+Verify | Think: N/A
  104 | Gemma3-12B | LoRA | Act | Aug: Qwen3-Gen+Verify | Think: N/A
  105 | Gemma3-12B | LoRA | MT  | Aug: Qwen3-Gen+Verify | Think: N/A

Important:
- Training uses augmented train JSONL.
- Inference uses original validation JSONL.
- Prediction CSVs are saved in the current folder.
- Temporary trainer outputs are saved in ./tmp_gemma12b_aug/runXXX_task.
- No adapter is permanently saved; train -> infer -> cleanup.
- Full bfloat16, no quantization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import sys as _sys
from pathlib import Path as _Path
for _candidate in _Path(__file__).resolve().parents:
    if (_candidate / "utils" / "codecarbon_helper.py").is_file():
        if str(_candidate) not in _sys.path:
            _sys.path.insert(0, str(_candidate))
        break

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Gemma3-12B"
MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")

OUT_DIR = Path.cwd()
TMP_DIR = Path.cwd() / "tmp_gemma12b_aug"

UNKNOWN_LABEL = "Unknown"
SKIP_COMPLETED = True

TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}

RUNS: Dict[str, dict] = {
    "076": {
        "task": "MI",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl",
    },
    "077": {
        "task": "ML",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_location_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_location_val.jsonl",
    },
    "078": {
        "task": "PG",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/providing_guidance_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/providing_guidance_val.jsonl",
    },
    "079": {
        "task": "Act",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/actionability_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/actionability_val.jsonl",
    },
    "080": {
        "task": "MT",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl",
    },
    "101": {
        "task": "MI",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl",
    },
    "102": {
        "task": "ML",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_location_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_location_val.jsonl",
    },
    "103": {
        "task": "PG",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/providing_guidance_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/providing_guidance_val.jsonl",
    },
    "104": {
        "task": "Act",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/actionability_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/actionability_val.jsonl",
    },
    "105": {
        "task": "MT",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl",
    },
}

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=sorted(RUNS.keys()))
    parser.add_argument("--force", action="store_true")
    return parser


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


def load_jsonl(path: Path) -> List[dict]:
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

    return examples


def extract_user_content(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]

    raise ValueError("No user message found")


def format_chat_for_training(example: dict, processor) -> dict:
    messages = example["messages"]
    converted = []

    for msg in messages:
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


def get_output_path(run_id: str, task: str) -> Path:
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt.csv"

    return OUT_DIR / f"run{run_id}_{task.lower()}.csv"


def load_base_model():
    print(f"  Loading base model from {MODEL_PATH} ...")
    print("  Mode: full bfloat16, no quantization")
    start = time.time()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    elapsed = time.time() - start

    print(f"  Base model loaded in {elapsed:.1f}s")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {allocated:.1f}GB / {total:.1f}GB")

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


def cleanup_memory(model=None, trainer=None):
    if trainer is not None:
        del trainer

    if model is not None:
        del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(5)
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory after cleanup: {allocated:.2f}GB allocated")


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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = output_ids[0][input_len:]

    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def build_trainer(model, processor, train_dataset: Dataset, run_id: str) -> SFTTrainer:
    sft_config = SFTConfig(
        output_dir=str(TMP_DIR / f"run{run_id}"),
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


def run_single_task(run_id: str, run_config: dict, processor, system_prompt: str, retry_prompt: str) -> None:
    task = run_config["task"]
    aug = run_config["aug"]
    train_path = Path(run_config["train_jsonl"])
    val_path = Path(run_config["val_jsonl"])

    print(f"\n{'=' * 60}")
    print(f"Run {run_id} | Gemma3-12B | LoRA | Task: {task} | Aug: {aug} | Think: N/A")
    print(f"Train file: {train_path}")
    print(f"Val file: {val_path}")

    model = load_base_model()
    model = apply_lora(model)

    print("  Loading train data...")
    raw_train = load_jsonl(train_path)
    formatted = [format_chat_for_training(ex, processor) for ex in raw_train]
    train_dataset = Dataset.from_list(formatted)
    print(f"  Train examples: {len(train_dataset)}")

    trainer = build_trainer(model, processor, train_dataset, run_id)

    print(f"  Training for {NUM_EPOCHS} epochs ...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"  Training done in {train_elapsed / 60:.1f} min")

    model.eval()

    print("  Evaluating on original val set...")
    val_examples = load_jsonl(val_path)
    print(f"  Val examples: {len(val_examples)}")

    predictions = []
    unknowns = 0

    for i, example in enumerate(val_examples):
        user_content = extract_user_content(example)
        raw_output = run_inference(model, processor, system_prompt, user_content)
        pred_label = normalize_label(raw_output)

        if pred_label == UNKNOWN_LABEL:
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            pred_label = normalize_label(raw_output2)
            raw_output = raw_output2

            if pred_label == UNKNOWN_LABEL:
                unknowns += 1

        predictions.append(
            {
                "pred_label": pred_label,
                "raw_output": raw_output,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(val_examples)} | Unknowns: {unknowns}")

    out_path = get_output_path(run_id, task)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_label", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"  Saved predictions → {out_path}")
    print(f"  Unknowns: {unknowns}/{len(predictions)} ({unknowns / len(predictions) * 100:.1f}%)")

    cleanup_memory(model=model, trainer=trainer)


def run_multitask(run_id: str, run_config: dict, processor, system_prompt: str, retry_prompt: str) -> None:
    task = run_config["task"]
    aug = run_config["aug"]
    train_path = Path(run_config["train_jsonl"])
    val_path = Path(run_config["val_jsonl"])

    print(f"\n{'=' * 60}")
    print(f"Run {run_id} | Gemma3-12B | LoRA | Task: {task} | Aug: {aug} | Think: N/A")
    print(f"Train file: {train_path}")
    print(f"Val file: {val_path}")

    model = load_base_model()
    model = apply_lora(model)

    print("  Loading train data...")
    raw_train = load_jsonl(train_path)
    formatted = [format_chat_for_training(ex, processor) for ex in raw_train]
    train_dataset = Dataset.from_list(formatted)
    print(f"  Train examples: {len(train_dataset)}")

    trainer = build_trainer(model, processor, train_dataset, run_id)

    print(f"  Training for {NUM_EPOCHS} epochs ...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"  Training done in {train_elapsed / 60:.1f} min")

    model.eval()

    print("  Evaluating on original val set...")
    val_examples = load_jsonl(val_path)
    print(f"  Val examples: {len(val_examples)}")

    predictions = []
    unknowns = 0

    for i, example in enumerate(val_examples):
        user_content = extract_user_content(example)
        raw_output = run_inference(model, processor, system_prompt, user_content)
        parsed = parse_multitask_output(raw_output)

        if UNKNOWN_LABEL in parsed.values():
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            parsed2 = parse_multitask_output(raw_output2)

            for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
                if parsed[col] == UNKNOWN_LABEL:
                    parsed[col] = parsed2[col]

            raw_output = raw_output2

        if UNKNOWN_LABEL in parsed.values():
            unknowns += 1

        predictions.append(
            {
                "pred_mi": parsed["pred_mi"],
                "pred_ml": parsed["pred_ml"],
                "pred_pg": parsed["pred_pg"],
                "pred_act": parsed["pred_act"],
                "raw_output": raw_output,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(val_examples)} | Rows with Unknown: {unknowns}")

    out_path = get_output_path(run_id, task)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(predictions)

    print(f"  Saved predictions → {out_path}")
    print(f"  Rows with Unknown: {unknowns}/{len(predictions)} ({unknowns / len(predictions) * 100:.1f}%)")

    cleanup_memory(model=model, trainer=trainer)


def main():
    args = build_parser().parse_args()

    run_id = args.run_id
    run_config = RUNS[run_id]
    task = run_config["task"]

    out_path = get_output_path(run_id, task)

    if SKIP_COMPLETED and out_path.exists() and not args.force:
        print(f"Skipping Run {run_id} because prediction CSV already exists:")
        print(out_path)
        print("Use --force to rerun.")
        return

    with track_emissions(f"dataaug-gemma12b-lora-{task.lower()}"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        print("TutorMind — Gemma3-12B LoRA Train + Inference")
        print(f"Run ID: {run_id}")
        print(f"Model: {MODEL_NAME}")
        print(f"Output folder: {OUT_DIR}")
        print(f"Temporary folder: {TMP_DIR}")
        print("Mode: full bfloat16, no quantization")

        print(f"\nLoading prompts from {PROMPTS_PATH} ...")
        with PROMPTS_PATH.open("r", encoding="utf-8") as f:
            prompts = json.load(f)

        single_task_prompts = prompts["single_task_training"]["prompts"]
        multitask_prompt = prompts["multitask_training"]["prompt"]
        retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
        retry_multitask = prompts["retry_prompts"]["prompts"]["multitask"]

        print(f"\nLoading processor from {MODEL_PATH} ...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("Processor loaded.")

        start = time.time()

        if task == "MT":
            run_multitask(
                run_id=run_id,
                run_config=run_config,
                processor=processor,
                system_prompt=multitask_prompt,
                retry_prompt=retry_multitask,
            )
        else:
            prompt_key = TASK_TO_PROMPT_KEY[task]
            system_prompt = single_task_prompts[prompt_key]

            run_single_task(
                run_id=run_id,
                run_config=run_config,
                processor=processor,
                system_prompt=system_prompt,
                retry_prompt=retry_single,
            )

        elapsed = time.time() - start
        print(f"\nRun {run_id} completed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()