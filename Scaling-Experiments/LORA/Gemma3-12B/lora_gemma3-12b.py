#!/usr/bin/env python3
"""
lora_gemma12b.py
LoRA fine-tuning + inference for Gemma3-12B on all 5 tasks.
Scaling Experiments — Runs 036-040.

Uses full bfloat16 (no quantization) for proper gradient flow.
Gemma3-12B bfloat16 (~25GB) + LoRA overhead (~8GB) = ~33GB
fits comfortably on L40S 46GB.

Workflow per task:
  1. Load base model in bfloat16 (no quantization)
  2. Apply fresh LoRA adapters
  3. Train on task training data
  4. Switch to inference mode
  5. Evaluate on val set → save predictions CSV
  6. Clean up memory
  7. Repeat for next task

No adapters saved to disk — train → eval → discard.

Runs:
  036 | Gemma3-12B | LoRA | MI  | Aug: None | Think: N/A
  037 | Gemma3-12B | LoRA | ML  | Aug: None | Think: N/A
  038 | Gemma3-12B | LoRA | PG  | Aug: None | Think: N/A
  039 | Gemma3-12B | LoRA | Act | Aug: None | Think: N/A
  040 | Gemma3-12B | LoRA | MT  | Aug: None | Think: N/A
"""

import json
import re
import csv
import gc
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/WAVE/datasets/oignat_lab/Gemma3"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
TRAIN_DIR    = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train")
VAL_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
OUT_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling-Experiments/LORA-Gemma3-12B")

# ── Run config ────────────────────────────────────────────────────────────────
RUNS = [
    {"run_id": "036", "task": "MI",  "train_file": "mistake_identification_train.jsonl", "val_file": "mistake_identification_val.jsonl"},
    {"run_id": "037", "task": "ML",  "train_file": "mistake_location_train.jsonl",       "val_file": "mistake_location_val.jsonl"},
    {"run_id": "038", "task": "PG",  "train_file": "providing_guidance_train.jsonl",     "val_file": "providing_guidance_val.jsonl"},
    {"run_id": "039", "task": "Act", "train_file": "actionability_train.jsonl",          "val_file": "actionability_val.jsonl"},
    {"run_id": "040", "task": "MT",  "train_file": "multitask_train.jsonl",              "val_file": "multitask_val.jsonl"},
]

# ── Resume config ─────────────────────────────────────────────────────────────
SKIP_COMPLETED = True

# Task to prompt key mapping
TASK_TO_PROMPT_KEY = {
    "MI":  "Mistake_Identification",
    "ML":  "Mistake_Location",
    "PG":  "Providing_Guidance",
    "Act": "Actionability",
}

# ── LoRA config ───────────────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training config ───────────────────────────────────────────────────────────
NUM_EPOCHS            = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUM_STEPS  = 8
LEARNING_RATE         = 2e-4
WARMUP_STEPS          = 5
WEIGHT_DECAY          = 0.01
SEED                  = 3407

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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_user_content(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found")


def format_chat_for_training(example: dict, processor) -> dict:
    """Format a chat example into text using the processor's chat template."""
    messages = example["messages"]
    converted = []
    for msg in messages:
        converted.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    text = processor.apply_chat_template(
        converted,
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"text": text}


def get_output_path(run_id: str, task: str) -> Path:
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt_predictions.csv"
    return OUT_DIR / f"run{run_id}_{task.lower()}_predictions.csv"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_base_model():
    """Load Gemma3-12B in full bfloat16 — no quantization for proper gradient flow."""
    print(f"  Loading base model from {MODEL_PATH} ...")
    print(f"  Mode: full bfloat16 (no quantization)")
    start = time.time()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    elapsed = time.time() - start
    print(f"  Base model loaded in {elapsed:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    return model


def apply_lora(model):
    """Apply fresh LoRA adapters to the model."""
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
    """Aggressively clean GPU memory between runs."""
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import time
    time.sleep(5)
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory after cleanup: {allocated:.2f}GB allocated")


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_content}]},
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


# ── Training helper ───────────────────────────────────────────────────────────
def build_trainer(model, processor, train_dataset: Dataset, run_id: str) -> SFTTrainer:
    """Build SFTTrainer using trl 0.23.0 SFTConfig API."""
    sft_config = SFTConfig(
        output_dir=str(OUT_DIR / f"tmp_{run_id}"),
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


# ── Single-task run ───────────────────────────────────────────────────────────
def run_single_task(run_config: dict, processor,
                    system_prompt: str, retry_prompt: str) -> None:
    run_id     = run_config["run_id"]
    task       = run_config["task"]
    train_path = TRAIN_DIR / run_config["train_file"]
    val_path   = VAL_DIR   / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | LoRA | Gemma3-12B")

    # Load + apply LoRA
    model = load_base_model()
    model = apply_lora(model)

    # Prepare training data
    print(f"  Loading train data: {train_path.name}")
    raw_train = load_jsonl(train_path)
    formatted = [format_chat_for_training(ex, processor) for ex in raw_train]
    train_dataset = Dataset.from_list(formatted)
    print(f"  Train examples: {len(train_dataset)}")

    # Train
    trainer = build_trainer(model, processor, train_dataset, run_id)
    print(f"  Training for {NUM_EPOCHS} epochs ...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"  Training done in {train_elapsed/60:.1f} min")

    # Switch to inference mode
    model.eval()

    # Evaluate on val set
    print(f"  Evaluating on val set: {val_path.name}")
    val_examples = load_jsonl(val_path)
    print(f"  Val examples: {len(val_examples)}")

    predictions = []
    unknowns    = 0

    for i, example in enumerate(val_examples):
        user_content = extract_user_content(example)
        raw_output   = run_inference(model, processor, system_prompt, user_content)
        pred_label   = normalize_label(raw_output)

        if pred_label == "Unknown":
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            pred_label  = normalize_label(raw_output2)
            raw_output  = raw_output2
            if pred_label == "Unknown":
                unknowns += 1

        predictions.append({"pred_label": pred_label, "raw_output": raw_output})

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(val_examples)} | Unknowns: {unknowns}")

    # Save CSV
    out_path = get_output_path(run_id, task)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_label", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"  Saved → {out_path.name}")
    print(f"  Unknowns: {unknowns}/{len(predictions)} ({unknowns/len(predictions)*100:.1f}%)")

    cleanup_memory(model=model, trainer=trainer)


# ── Multitask run ─────────────────────────────────────────────────────────────
def run_multitask(run_config: dict, processor,
                  system_prompt: str, retry_prompt: str) -> None:
    run_id     = run_config["run_id"]
    task       = run_config["task"]
    train_path = TRAIN_DIR / run_config["train_file"]
    val_path   = VAL_DIR   / run_config["val_file"]

    print(f"\n{'='*60}")
    print(f"Run {run_id} | Task: {task} | LoRA | Gemma3-12B")

    # Load + apply LoRA
    model = load_base_model()
    model = apply_lora(model)

    # Prepare training data
    print(f"  Loading train data: {train_path.name}")
    raw_train = load_jsonl(train_path)
    formatted = [format_chat_for_training(ex, processor) for ex in raw_train]
    train_dataset = Dataset.from_list(formatted)
    print(f"  Train examples: {len(train_dataset)}")

    # Train
    trainer = build_trainer(model, processor, train_dataset, run_id)
    print(f"  Training for {NUM_EPOCHS} epochs ...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"  Training done in {train_elapsed/60:.1f} min")

    # Switch to inference mode
    model.eval()

    # Evaluate on val set
    print(f"  Evaluating on val set: {val_path.name}")
    val_examples = load_jsonl(val_path)
    print(f"  Val examples: {len(val_examples)}")

    predictions = []
    unknowns    = 0

    for i, example in enumerate(val_examples):
        user_content = extract_user_content(example)
        raw_output   = run_inference(model, processor, system_prompt, user_content)
        parsed       = parse_multitask_output(raw_output)

        if "Unknown" in parsed.values():
            raw_output2 = run_inference(model, processor, retry_prompt, user_content)
            parsed2     = parse_multitask_output(raw_output2)
            for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
                if parsed[col] == "Unknown":
                    parsed[col] = parsed2[col]
            raw_output = raw_output2

        if "Unknown" in parsed.values():
            unknowns += 1

        predictions.append({
            "pred_mi":    parsed["pred_mi"],
            "pred_ml":    parsed["pred_ml"],
            "pred_pg":    parsed["pred_pg"],
            "pred_act":   parsed["pred_act"],
            "raw_output": raw_output,
        })

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(val_examples)} | Rows with Unknown: {unknowns}")

    # Save CSV
    out_path = get_output_path(run_id, task)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"  Saved → {out_path.name}")
    print(f"  Rows with Unknown: {unknowns}/{len(predictions)} ({unknowns/len(predictions)*100:.1f}%)")

    cleanup_memory(model=model, trainer=trainer)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("TutorMind — LoRA Fine-tuning + Inference")
    print("Model: Gemma3-12B")
    print("Experiment: Scaling — Runs 036-040")
    print(f"Output dir: {OUT_DIR}")
    print(f"Skip completed runs: {SKIP_COMPLETED}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"\nLoading prompts from {PROMPTS_PATH} ...")
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    single_task_prompts = prompts["single_task_training"]["prompts"]
    multitask_prompt    = prompts["multitask_training"]["prompt"]
    retry_single        = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask     = prompts["retry_prompts"]["prompts"]["multitask"]

    # Load processor ONCE
    print(f"\nLoading processor from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Processor loaded!")

    # Run all 5 tasks
    total_start = time.time()

    for run_config in RUNS:
        run_id = run_config["run_id"]
        task   = run_config["task"]

        if SKIP_COMPLETED and get_output_path(run_id, task).exists():
            print(f"\nSkipping Run {run_id} | Task: {task} — CSV already exists ✅")
            continue

        run_start = time.time()

        if task == "MT":
            run_multitask(
                run_config, processor,
                system_prompt=multitask_prompt,
                retry_prompt=retry_multitask,
            )
        else:
            prompt_key    = TASK_TO_PROMPT_KEY[task]
            system_prompt = single_task_prompts[prompt_key]
            run_single_task(
                run_config, processor,
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
            f"--model Gemma3-12B --method LoRA --aug None --think N/A "
            f"--{val_flag} {VAL_DIR}/{val_file} "
            f"--out {OUT_DIR}/run{run_id}_metrics.csv"
        )


if __name__ == "__main__":
    main()