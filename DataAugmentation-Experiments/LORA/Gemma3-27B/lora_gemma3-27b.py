#!/usr/bin/env python3

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
import re
import subprocess
import sys
import time
from pathlib import Path

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
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Gemma3-27B"
MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3-27b"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
TRYEVAL_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/tryeval.py")

OUT_DIR = Path.cwd()
TMP_DIR = OUT_DIR / "tmp_gemma3_27b_aug"

UNKNOWN_LABEL = "Unknown"
SKIP_COMPLETED = True

RUNS = {
    "081": {
        "run_id": "081",
        "task": "MI",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl",
    },
    "082": {
        "run_id": "082",
        "task": "ML",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_location_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_location_val.jsonl",
    },
    "083": {
        "run_id": "083",
        "task": "PG",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/providing_guidance_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/providing_guidance_val.jsonl",
    },
    "084": {
        "run_id": "084",
        "task": "Act",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/actionability_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/actionability_val.jsonl",
    },
    "085": {
        "run_id": "085",
        "task": "MT",
        "aug": "Qwen3-Gen",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl",
    },
    "106": {
        "run_id": "106",
        "task": "MI",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl",
    },
    "107": {
        "run_id": "107",
        "task": "ML",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_location_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_location_val.jsonl",
    },
    "108": {
        "run_id": "108",
        "task": "PG",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/providing_guidance_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/providing_guidance_val.jsonl",
    },
    "109": {
        "run_id": "109",
        "task": "Act",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/actionability_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/actionability_val.jsonl",
    },
    "110": {
        "run_id": "110",
        "task": "MT",
        "aug": "Qwen3-Gen+Verify",
        "train_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl",
        "val_jsonl": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl",
    },
}

TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}

VAL_FLAG_MAP = {
    "MI": "val-mi",
    "ML": "val-ml",
    "PG": "val-pg",
    "Act": "val-act",
    "MT": "val-mt",
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
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
SEED = 3407
LOGGING_STEPS = 10
MAX_SEQ_LENGTH = 768
MAX_NEW_TOKENS = 64


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


def load_jsonl(path: Path) -> list:
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


def get_output_path(run_id: str, task: str) -> Path:
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt.csv"

    return OUT_DIR / f"run{run_id}_{task.lower()}.csv"


def get_metrics_path(run_id: str) -> Path:
    return OUT_DIR / f"run{run_id}_metrics.csv"


def print_gpu_memory(label: str):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[GPU Memory] {label} | "
        f"allocated={allocated:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"max={max_allocated:.2f}GB"
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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
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


def run_tryeval(run_config: dict, predictions_path: Path) -> None:
    run_id = run_config["run_id"]
    task = run_config["task"]
    aug = run_config["aug"]
    val_path = Path(run_config["val_jsonl"])
    metrics_path = get_metrics_path(run_id)
    val_flag = VAL_FLAG_MAP[task]

    if not TRYEVAL_PATH.is_file():
        raise FileNotFoundError(f"tryeval.py not found at: {TRYEVAL_PATH}")

    command = [
        sys.executable,
        str(TRYEVAL_PATH),
        "--predictions",
        str(predictions_path),
        "--task",
        task,
        "--run-id",
        run_id,
        "--model",
        MODEL_NAME,
        "--method",
        "LoRA",
        "--aug",
        aug,
        "--think",
        "N/A",
        f"--{val_flag}",
        str(val_path),
        "--out",
        str(metrics_path),
    ]

    print("\nRunning tryeval automatically:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print(f"\n✅ Metrics saved → {metrics_path}")
    print("✅ If your tryeval.py updates master_metrics.csv automatically, the master file is updated too.")


def run_experiment(run_config: dict, force: bool = False, skip_eval: bool = False) -> None:
    run_id = run_config["run_id"]
    task = run_config["task"]
    aug = run_config["aug"]

    train_path = Path(run_config["train_jsonl"])
    val_path = Path(run_config["val_jsonl"])
    out_path = get_output_path(run_id, task)

    print("\n" + "=" * 80)
    print(f"Run {run_id} | Task: {task} | Gemma3-27B | 8-bit Text Attention + MLP LoRA")
    print("=" * 80)
    print(f"Aug:         {aug}")
    print(f"Train file:  {train_path}")
    print(f"Val file:    {val_path}")
    print(f"Output CSV:  {out_path}")
    print(f"Batch size:  {PER_DEVICE_BATCH_SIZE}")
    print(f"Grad accum:  {GRADIENT_ACCUM_STEPS}")
    print(f"Max length:  {MAX_SEQ_LENGTH}")
    print(f"LR:          {LEARNING_RATE}")
    print(f"LoRA target: {LORA_TARGET_MODULES}")
    print(f"Eval auto:   {not skip_eval}")

    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not val_path.is_file():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if SKIP_COMPLETED and out_path.exists() and not force:
        print(f"\nPrediction CSV already exists, skipping train/infer:")
        print(out_path)
        print("Use --force to rerun training and overwrite predictions.")

        if not skip_eval:
            run_tryeval(run_config, out_path)

        return

    print(f"\nLoading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    print(f"\nLoading prompts from {PROMPTS_PATH}...")
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    single_task_prompts = prompts["single_task_training"]["prompts"]
    multitask_prompt = prompts["multitask_training"]["prompt"]
    retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
    retry_multitask = prompts["retry_prompts"]["prompts"]["multitask"]

    if task == "MT":
        system_prompt = multitask_prompt
        retry_prompt = retry_multitask
    else:
        prompt_key = TASK_TO_PROMPT_KEY[task]
        system_prompt = single_task_prompts[prompt_key]
        retry_prompt = retry_single

    print("\nLoading train data...")
    raw_train = load_jsonl(train_path)
    formatted_train = [format_chat_for_training(ex, processor) for ex in raw_train]
    train_dataset = Dataset.from_list(formatted_train)
    print(f"Train examples: {len(train_dataset)}")

    print("\nLoading val data...")
    val_examples = load_jsonl(val_path)
    print(f"Val examples: {len(val_examples)}")

    model = None
    trainer = None

    try:
        model = load_model()
        model = apply_lora(model)

        trainer = build_trainer(model, processor, train_dataset, run_id)

        trainer.model = force_text_lora_trainable(trainer.model)

        print("\nStarting training...")
        train_start = time.time()
        trainer.train()
        train_elapsed = time.time() - train_start
        print(f"\nTraining completed in {train_elapsed / 60:.1f} min")

        model.eval()
        model.config.use_cache = True

        print("\nStarting validation inference...")
        predictions = []
        unknowns = 0

        for i, example in enumerate(val_examples):
            user_content = extract_user_content(example)
            raw_output = run_inference(model, processor, system_prompt, user_content)

            if task == "MT":
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

            else:
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
                print(f"Validation progress: {i + 1}/{len(val_examples)} | Unknowns: {unknowns}")

        fieldnames = list(predictions[0].keys())

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        print(f"\nSaved predictions → {out_path}")
        print(f"Unknowns: {unknowns}/{len(predictions)} ({unknowns / len(predictions) * 100:.1f}%)")

    finally:
        cleanup_memory(model=model, trainer=trainer)

    if not skip_eval:
        run_tryeval(run_config, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma3-27B 8-bit LoRA train + validation + automatic tryeval"
    )

    parser.add_argument(
        "--run-id",
        required=True,
        choices=sorted(RUNS.keys()),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun training/inference even if prediction CSV already exists.",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train/infer; do not run tryeval.py at the end.",
    )

    args = parser.parse_args()

    run_config = RUNS[args.run_id]
    task = run_config["task"]

    with track_emissions(f"dataaug-gemma27b-lora-{task.lower()}"):
        run_experiment(
            run_config=run_config,
            force=args.force,
            skip_eval=args.skip_eval,
        )


if __name__ == "__main__":
    main()