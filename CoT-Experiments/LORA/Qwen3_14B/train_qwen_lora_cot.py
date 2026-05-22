#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from utils.codecarbon_helper import track_emissions


BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"
PROMPTS_JSON = "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json"

SINGLE_TASK_PROMPT_KEYS = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    return parser


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    return [json.loads(x) for x in open(path)]


def select_prompt(prompts, task):
    if task == "MT":
        return prompts["multitask_thinking"]["prompt"]
    return prompts["single_task_thinking"]["prompts"][SINGLE_TASK_PROMPT_KEYS[task]]


def build_dataset(tokenizer, path, prompt):
    data = load_jsonl(path)
    texts = []

    for row in data:
        msgs = row["messages"]

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": msgs[-2]["content"]},
            {"role": "assistant", "content": msgs[-1]["content"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,  # ✅ CoT in training
        )

        texts.append({"text": text})

    return Dataset.from_list(texts)


def main():
    args = build_parser().parse_args()

    with track_emissions(f"cot-qwen14b-lora-train-{args.task.lower()}"):
        set_seed(42)

        prompts = load_json(PROMPTS_JSON)
        prompt = select_prompt(prompts, args.task)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(tokenizer, args.train_jsonl, prompt)

        print("Loading Qwen3-14B in bf16...")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.config.use_cache = False

        # ✅ LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=str(Path(args.adapter_out) / args.task),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-4,
            max_length=2048,
            dataset_text_field="text",
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        save_path = Path(args.adapter_out) / args.task
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()