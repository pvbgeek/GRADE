#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Qwen3-14B"
BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"
PROMPTS_JSON = "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json"

SINGLE_TASKS = ["MI", "ML", "PG", "Act"]
ALL_TASKS = SINGLE_TASKS + ["MT"]

SINGLE_TASK_PROMPT_KEYS = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}


# ---------------- ARGUMENTS ----------------
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ---------------- HELPERS ----------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def select_prompt(prompts, task):
    if task == "MT":
        return prompts["multitask_training"]["prompt"]
    return prompts["single_task_training"]["prompts"][SINGLE_TASK_PROMPT_KEYS[task]]


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
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


# ---------------- MAIN ----------------
def main():
    args = build_parser().parse_args()

    with track_emissions(f"scaling-qwen14b-lora-train-{args.task.lower()}"):
        set_seed(args.seed)

        prompts = load_json(PROMPTS_JSON)
        prompt = select_prompt(prompts, args.task)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(tokenizer, args.train_jsonl, prompt)

        # =======================
        # 🔥 16-bit (bf16) MODEL
        # =======================
        print("Loading Qwen3-14B in bf16...")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.config.use_cache = False

        print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

        # =======================
        # LoRA CONFIG
        # =======================
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

        # =======================
        # TRAINING CONFIG
        # =======================
        training_args = SFTConfig(
            output_dir=str(Path(args.adapter_out) / args.task),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_length=args.max_seq_length,
            dataset_text_field="text",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
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

        print(f"Saved adapter to {save_path}")


if __name__ == "__main__":
    main()


















# #!/usr/bin/env python3
# """Train a Qwen3-14B LoRA adapter for TutorMind.

# Purpose
# -------
# Fine-tune a task-specific LoRA adapter for MI, ML, PG, Act, or MT using the
# TutorMind chat-format training JSONL and the prompt registry. Runs with
# Qwen3 thinking mode OFF.

# Output
# ------
# - One task-specific adapter directory under ``--adapter-out`` / ``<task>``.
# - A ``train_manifest.json`` file saved alongside the adapter.

# Example
# -------
#     python '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling-Experiments/LORA/Qwen3-14B/train_qwen_lora.py' \\
#       --task MI \\
#       --train-jsonl '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/mistake_identification_train.jsonl' \\
#       --adapter-out '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling-Experiments/LORA/Qwen3-14B/adapters'
# """

# from __future__ import annotations

# import argparse
# import json
# from datetime import datetime
# from pathlib import Path
# from typing import List

# import torch
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     PreTrainedTokenizerFast,
#     set_seed,
# )
# from trl import SFTConfig, SFTTrainer


# MODEL_NAME = "Qwen3-14B"
# BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"
# PROMPTS_JSON = "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json"
# SINGLE_TASKS = ["MI", "ML", "PG", "Act"]
# ALL_TASKS = SINGLE_TASKS + ["MT"]
# METHOD = "LoRA"
# AUG = "None"
# THINK = "OFF"

# SINGLE_TASK_PROMPT_KEYS = {
#     "MI": "Mistake_Identification",
#     "ML": "Mistake_Location",
#     "PG": "Providing_Guidance",
#     "Act": "Actionability",
# }


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="Train a Qwen3-14B LoRA adapter for a TutorMind task."
#     )
#     parser.add_argument("--task", required=True, choices=ALL_TASKS)
#     parser.add_argument("--train-jsonl", required=True)
#     parser.add_argument("--adapter-out", required=True)
#     parser.add_argument("--epochs", type=float, default=3)
#     parser.add_argument("--learning-rate", type=float, default=2e-4)
#     parser.add_argument("--batch-size", type=int, default=2)
#     parser.add_argument("--grad-accum", type=int, default=8)
#     parser.add_argument("--max-seq-length", type=int, default=2048)
#     parser.add_argument("--seed", type=int, default=42)
#     return parser


# def ensure_existing_file(path_str: str, arg_name: str) -> Path:
#     path = Path(path_str)
#     if not path.is_file():
#         raise ValueError(f"{arg_name} does not exist or is not a file: {path}")
#     return path.resolve()


# def validate_args(args: argparse.Namespace) -> None:
#     if args.epochs <= 0:
#         raise ValueError("--epochs must be greater than zero.")
#     if args.learning_rate <= 0:
#         raise ValueError("--learning-rate must be greater than zero.")
#     if args.batch_size <= 0:
#         raise ValueError("--batch-size must be greater than zero.")
#     if args.grad_accum <= 0:
#         raise ValueError("--grad-accum must be greater than zero.")
#     if args.max_seq_length <= 0:
#         raise ValueError("--max-seq-length must be greater than zero.")


# def load_json(path: Path) -> dict:
#     with path.open("r", encoding="utf-8") as handle:
#         return json.load(handle)


# def get_special_token_content(value: object, default: str | None = None) -> str | None:
#     if isinstance(value, str):
#         return value
#     if isinstance(value, dict):
#         content = value.get("content")
#         if isinstance(content, str):
#             return content
#     return default


# def load_tokenizer(model_path: str) -> AutoTokenizer | PreTrainedTokenizerFast:
#     try:
#         return AutoTokenizer.from_pretrained(model_path)
#     except (AttributeError, TypeError, ValueError):
#         base_model_path = Path(model_path)
#         tokenizer_config = load_json(base_model_path / "tokenizer_config.json")
#         tokenizer = PreTrainedTokenizerFast(
#             tokenizer_file=str(base_model_path / "tokenizer.json"),
#             bos_token=get_special_token_content(tokenizer_config.get("bos_token"), "<s>"),
#             eos_token=get_special_token_content(tokenizer_config.get("eos_token"), "</s>"),
#             unk_token=get_special_token_content(tokenizer_config.get("unk_token"), "<unk>"),
#             pad_token=get_special_token_content(tokenizer_config.get("pad_token"), "<pad>"),
#         )
#         tokenizer.chat_template = tokenizer_config.get("chat_template")
#         return tokenizer


# def should_use_bf16() -> bool:
#     return bool(
#         torch.cuda.is_available()
#         and hasattr(torch.cuda, "is_bf16_supported")
#         and torch.cuda.is_bf16_supported()
#     )


# def load_jsonl_records(path: Path) -> List[dict]:
#     records: List[dict] = []
#     with path.open("r", encoding="utf-8") as handle:
#         for line_number, line in enumerate(handle, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 records.append(json.loads(line))
#             except json.JSONDecodeError as exc:
#                 raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
#     return records


# def select_prompt(prompts: dict, task: str) -> str:
#     if task == "MT":
#         prompt = prompts["multitask_training"]["prompt"]
#     else:
#         prompt_key = SINGLE_TASK_PROMPT_KEYS[task]
#         prompt = prompts["single_task_training"]["prompts"][prompt_key]

#     if not isinstance(prompt, str) or not prompt.strip():
#         raise ValueError(f"Prompt registry did not contain a usable prompt for task {task}.")
#     return prompt


# def extract_last_role_content(record: dict, role: str, source_path: Path, row_index: int) -> str:
#     messages = record.get("messages")
#     if not isinstance(messages, list):
#         raise ValueError(f"{source_path} row {row_index} is missing a valid 'messages' list.")

#     for message in reversed(messages):
#         if isinstance(message, dict) and message.get("role") == role:
#             content = message.get("content")
#             if not isinstance(content, str) or not content.strip():
#                 raise ValueError(f"{source_path} row {row_index} has a non-string or empty {role} content.")
#             return content

#     raise ValueError(f"{source_path} row {row_index} has no {role} message.")


# def build_training_messages(record: dict, prompt: str, source_path: Path, row_index: int):
#     user_content = extract_last_role_content(record, "user", source_path, row_index)
#     assistant_content = extract_last_role_content(record, "assistant", source_path, row_index)
#     return [
#         {"role": "system", "content": prompt},
#         {"role": "user", "content": user_content},
#         {"role": "assistant", "content": assistant_content},
#     ]


# def apply_chat_template_no_think(tokenizer: AutoTokenizer, messages, add_generation_prompt: bool) -> str:
#     # Qwen3 chat template accepts enable_thinking; fall back to /no_think hint if
#     # the template variant does not expose the kwarg.
#     try:
#         return tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=add_generation_prompt,
#             enable_thinking=False,
#         )
#     except TypeError:
#         patched = list(messages)
#         for i, message in enumerate(patched):
#             if message.get("role") == "system":
#                 patched[i] = {
#                     "role": "system",
#                     "content": f"{message['content']}\n\n/no_think",
#                 }
#                 break
#         else:
#             patched.insert(0, {"role": "system", "content": "/no_think"})
#         return tokenizer.apply_chat_template(
#             patched,
#             tokenize=False,
#             add_generation_prompt=add_generation_prompt,
#         )


# def build_training_dataset(tokenizer: AutoTokenizer, train_path: Path, prompt: str) -> Dataset:
#     rows = load_jsonl_records(train_path)
#     formatted_rows = []
#     for row_index, record in enumerate(rows):
#         messages = build_training_messages(record, prompt, train_path, row_index)
#         formatted_rows.append(
#             {
#                 "text": apply_chat_template_no_think(
#                     tokenizer,
#                     messages,
#                     add_generation_prompt=False,
#                 )
#             }
#         )
#     return Dataset.from_list(formatted_rows)


# def resolve_adapter_dir(adapter_out: str, task: str) -> Path:
#     return Path(adapter_out).resolve() / task


# def build_manifest(args: argparse.Namespace, train_jsonl: Path, adapter_dir: Path) -> dict:
#     return {
#         "model": MODEL_NAME,
#         "base_model_path": BASE_MODEL_PATH,
#         "method": METHOD,
#         "task": args.task,
#         "aug": AUG,
#         "think": THINK,
#         "train_jsonl": str(train_jsonl),
#         "adapter_out": str(adapter_dir),
#         "epochs": args.epochs,
#         "learning_rate": args.learning_rate,
#         "batch_size": args.batch_size,
#         "grad_accum": args.grad_accum,
#         "max_seq_length": args.max_seq_length,
#         "seed": args.seed,
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#     }


# def save_manifest(path: Path, manifest: dict) -> None:
#     with path.open("w", encoding="utf-8") as handle:
#         json.dump(manifest, handle, indent=2)


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     try:
#         validate_args(args)
#         train_jsonl = ensure_existing_file(args.train_jsonl, "--train-jsonl")
#         prompts_path = ensure_existing_file(PROMPTS_JSON, "PROMPTS_JSON")
#         adapter_dir = resolve_adapter_dir(args.adapter_out, args.task)
#         adapter_dir.mkdir(parents=True, exist_ok=True)

#         set_seed(args.seed)
#         prompts = load_json(prompts_path)
#         prompt = select_prompt(prompts, args.task)

#         tokenizer = load_tokenizer(BASE_MODEL_PATH)
#         if tokenizer.pad_token_id is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"

#         train_dataset = build_training_dataset(tokenizer, train_jsonl, prompt)

#         # ── 4-bit QLoRA — Qwen3-14B ~28GB in bf16 needs quantization for headroom
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#         )
#         print(f"Loading Qwen3-14B (4-bit) from {BASE_MODEL_PATH} ...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL_PATH,
#             quantization_config=quantization_config,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#         )
#         base_model = prepare_model_for_kbit_training(base_model)
#         print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules=[
#                 "q_proj",
#                 "k_proj",
#                 "v_proj",
#                 "o_proj",
#                 "gate_proj",
#                 "up_proj",
#                 "down_proj",
#             ],
#         )
#         model = get_peft_model(base_model, lora_config)
#         model.print_trainable_parameters()

#         training_args = SFTConfig(
#             output_dir=str(adapter_dir),
#             num_train_epochs=args.epochs,
#             per_device_train_batch_size=args.batch_size,
#             gradient_accumulation_steps=args.grad_accum,
#             learning_rate=args.learning_rate,
#             dataset_text_field="text",
#             max_length=args.max_seq_length,
#             packing=False,
#             bf16=should_use_bf16(),
#             fp16=False,
#             save_strategy="epoch",
#             save_total_limit=1,
#             logging_steps=10,
#             report_to="none",
#             seed=args.seed,
#             gradient_checkpointing=True,
#             gradient_checkpointing_kwargs={"use_reentrant": False},
#             optim="adamw_8bit",
#         )

#         trainer = SFTTrainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             processing_class=tokenizer,
#         )
#         trainer.train()
#         trainer.save_model(str(adapter_dir))
#         tokenizer.save_pretrained(str(adapter_dir))

#         manifest = build_manifest(args, train_jsonl, adapter_dir)
#         save_manifest(adapter_dir / "train_manifest.json", manifest)
#         print(f"Saved Qwen3-14B LoRA adapter for task {args.task} to {adapter_dir}")

#     except ValueError as exc:
#         parser.exit(status=2, message=f"Error: {exc}\n")


# if __name__ == "__main__":
#     main()
