#!/usr/bin/env python3

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse, csv, gc, json, re, time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer


MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3-27b"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
TRAIN_DIR = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train")
VAL_DIR = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
OUT_DIR = Path(__file__).resolve().parent

RUNS = {
    "046": {"run_id": "046", "task": "MI", "train_file": "mistake_identification_train.jsonl", "val_file": "mistake_identification_val.jsonl"},
    "047": {"run_id": "047", "task": "ML", "train_file": "mistake_location_train.jsonl", "val_file": "mistake_location_val.jsonl"},
    "048": {"run_id": "048", "task": "PG", "train_file": "providing_guidance_train.jsonl", "val_file": "providing_guidance_val.jsonl"},
    "049": {"run_id": "049", "task": "Act", "train_file": "actionability_train.jsonl", "val_file": "actionability_val.jsonl"},
    "050": {"run_id": "050", "task": "MT", "train_file": "multitask_train.jsonl", "val_file": "multitask_val.jsonl"},
}

TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}

# IMPORTANT: False because you are rerunning failed 046/047 CSVs.
SKIP_COMPLETED = False

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Attention + MLP LoRA, matching Gemma3-12B setup.
# Vision LoRA will still be frozen later.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
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


def normalize_label(text: str) -> str:
    if not text:
        return "Unknown"

    text = str(text).replace("<end_of_turn>", "").replace("</s>", "").strip()

    candidates = [text]

    for line in text.splitlines():
        line = line.strip()
        if line:
            candidates.append(line)
        if ":" in line:
            candidates.append(line.split(":", 1)[1].strip())

    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'`.,!?:;").lower()
        cleaned = re.sub(r"\s+", " ", cleaned)

        if "to some extent" in cleaned or "to some extend" in cleaned:
            return "To some extent"
        if cleaned == "no" or cleaned.startswith("no "):
            return "No"
        if cleaned == "yes" or cleaned.startswith("yes "):
            return "Yes"

    return "Unknown"


def parse_multitask_output(text: str) -> dict:
    field_map = {
        "mistakeidentification": "pred_mi",
        "mistakelocation": "pred_ml",
        "providingguidance": "pred_pg",
        "actionability": "pred_act",
    }

    result = {"pred_mi": "Unknown", "pred_ml": "Unknown", "pred_pg": "Unknown", "pred_act": "Unknown"}

    text = str(text).replace("<end_of_turn>", "").replace("</s>", "").strip()

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
    converted = []
    for msg in example["messages"]:
        converted.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}],
        })

    text = processor.apply_chat_template(
        converted,
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"text": text}


def get_output_path(run_id: str) -> Path:
    return OUT_DIR / f"run{run_id}.csv"


def print_gpu_memory(label: str):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[GPU Memory] {label} | "
        f"allocated={allocated:.2f}GB | reserved={reserved:.2f}GB | max={max_allocated:.2f}GB"
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
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print_gpu_memory("after cleanup")


def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(model.device) for key, value in inputs.items() if torch.is_tensor(value)}
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
        output_dir=str(OUT_DIR / f"tmp_{run_id}"),
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


def run_experiment(run_config: dict):
    run_id = run_config["run_id"]
    task = run_config["task"]

    train_path = TRAIN_DIR / run_config["train_file"]
    val_path = VAL_DIR / run_config["val_file"]
    out_path = get_output_path(run_id)

    print("\n" + "=" * 80)
    print(f"Run {run_id} | Task: {task} | Gemma3-27B | 8-bit Text Attention + MLP LoRA")
    print("=" * 80)
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")
    print(f"Output:     {out_path}")
    print(f"Batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"Grad accum: {GRADIENT_ACCUM_STEPS}")
    print(f"Max length: {MAX_SEQ_LENGTH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"LoRA target modules: {LORA_TARGET_MODULES}")
    print(f"Logging every {LOGGING_STEPS} optimizer steps")

    if SKIP_COMPLETED and out_path.exists():
        print(f"\nRun {run_id} already completed — CSV exists ✅")
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

        # SFTTrainer may alter requires_grad, so force again after trainer construction.
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

                if "Unknown" in parsed.values():
                    raw_output2 = run_inference(model, processor, retry_prompt, user_content)
                    parsed2 = parse_multitask_output(raw_output2)

                    for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
                        if parsed[col] == "Unknown":
                            parsed[col] = parsed2[col]

                    raw_output = raw_output2

                if "Unknown" in parsed.values():
                    unknowns += 1

                predictions.append({
                    "pred_mi": parsed["pred_mi"],
                    "pred_ml": parsed["pred_ml"],
                    "pred_pg": parsed["pred_pg"],
                    "pred_act": parsed["pred_act"],
                    "raw_output": raw_output,
                })

            else:
                pred_label = normalize_label(raw_output)

                if pred_label == "Unknown":
                    raw_output2 = run_inference(model, processor, retry_prompt, user_content)
                    pred_label = normalize_label(raw_output2)
                    raw_output = raw_output2

                    if pred_label == "Unknown":
                        unknowns += 1

                predictions.append({
                    "pred_label": pred_label,
                    "raw_output": raw_output,
                })

            if (i + 1) % 50 == 0:
                print(f"Validation progress: {i + 1}/{len(val_examples)} | Unknowns: {unknowns}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = list(predictions[0].keys())

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        print(f"\nSaved predictions → {out_path}")
        print(f"Unknowns: {unknowns}/{len(predictions)} ({unknowns / len(predictions) * 100:.1f}%)")

        val_flag = {"MI": "val-mi", "ML": "val-ml", "PG": "val-pg", "Act": "val-act", "MT": "val-mt"}[task]

        print("\nNext eval command:")
        print(
            f"python tryeval.py "
            f"--predictions {out_path} "
            f"--task {task} --run-id {run_id} "
            f"--model Gemma3-27B --method LoRA --aug None --think N/A "
            f"--{val_flag} {val_path} "
            f"--out {OUT_DIR}/run{run_id}_metrics.csv"
        )

    finally:
        cleanup_memory(model=model, trainer=trainer)


def main():
    parser = argparse.ArgumentParser(description="Gemma3-27B 8-bit text attention + MLP LoRA training + validation")
    parser.add_argument("--run-id", required=True, choices=list(RUNS.keys()))
    args = parser.parse_args()
    run_experiment(RUNS[args.run_id])


if __name__ == "__main__":
    main()


















# #!/usr/bin/env python3

# import os
# os.environ["USE_TF"] = "0"
# os.environ["USE_TORCH"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# import argparse, csv, gc, json, re, time
# from pathlib import Path

# import torch
# from datasets import Dataset
# from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration, TrainerCallback
# from peft import LoraConfig, TaskType, get_peft_model
# from trl import SFTConfig, SFTTrainer


# MODEL_PATH = "/WAVE/datasets/oignat_lab/Gemma3-27b"
# PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
# TRAIN_DIR = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train")
# VAL_DIR = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
# OUT_DIR = Path(__file__).resolve().parent

# RUNS = {
#     "046": {"run_id": "046", "task": "MI", "train_file": "mistake_identification_train.jsonl", "val_file": "mistake_identification_val.jsonl"},
#     "047": {"run_id": "047", "task": "ML", "train_file": "mistake_location_train.jsonl", "val_file": "mistake_location_val.jsonl"},
#     "048": {"run_id": "048", "task": "PG", "train_file": "providing_guidance_train.jsonl", "val_file": "providing_guidance_val.jsonl"},
#     "049": {"run_id": "049", "task": "Act", "train_file": "actionability_train.jsonl", "val_file": "actionability_val.jsonl"},
#     "050": {"run_id": "050", "task": "MT", "train_file": "multitask_train.jsonl", "val_file": "multitask_val.jsonl"},
# }

# TASK_TO_PROMPT_KEY = {
#     "MI": "Mistake_Identification",
#     "ML": "Mistake_Location",
#     "PG": "Providing_Guidance",
#     "Act": "Actionability",
# }

# SKIP_COMPLETED = True

# LORA_R = 16
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.0

# # Text-attention only. This avoids MLP LoRA.
# # We still force-freeze vision LoRA later because Gemma3 vision tower also has q/k/v/o.
# LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# NUM_EPOCHS = 3
# PER_DEVICE_BATCH_SIZE = 1
# GRADIENT_ACCUM_STEPS = 8
# LEARNING_RATE = 1e-4
# WARMUP_STEPS = 5
# WEIGHT_DECAY = 0.01
# SEED = 3407
# LOGGING_STEPS = 10
# MAX_SEQ_LENGTH = 768
# MAX_NEW_TOKENS = 64


# def normalize_label(text: str) -> str:
#     if not text:
#         return "Unknown"

#     candidates = [text.strip()]
#     for line in text.splitlines():
#         line = line.strip()
#         if line:
#             candidates.append(line)
#         if ":" in line:
#             candidates.append(line.split(":", 1)[1].strip())

#     for candidate in candidates:
#         cleaned = candidate.strip().strip("\"'`.,!?:;").lower()
#         cleaned = re.sub(r"\s+", " ", cleaned)
#         if cleaned == "yes":
#             return "Yes"
#         if cleaned == "no":
#             return "No"
#         if cleaned in {"to some extent", "to some extend"}:
#             return "To some extent"

#     return "Unknown"


# def parse_multitask_output(text: str) -> dict:
#     field_map = {
#         "mistakeidentification": "pred_mi",
#         "mistakelocation": "pred_ml",
#         "providingguidance": "pred_pg",
#         "actionability": "pred_act",
#     }

#     result = {"pred_mi": "Unknown", "pred_ml": "Unknown", "pred_pg": "Unknown", "pred_act": "Unknown"}

#     for line in text.splitlines():
#         line = line.strip()
#         if ":" not in line:
#             continue

#         field, value = line.split(":", 1)
#         key = field.strip().lower().replace(" ", "").replace("_", "")
#         col = field_map.get(key)

#         if col:
#             result[col] = normalize_label(value.strip())

#     return result


# def load_jsonl(path: Path) -> list:
#     examples = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 examples.append(json.loads(line))
#     return examples


# def extract_user_content(example: dict) -> str:
#     for msg in example["messages"]:
#         if msg["role"] == "user":
#             return msg["content"]
#     raise ValueError("No user message found")


# def format_chat_for_training(example: dict, processor) -> dict:
#     converted = []
#     for msg in example["messages"]:
#         converted.append({
#             "role": msg["role"],
#             "content": [{"type": "text", "text": msg["content"]}],
#         })

#     text = processor.apply_chat_template(
#         converted,
#         add_generation_prompt=False,
#         tokenize=False,
#     )
#     return {"text": text}


# def get_output_path(run_id: str) -> Path:
#     return OUT_DIR / f"run{run_id}.csv"


# def print_gpu_memory(label: str):
#     if not torch.cuda.is_available():
#         return

#     allocated = torch.cuda.memory_allocated() / 1024**3
#     reserved = torch.cuda.memory_reserved() / 1024**3
#     max_allocated = torch.cuda.max_memory_allocated() / 1024**3

#     print(
#         f"[GPU Memory] {label} | "
#         f"allocated={allocated:.2f}GB | reserved={reserved:.2f}GB | max={max_allocated:.2f}GB"
#     )


# def force_text_lora_trainable(model):
#     """
#     Critical fix:
#     1. Freeze everything.
#     2. Enable only text-side LoRA params.
#     3. Explicitly exclude vision_tower LoRA params.
#     """
#     for _, param in model.named_parameters():
#         param.requires_grad = False

#     if hasattr(model, "enable_adapter_layers"):
#         model.enable_adapter_layers()

#     text_tensors = 0
#     text_params = 0
#     vision_tensors = 0
#     vision_params = 0

#     for name, param in model.named_parameters():
#         lname = name.lower()

#         is_lora = "lora" in lname
#         is_vision = "vision_tower" in lname or "vision_model" in lname
#         is_text = "language_model" in lname or "text_model" in lname

#         if is_lora and is_vision:
#             param.requires_grad = False
#             vision_tensors += 1
#             vision_params += param.numel()

#         elif is_lora and is_text:
#             param.requires_grad = True
#             text_tensors += 1
#             text_params += param.numel()

#     if text_params == 0:
#         print("\n[Warning] No params matched language_model/text_model. Falling back to non-vision LoRA params.")
#         for name, param in model.named_parameters():
#             lname = name.lower()
#             is_lora = "lora" in lname
#             is_vision = "vision_tower" in lname or "vision_model" in lname

#             if is_lora and not is_vision:
#                 param.requires_grad = True
#                 text_tensors += 1
#                 text_params += param.numel()

#     print("\n[Text-Only LoRA Trainable]")
#     print(f"  text LoRA tensors trainable: {text_tensors}")
#     print(f"  text LoRA params trainable:  {text_params:,}")
#     print(f"  vision LoRA tensors frozen:  {vision_tensors}")
#     print(f"  vision LoRA params frozen:   {vision_params:,}")

#     if text_params == 0:
#         raise RuntimeError("No text LoRA params found. Check Gemma3 module names.")

#     return model


# class TrainDebugCallback(TrainerCallback):
#     def __init__(self, every_n_steps: int = 10):
#         self.every_n_steps = every_n_steps
#         self.tracked_name = None
#         self.initial_tensor = None

#     def _find_trainable_param(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 return name, param
#         return None, None

#     def on_train_begin(self, args, state, control, model=None, **kwargs):
#         trainable_count = 0
#         trainable_tensors = 0
#         vision_trainable = 0

#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 trainable_count += param.numel()
#                 trainable_tensors += 1
#                 if "vision_tower" in name.lower() or "vision_model" in name.lower():
#                     vision_trainable += param.numel()

#         print("\n[Trainable Check]")
#         print(f"  trainable tensors: {trainable_tensors}")
#         print(f"  trainable params:  {trainable_count:,}")
#         print(f"  trainable vision params: {vision_trainable:,}")

#         name, param = self._find_trainable_param(model)

#         if param is None:
#             print("[Trainable Check] ERROR: No trainable parameter found.")
#             return

#         self.tracked_name = name
#         self.initial_tensor = param.detach().float().cpu().clone()

#         print("\n[Weight Tracking]")
#         print(f"  tracking: {self.tracked_name}")
#         print(f"  shape: {tuple(param.shape)}")
#         print(f"  requires_grad: {param.requires_grad}")

#     def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
#         if state.global_step % self.every_n_steps != 0:
#             return

#         total_sq = 0.0
#         max_abs = 0.0
#         tensors_with_grad = 0
#         trainable_tensors = 0

#         for _, param in model.named_parameters():
#             if param.requires_grad:
#                 trainable_tensors += 1
#                 if param.grad is not None:
#                     grad = param.grad.detach().float()
#                     norm = grad.norm().item()
#                     total_sq += norm * norm
#                     max_abs = max(max_abs, grad.abs().max().item())
#                     tensors_with_grad += 1

#         grad_norm = total_sq ** 0.5

#         print(
#             "\n[Real Grad Check]"
#             f" step={state.global_step}"
#             f" | trainable_grad_norm={grad_norm:.8f}"
#             f" | max_abs_grad={max_abs:.8e}"
#             f" | tensors_with_grad={tensors_with_grad}/{trainable_tensors}"
#         )

#     def on_log(self, args, state, control, logs=None, model=None, **kwargs):
#         if self.tracked_name is None or self.initial_tensor is None:
#             return
#         if state.global_step % self.every_n_steps != 0:
#             return

#         current_param = None
#         for name, param in model.named_parameters():
#             if name == self.tracked_name:
#                 current_param = param
#                 break

#         if current_param is None:
#             return

#         current = current_param.detach().float().cpu()
#         delta = (current - self.initial_tensor).abs().mean().item()

#         print(f"[Weight Change] step={state.global_step} | mean_abs_delta={delta:.8e}")
#         print_gpu_memory(f"after log step {state.global_step}")


# def load_model():
#     print("\nLoading Gemma3-27B in 8-bit...")
#     print(f"Model path: {MODEL_PATH}")

#     if torch.cuda.is_available():
#         torch.cuda.reset_peak_memory_stats()

#     start = time.time()

#     bnb_config = BitsAndBytesConfig(load_in_8bit=True)

#     model = Gemma3ForConditionalGeneration.from_pretrained(
#         MODEL_PATH,
#         quantization_config=bnb_config,
#         device_map="auto",
#         attn_implementation="eager",
#         dtype=torch.bfloat16,
#     )

#     model.config.use_cache = False
#     if hasattr(model, "generation_config"):
#         model.generation_config.use_cache = False

#     if hasattr(model, "gradient_checkpointing_enable"):
#         model.gradient_checkpointing_enable(
#             gradient_checkpointing_kwargs={"use_reentrant": False}
#         )

#     if hasattr(model, "enable_input_require_grads"):
#         model.enable_input_require_grads()

#     elapsed = time.time() - start
#     print(f"Base model loaded in {elapsed:.1f}s")
#     print_gpu_memory("after base model load")

#     return model


# def apply_lora(model):
#     print("\nApplying text-attention LoRA adapters...")

#     lora_config = LoraConfig(
#         r=LORA_R,
#         lora_alpha=LORA_ALPHA,
#         lora_dropout=LORA_DROPOUT,
#         bias="none",
#         task_type=TaskType.CAUSAL_LM,
#         target_modules=LORA_TARGET_MODULES,
#     )

#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     model = force_text_lora_trainable(model)

#     trainable = 0
#     total = 0
#     vision_trainable = 0

#     for name, param in model.named_parameters():
#         total += param.numel()
#         if param.requires_grad:
#             trainable += param.numel()
#             if "vision_tower" in name.lower() or "vision_model" in name.lower():
#                 vision_trainable += param.numel()

#     print(f"Final trainable parameters check: {trainable:,} / {total:,}")
#     print(f"Final trainable vision params: {vision_trainable:,}")

#     if trainable == 0:
#         raise RuntimeError("No trainable parameters found after text-only LoRA filtering.")

#     if vision_trainable != 0:
#         raise RuntimeError("Vision parameters are still trainable. Stop and fix filtering.")

#     print_gpu_memory("after text-only LoRA attach")
#     return model


# def cleanup_memory(model=None, trainer=None):
#     print("\nCleaning memory...")

#     if trainer is not None:
#         del trainer
#     if model is not None:
#         del model

#     gc.collect()
#     torch.cuda.empty_cache()

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         print_gpu_memory("after cleanup")


# def run_inference(model, processor, system_prompt: str, user_content: str) -> str:
#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
#         {"role": "user", "content": [{"type": "text", "text": user_content}]},
#     ]

#     inputs = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     )

#     inputs = {key: value.to(model.device) for key, value in inputs.items() if torch.is_tensor(value)}
#     input_len = inputs["input_ids"].shape[-1]

#     with torch.inference_mode():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             use_cache=True,
#         )

#     new_tokens = output_ids[0][input_len:]
#     return processor.decode(new_tokens, skip_special_tokens=True).strip()


# def build_trainer(model, processor, train_dataset: Dataset, run_id: str) -> SFTTrainer:
#     sft_config = SFTConfig(
#         output_dir=str(OUT_DIR / f"tmp_{run_id}"),
#         per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
#         gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
#         warmup_steps=WARMUP_STEPS,
#         num_train_epochs=NUM_EPOCHS,
#         learning_rate=LEARNING_RATE,
#         bf16=True,
#         fp16=False,
#         logging_steps=LOGGING_STEPS,
#         optim="adamw_8bit",
#         weight_decay=WEIGHT_DECAY,
#         lr_scheduler_type="linear",
#         seed=SEED,
#         report_to="none",
#         save_strategy="no",
#         dataset_text_field="text",
#         max_length=MAX_SEQ_LENGTH,
#         gradient_checkpointing=False,
#     )

#     trainer = SFTTrainer(
#         model=model,
#         processing_class=processor.tokenizer,
#         train_dataset=train_dataset,
#         args=sft_config,
#         callbacks=[TrainDebugCallback(every_n_steps=LOGGING_STEPS)],
#     )

#     return trainer


# def run_experiment(run_config: dict):
#     run_id = run_config["run_id"]
#     task = run_config["task"]

#     train_path = TRAIN_DIR / run_config["train_file"]
#     val_path = VAL_DIR / run_config["val_file"]
#     out_path = get_output_path(run_id)

#     print("\n" + "=" * 80)
#     print(f"Run {run_id} | Task: {task} | Gemma3-27B | 8-bit Text-Only LoRA")
#     print("=" * 80)
#     print(f"Train file: {train_path}")
#     print(f"Val file:   {val_path}")
#     print(f"Output:     {out_path}")
#     print(f"Batch size: {PER_DEVICE_BATCH_SIZE}")
#     print(f"Grad accum: {GRADIENT_ACCUM_STEPS}")
#     print(f"Max length: {MAX_SEQ_LENGTH}")
#     print(f"Learning rate: {LEARNING_RATE}")
#     print(f"LoRA target modules: {LORA_TARGET_MODULES}")
#     print(f"Logging every {LOGGING_STEPS} optimizer steps")

#     if SKIP_COMPLETED and out_path.exists():
#         print(f"\nRun {run_id} already completed — CSV exists ✅")
#         return

#     print(f"\nLoading processor from {MODEL_PATH}...")
#     processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

#     print(f"\nLoading prompts from {PROMPTS_PATH}...")
#     with PROMPTS_PATH.open("r", encoding="utf-8") as f:
#         prompts = json.load(f)

#     single_task_prompts = prompts["single_task_training"]["prompts"]
#     multitask_prompt = prompts["multitask_training"]["prompt"]
#     retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
#     retry_multitask = prompts["retry_prompts"]["prompts"]["multitask"]

#     if task == "MT":
#         system_prompt = multitask_prompt
#         retry_prompt = retry_multitask
#     else:
#         prompt_key = TASK_TO_PROMPT_KEY[task]
#         system_prompt = single_task_prompts[prompt_key]
#         retry_prompt = retry_single

#     print("\nLoading train data...")
#     raw_train = load_jsonl(train_path)
#     formatted_train = [format_chat_for_training(ex, processor) for ex in raw_train]
#     train_dataset = Dataset.from_list(formatted_train)
#     print(f"Train examples: {len(train_dataset)}")

#     print("\nLoading val data...")
#     val_examples = load_jsonl(val_path)
#     print(f"Val examples: {len(val_examples)}")

#     model = None
#     trainer = None

#     try:
#         model = load_model()
#         model = apply_lora(model)

#         trainer = build_trainer(model, processor, train_dataset, run_id)

#         # SFTTrainer may alter requires_grad, so force again after trainer construction.
#         trainer.model = force_text_lora_trainable(trainer.model)

#         print("\nStarting training...")
#         train_start = time.time()
#         trainer.train()
#         train_elapsed = time.time() - train_start
#         print(f"\nTraining completed in {train_elapsed / 60:.1f} min")

#         model.eval()
#         model.config.use_cache = True

#         print("\nStarting validation inference...")
#         predictions = []
#         unknowns = 0

#         for i, example in enumerate(val_examples):
#             user_content = extract_user_content(example)
#             raw_output = run_inference(model, processor, system_prompt, user_content)

#             if task == "MT":
#                 parsed = parse_multitask_output(raw_output)

#                 if "Unknown" in parsed.values():
#                     raw_output2 = run_inference(model, processor, retry_prompt, user_content)
#                     parsed2 = parse_multitask_output(raw_output2)

#                     for col in ["pred_mi", "pred_ml", "pred_pg", "pred_act"]:
#                         if parsed[col] == "Unknown":
#                             parsed[col] = parsed2[col]

#                     raw_output = raw_output2

#                 if "Unknown" in parsed.values():
#                     unknowns += 1

#                 predictions.append({
#                     "pred_mi": parsed["pred_mi"],
#                     "pred_ml": parsed["pred_ml"],
#                     "pred_pg": parsed["pred_pg"],
#                     "pred_act": parsed["pred_act"],
#                     "raw_output": raw_output,
#                 })

#             else:
#                 pred_label = normalize_label(raw_output)

#                 if pred_label == "Unknown":
#                     raw_output2 = run_inference(model, processor, retry_prompt, user_content)
#                     pred_label = normalize_label(raw_output2)
#                     raw_output = raw_output2

#                     if pred_label == "Unknown":
#                         unknowns += 1

#                 predictions.append({
#                     "pred_label": pred_label,
#                     "raw_output": raw_output,
#                 })

#             if (i + 1) % 50 == 0:
#                 print(f"Validation progress: {i + 1}/{len(val_examples)} | Unknowns: {unknowns}")

#         OUT_DIR.mkdir(parents=True, exist_ok=True)
#         fieldnames = list(predictions[0].keys())

#         with out_path.open("w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(predictions)

#         print(f"\nSaved predictions → {out_path}")
#         print(f"Unknowns: {unknowns}/{len(predictions)} ({unknowns / len(predictions) * 100:.1f}%)")

#         val_flag = {"MI": "val-mi", "ML": "val-ml", "PG": "val-pg", "Act": "val-act", "MT": "val-mt"}[task]

#         print("\nNext eval command:")
#         print(
#             f"python tryeval.py "
#             f"--predictions {out_path} "
#             f"--task {task} --run-id {run_id} "
#             f"--model Gemma3-27B --method LoRA --aug None --think N/A "
#             f"--{val_flag} {val_path} "
#             f"--out {OUT_DIR}/run{run_id}_metrics.csv"
#         )

#     finally:
#         cleanup_memory(model=model, trainer=trainer)


# def main():
#     parser = argparse.ArgumentParser(description="Gemma3-27B 8-bit text-only LoRA training + validation")
#     parser.add_argument("--run-id", required=True, choices=list(RUNS.keys()))
#     args = parser.parse_args()
#     run_experiment(RUNS[args.run_id])


# if __name__ == "__main__":
#     main()