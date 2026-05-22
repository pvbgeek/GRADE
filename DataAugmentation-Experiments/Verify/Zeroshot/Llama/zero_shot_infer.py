#!/usr/bin/env python3

import argparse
import csv
import json
import re
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.codecarbon_helper import track_emissions


MODEL_PATH = "/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct"

PROMPTS_PATH = Path(
    "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json"
)

OUT_DIR = Path(
    "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Baseline-Experiments/LLaMA-ZeroShot"
)

UNKNOWN_LABEL = "Unknown"
MAX_NEW_TOKENS = 64
SKIP_COMPLETED = True


RUN_CONFIGS = {
    "131": {
        "task": "MI",
        "aug": "Qwen3-Gen",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl",
    },
    "132": {
        "task": "ML",
        "aug": "Qwen3-Gen",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/mistake_location_train_aug_qwen3_gen500.jsonl",
    },
    "133": {
        "task": "PG",
        "aug": "Qwen3-Gen",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/providing_guidance_train_aug_qwen3_gen500.jsonl",
    },
    "134": {
        "task": "Act",
        "aug": "Qwen3-Gen",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/actionability_train_aug_qwen3_gen500.jsonl",
    },
    "135": {
        "task": "MT",
        "aug": "Qwen3-Gen",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl",
    },
    "156": {
        "task": "MI",
        "aug": "Qwen3-Gen+Verify",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl",
    },
    "157": {
        "task": "ML",
        "aug": "Qwen3-Gen+Verify",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/mistake_location_train_aug_qwen3_genverify500.jsonl",
    },
    "158": {
        "task": "PG",
        "aug": "Qwen3-Gen+Verify",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/providing_guidance_train_aug_qwen3_genverify500.jsonl",
    },
    "159": {
        "task": "Act",
        "aug": "Qwen3-Gen+Verify",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/actionability_train_aug_qwen3_genverify500.jsonl",
    },
    "160": {
        "task": "MT",
        "aug": "Qwen3-Gen+Verify",
        "data_path": "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl",
    },
}


TASK_TO_PROMPT_KEY = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}


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

    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "",
    )

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


def parse_multitask_output(raw_output: str):
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
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

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


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_user(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]

    raise ValueError("No user message found in example")


def get_out(run_id: str, task: str) -> Path:
    if task == "MT":
        return OUT_DIR / f"run{run_id}_mt_predictions.csv"

    return OUT_DIR / f"run{run_id}_{task.lower()}_predictions.csv"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        required=True,
        choices=sorted(RUN_CONFIGS.keys()),
        help="Run ID to execute: 131-135 or 156-160.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if prediction CSV already exists.",
    )

    args = parser.parse_args()

    run_id = args.run_id
    config = RUN_CONFIGS[run_id]

    task = config["task"]
    aug = config["aug"]
    data_path = Path(config["data_path"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = get_out(run_id, task)

    if SKIP_COMPLETED and out_path.exists() and not args.force:
        print(f"✅ Skipping Run {run_id} ({task}) — already done")
        print(f"Existing file: {out_path}")
        print("Use --force if you want to rerun.")
        return

    with track_emissions(f"dataaug-verify-llama8b-zeroshot-{task.lower()}"):
        with PROMPTS_PATH.open("r", encoding="utf-8") as f:
            prompts = json.load(f)

        single_prompts = prompts["single_task_zero_shot"]["prompts"]
        multi_prompt = prompts["multitask_training"]["prompt"]
        retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
        retry_multi = prompts["retry_prompts"]["prompts"]["multitask"]

        print(f"\n🚀 Run {run_id}")
        print(f"Model: LLaMA-3.1-8B")
        print(f"Method: ZeroShot")
        print(f"Task: {task}")
        print(f"Aug: {aug}")
        print(f"Input file: {data_path}")
        print(f"Output file: {out_path}")

        data = load_jsonl(data_path)

        print(f"Total examples: {len(data)}")

        model, tokenizer = load_model()

        preds = []
        unknowns = 0
        start = time.time()

        for i, ex in enumerate(data):
            user = extract_user(ex)

            if task == "MT":
                raw = run_inference(model, tokenizer, multi_prompt, user)
                parsed = parse_multitask_output(raw)

                if UNKNOWN_LABEL in parsed.values():
                    raw2 = run_inference(model, tokenizer, retry_multi, user)
                    parsed2 = parse_multitask_output(raw2)

                    for k in parsed:
                        if parsed[k] == UNKNOWN_LABEL:
                            parsed[k] = parsed2[k]

                    raw = raw2

                if UNKNOWN_LABEL in parsed.values():
                    unknowns += 1

                preds.append(
                    {
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
                    raw2 = run_inference(model, tokenizer, retry_single, user)
                    label = recover_single_label(raw2)
                    raw = raw2

                if label == UNKNOWN_LABEL:
                    unknowns += 1

                preds.append(
                    {
                        "pred_label": label,
                        "raw_output": raw,
                    }
                )

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start

                print(
                    f"  Progress: {i + 1}/{len(data)} | "
                    f"Unknowns: {unknowns} | "
                    f"Time: {elapsed / 60:.1f} min"
                )

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=preds[0].keys())
            writer.writeheader()
            writer.writerows(preds)

        total_elapsed = time.time() - start

        print(f"\n✅ Saved → {out_path}")
        print(f"Unknowns: {unknowns}/{len(preds)}")
        print(f"Run time: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()