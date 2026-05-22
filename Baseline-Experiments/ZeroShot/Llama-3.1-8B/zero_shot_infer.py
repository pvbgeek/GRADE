#!/usr/bin/env python3

import argparse
import json
import re
import csv
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


MODEL_PATH   = "/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct"
PROMPTS_PATH = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json")
VAL_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val")
OUT_DIR      = Path("/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Baseline-Experiments/LLaMA-ZeroShot")

RUNS = [
    {"run_id": "001", "task": "MI",  "val_file": "mistake_identification_val.jsonl"},
    {"run_id": "002", "task": "ML",  "val_file": "mistake_location_val.jsonl"},
    {"run_id": "003", "task": "PG",  "val_file": "providing_guidance_val.jsonl"},
    {"run_id": "004", "task": "Act", "val_file": "actionability_val.jsonl"},
    {"run_id": "005", "task": "MT",  "val_file": "multitask_val.jsonl"},
]

TASK_TO_PROMPT_KEY = {
    "MI":  "Mistake_Identification",
    "ML":  "Mistake_Location",
    "PG":  "Providing_Guidance",
    "Act": "Actionability",
}

UNKNOWN_LABEL = "Unknown"
MAX_NEW_TOKENS = 64
SKIP_COMPLETED = True


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

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
        choices=[r["run_id"] for r in RUNS],
        help="Run only the specified run-id (e.g. 005 = MT). Omit to run all 5.",
    )
    args = parser.parse_args()
    selected_runs = [r for r in RUNS if r["run_id"] == args.run_id] if args.run_id else RUNS
    project_name = (
        f"baseline-llama8b-zeroshot-run{args.run_id}"
        if args.run_id
        else "baseline-llama8b-zeroshot"
    )

    with track_emissions(project_name):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        with PROMPTS_PATH.open("r", encoding="utf-8") as f:
            prompts = json.load(f)

        single_prompts = prompts["single_task_zero_shot"]["prompts"]
        multi_prompt   = prompts["multitask_training"]["prompt"]
        retry_single   = prompts["retry_prompts"]["prompts"]["single_task"]
        retry_multi    = prompts["retry_prompts"]["prompts"]["multitask"]

        model, tokenizer = load_model()

        for run in selected_runs:
            run_id = run["run_id"]
            task = run["task"]
            out_path = get_out(run_id, task)

            if SKIP_COMPLETED and out_path.exists():
                print(f"✅ Skipping Run {run_id} ({task}) — already done")
                continue

            print(f"\n🚀 Run {run_id} | Task: {task}")

            data = load_jsonl(VAL_DIR / run["val_file"])
            print(f"Total examples: {len(data)}")

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

                    preds.append({
                        "pred_mi": parsed["pred_mi"],
                        "pred_ml": parsed["pred_ml"],
                        "pred_pg": parsed["pred_pg"],
                        "pred_act": parsed["pred_act"],
                        "raw_output": raw,
                    })

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

                    preds.append({
                        "pred_label": label,
                        "raw_output": raw,
                    })

                if (i + 1) % 25 == 0:
                    elapsed = time.time() - start
                    print(
                        f"  Progress: {i+1}/{len(data)} | Unknowns: {unknowns} | "
                        f"Time: {elapsed/60:.1f} min"
                    )

            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=preds[0].keys())
                writer.writeheader()
                writer.writerows(preds)

            total_elapsed = time.time() - start
            print(f"✅ Saved → {out_path}")
            print(f"Unknowns: {unknowns}/{len(preds)}")
            print(f"Run time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()


















# import argparse
# import csv
# import json
# import os
# import re
# from pathlib import Path

# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer


# ALL_LABELS = [
#     "Mistake_Identification",
#     "Mistake_Location",
#     "Providing_Guidance",
#     "Actionability",
#     "Multitask",
# ]

# DIMS = ALL_LABELS[:4]  # dimensions evaluated inside Multitask
# LABELS = ("Yes", "No", "To some extent")


# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f if line.strip()]


# def parse_single(text):
#     m = re.search(r"Evaluation\s*:\s*(Yes|No|To some extent)", text)
#     if m:
#         return m.group(1)
#     for label in ("To some extent", "Yes", "No"):
#         if re.search(rf"\b{re.escape(label)}\b", text):
#             return label
#     return "Unknown"


# def parse_multitask(text):
#     out = {}
#     for dim in DIMS:
#         m = re.search(rf"{dim}\s*:\s*(Yes|No|To some extent)", text)
#         out[dim] = m.group(1) if m else "Unknown"
#     return out


# def build_retry_messages(original_messages, retry_prompt, prior_output):
#     return original_messages + [
#         {"role": "assistant", "content": prior_output},
#         {"role": "user", "content": retry_prompt},
#     ]


# @torch.inference_mode()
# def generate(model, tokenizer, messages, max_new_tokens):
#     # Drop the gold assistant turn if present; keep system+user only.
#     inference_msgs = [m for m in messages if m["role"] != "assistant"]
#     inputs = tokenizer.apply_chat_template(
#         inference_msgs,
#         add_generation_prompt=True,
#         return_tensors="pt",
#     ).to(model.device)
#     out = model.generate(
#         inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         temperature=1.0,
#         top_p=1.0,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     return tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True).strip()


# def run_task(model, tokenizer, task, data, max_new_tokens, retry_prompt):
#     is_multi = task == "Multitask"
#     results = []
#     for ex in tqdm(data, desc=task):
#         out = generate(model, tokenizer, ex["messages"], max_new_tokens)
#         pred = parse_multitask(out) if is_multi else parse_single(out)

#         needs_retry = any(v == "Unknown" for v in pred.values()) if is_multi else pred == "Unknown"
#         if needs_retry:
#             retry_msgs = build_retry_messages(
#                 [m for m in ex["messages"] if m["role"] != "assistant"],
#                 retry_prompt,
#                 out,
#             )
#             out2 = generate(model, tokenizer, retry_msgs, max_new_tokens)
#             pred2 = parse_multitask(out2) if is_multi else parse_single(out2)
#             if is_multi:
#                 pred = {k: (pred2[k] if pred[k] == "Unknown" else pred[k]) for k in pred}
#                 pred = {k: ("Yes" if v == "Unknown" else v) for k, v in pred.items()}
#             else:
#                 pred = pred2 if pred2 != "Unknown" else "Yes"  # majority-class fallback
#             out = out + "\n---RETRY---\n" + out2

#         results.append({"raw": out, "pred": pred})
#     return results


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_path", default="/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct")
#     ap.add_argument("--data_dir", required=True, help="path to data/val (chat-format .jsonl)")
#     ap.add_argument("--prompts_json", required=True)
#     ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--tasks", nargs="+", default=list(ALL_LABELS))
#     ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
#     args = ap.parse_args()

#     Path(args.out_dir).mkdir(parents=True, exist_ok=True)
#     with open(args.prompts_json, "r", encoding="utf-8") as f:
#         prompts_cfg = json.load(f)
#     retry_single = prompts_cfg["retry_prompts"]["prompts"]["single_task"]
#     retry_multi = prompts_cfg["retry_prompts"]["prompts"]["multitask"]

#     dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path,
#         torch_dtype=dtype,
#         device_map="auto",
#     )
#     model.eval()

#     for task in args.tasks:
#         if task not in ALL_LABELS:
#             print(f"skipping unknown task {task}")
#             continue
#         data_file = Path(args.data_dir) / f"{task.lower()}_val.jsonl"
#         if not data_file.exists():
#             print(f"missing {data_file}; skipping")
#             continue
#         data = load_jsonl(data_file)
#         is_multi = task == "Multitask"
#         max_new = 64 if is_multi else 16
#         retry_prompt = retry_multi if is_multi else retry_single
#         results = run_task(model, tokenizer, task, data, max_new, retry_prompt)

#         out_path = Path(args.out_dir) / f"{task}_zeroshot_preds.csv"
#         with open(out_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             if is_multi:
#                 writer.writerow(["pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"])
#                 for r in results:
#                     p = r["pred"]
#                     writer.writerow([
#                         p["Mistake_Identification"],
#                         p["Mistake_Location"],
#                         p["Providing_Guidance"],
#                         p["Actionability"],
#                         r["raw"],
#                     ])
#             else:
#                 writer.writerow(["pred_label", "raw_output"])
#                 for r in results:
#                     writer.writerow([r["pred"], r["raw"]])
#         print(f"wrote {out_path}  ({len(results)} predictions)")


# if __name__ == "__main__":
#     main()