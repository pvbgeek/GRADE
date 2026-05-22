#!/usr/bin/env python3

from __future__ import annotations

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
from peft import PeftModel

from utils.codecarbon_helper import track_emissions


BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"

UNKNOWN = "Unknown"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser


def load_jsonl(path):
    return [json.loads(x) for x in open(path)]


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def normalize(text):
    text = text.lower()
    if "to some extent" in text:
        return "To some extent"
    if "yes" in text:
        return "Yes"
    if "no" in text:
        return "No"
    return UNKNOWN


def main():
    args = build_parser().parse_args()

    with track_emissions(f"cot-qwen14b-lora-infer-{args.task.lower()}"):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(base_model, args.adapter_path)

        data = load_jsonl(args.input_jsonl)

        outputs = []
        start = time.time()

        for i, row in enumerate(data):
            user = row["messages"][-2]["content"]

            messages = [{"role": "user", "content": user}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            parsed = normalize(strip_think(decoded))

            outputs.append({
                "row_index": i,
                "task": args.task,
                "pred_label": parsed,
                "raw_output": decoded
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                avg = elapsed / (i + 1)
                eta = (len(data) - (i + 1)) * avg / 60

                print(f"{i+1}/{len(data)} | Avg {avg:.2f}s | ETA {eta:.1f} min")

        with open(args.predictions_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=outputs[0].keys())
            writer.writeheader()
            writer.writerows(outputs)

        print(f"Saved → {args.predictions_out}")


if __name__ == "__main__":
    main()