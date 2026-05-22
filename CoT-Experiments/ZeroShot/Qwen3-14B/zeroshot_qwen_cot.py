#!/usr/bin/env python3
"""Zero-shot Qwen3-14B inference for TutorMind with thinking mode ON.

Purpose
-------
Render TutorMind chat examples with the Qwen3-14B chain-of-thought (thinking)
prompts and write evaluator-compatible prediction CSVs for MI, ML, PG, Act, or
MT. Runs with Qwen3 thinking mode ON; includes one retry pass for Unknowns.

Used by Experiment 2 — Thinking Mode (Runs 051-055).

Input
-----
- TutorMind JSONL chat data with a ``messages`` list per row.
- ``prompts.json`` as the prompt registry source of truth.

Output
------
- Single-task CSVs with ``pred_label``.
- MT CSVs with ``pred_mi``, ``pred_ml``, ``pred_pg``, and ``pred_act``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import sys as _sys
from pathlib import Path as _Path
for _candidate in _Path(__file__).resolve().parents:
    if (_candidate / "utils" / "codecarbon_helper.py").is_file():
        if str(_candidate) not in _sys.path:
            _sys.path.insert(0, str(_candidate))
        break

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast

from utils.codecarbon_helper import track_emissions


MODEL_NAME = "Qwen3-14B"
BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/QWEN3"
PROMPTS_JSON = "/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/prompts.json"
UNKNOWN_LABEL = "Unknown"
SINGLE_TASKS = ["MI", "ML", "PG", "Act"]
ALL_TASKS = SINGLE_TASKS + ["MT"]

SINGLE_TASK_PROMPT_KEYS = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "Act": "Actionability",
}

MT_FIELD_ALIASES = {
    "mistakeidentification": "MI",
    "mistakelocation": "ML",
    "providingguidance": "PG",
    "actionability": "Act",
    "mi": "MI",
    "ml": "ML",
    "pg": "PG",
    "act": "Act",
}

MT_REGEX_ALIASES = {
    "MI": ["Mistake_Identification", "Mistake Identification", "MI"],
    "ML": ["Mistake_Location", "Mistake Location", "ML"],
    "PG": ["Providing_Guidance", "Providing Guidance", "PG"],
    "Act": ["Actionability", "Act"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run zero-shot TutorMind inference with Qwen3-14B in thinking mode."
    )
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
    parser.add_argument("--input-jsonl", required=True, help="TutorMind chat-format JSONL file.")
    parser.add_argument(
        "--predictions-out",
        required=True,
        help="Prediction CSV path to write for the selected task.",
    )
    # Thinking mode emits long <think>...</think> traces before the final answer,
    # so the budget here is much larger than the no-think variant (default 128).
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit to lower VRAM usage.",
    )
    return parser


def canonicalize_text(text: str) -> str:
    stripped = text.strip().strip("\"'`")
    stripped = stripped.strip(" .,!?:;")
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


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


def strip_think_blocks(raw_output: str) -> str:
    # Thinking mode wraps reasoning in <think>...</think>. Strip them so label
    # parsing only sees the final answer; the original text is still saved to
    # raw_output for later inspection.
    if not raw_output:
        return raw_output
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def recover_single_task_label(raw_output: str) -> str:
    cleaned = strip_think_blocks(raw_output)
    parsed = normalize_label(cleaned)
    if parsed != UNKNOWN_LABEL:
        return parsed

    evaluation_matches = list(
        re.finditer(
            r"evaluation\s*:\s*(yes|no|to some extent)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    if evaluation_matches:
        return normalize_label(evaluation_matches[-1].group(1))

    trailing_match = re.search(
        r"(yes|no|to some extent)\s*$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if trailing_match:
        return normalize_label(trailing_match.group(1))

    return UNKNOWN_LABEL


def regex_for_alias(alias: str) -> str:
    parts = re.split(r"[\s_-]+", alias.strip())
    return r"[\s_-]*".join(re.escape(part) for part in parts if part)


def ensure_existing_file(path_str: str, arg_name: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{arg_name} does not exist or is not a file: {path}")
    return path.resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_special_token_content(value: object, default: str | None = None) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return default


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def select_prompt(prompts: dict, task: str) -> str:
    if task == "MT":
        prompt = prompts["multitask_thinking"]["prompt"]
    else:
        prompt_key = SINGLE_TASK_PROMPT_KEYS[task]
        prompt = prompts["single_task_thinking"]["prompts"][prompt_key]

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt registry did not contain a usable prompt for task {task}.")
    return prompt


def extract_last_user_content(record: dict, source_path: Path, row_index: int) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_path} row {row_index} is missing a valid 'messages' list.")

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{source_path} row {row_index} has a non-string or empty user content.")
            return content

    raise ValueError(f"{source_path} row {row_index} has no user message.")


def build_messages(record: dict, prompt: str, source_path: Path, row_index: int) -> List[dict]:
    user_content = extract_last_user_content(record, source_path, row_index)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_retry_messages(original_messages: List[dict], retry_prompt: str, prior_output: str) -> List[dict]:
    return original_messages + [
        {"role": "assistant", "content": prior_output},
        {"role": "user", "content": retry_prompt},
    ]


def apply_chat_template_think(tokenizer: AutoTokenizer, messages, add_generation_prompt: bool) -> str:
    # Qwen3 chat template accepts enable_thinking; fall back to /think hint if
    # the template variant does not expose the kwarg.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True,
        )
    except TypeError:
        patched = list(messages)
        for i, message in enumerate(patched):
            if message.get("role") == "system":
                patched[i] = {
                    "role": "system",
                    "content": f"{message['content']}\n\n/think",
                }
                break
        else:
            patched.insert(0, {"role": "system", "content": "/think"})
        return tokenizer.apply_chat_template(
            patched,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def render_prompt_texts(
    tokenizer: AutoTokenizer,
    records: Sequence[dict],
    prompt: str,
    source_path: Path,
    start_index: int,
) -> List[str]:
    rendered: List[str] = []
    for offset, record in enumerate(records):
        row_index = start_index + offset
        messages = build_messages(record, prompt, source_path, row_index)
        rendered.append(
            apply_chat_template_think(tokenizer, messages, add_generation_prompt=True)
        )
    return rendered


def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    encoded = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_device = get_model_input_device(model)
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

    generation_kwargs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**generation_kwargs)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    decoded: List[str] = []
    for batch_index, prompt_length in enumerate(prompt_lengths):
        completion_tokens = outputs[batch_index, int(prompt_length):]
        decoded.append(tokenizer.decode(completion_tokens, skip_special_tokens=True).strip())
    return decoded


def parse_single_task_output(raw_output: str) -> str:
    return recover_single_task_label(raw_output)


def parse_multitask_output(raw_output: str) -> Dict[str, str]:
    cleaned = strip_think_blocks(raw_output)
    parsed = {dimension: UNKNOWN_LABEL for dimension in SINGLE_TASKS}

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        field_name, raw_value = line.split(":", 1)
        field_key = canonicalize_text(field_name).replace(" ", "")
        dimension = MT_FIELD_ALIASES.get(field_key)
        if dimension and parsed[dimension] == UNKNOWN_LABEL:
            parsed[dimension] = normalize_label(raw_value)

    for dimension in SINGLE_TASKS:
        if parsed[dimension] != UNKNOWN_LABEL:
            continue
        alias_regex = "|".join(regex_for_alias(alias) for alias in MT_REGEX_ALIASES[dimension])
        match = re.search(
            rf"(?:{alias_regex})\s*:\s*(Yes|No|To some extent)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            parsed[dimension] = normalize_label(match.group(1))

    return parsed


def iter_batches(records: Sequence[dict], batch_size: int) -> Iterable[tuple[int, Sequence[dict]]]:
    for start_index in range(0, len(records), batch_size):
        yield start_index, records[start_index:start_index + batch_size]


def write_predictions_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_output_columns(task: str) -> List[str]:
    if task == "MT":
        return ["row_index", "task", "pred_mi", "pred_ml", "pred_pg", "pred_act", "raw_output"]
    return ["row_index", "task", "pred_label", "raw_output"]


def load_model_and_tokenizer(load_in_4bit: bool) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    except (AttributeError, TypeError, ValueError):
        base_model_path = Path(BASE_MODEL_PATH)
        tokenizer_config = load_json(base_model_path / "tokenizer_config.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(base_model_path / "tokenizer.json"),
            bos_token=get_special_token_content(tokenizer_config.get("bos_token"), "<s>"),
            eos_token=get_special_token_content(tokenizer_config.get("eos_token"), "</s>"),
            unk_token=get_special_token_content(tokenizer_config.get("unk_token"), "<unk>"),
            pad_token=get_special_token_content(tokenizer_config.get("pad_token"), "<pad>"),
        )
        tokenizer.chat_template = tokenizer_config.get("chat_template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be in the range (0, 1].")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        input_path = ensure_existing_file(args.input_jsonl, "--input-jsonl")
        prompts_path = ensure_existing_file(PROMPTS_JSON, "PROMPTS_JSON")
        predictions_out = Path(args.predictions_out).resolve()
    except ValueError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
        return

    with track_emissions(f"cot-qwen14b-zeroshot-{args.task.lower()}"):
        prompts = load_json(prompts_path)
        prompt = select_prompt(prompts, args.task)
        retry_single = prompts["retry_prompts"]["prompts"]["single_task"]
        retry_multi = prompts["retry_prompts"]["prompts"]["multitask"]

        records = load_jsonl_records(input_path)
        model, tokenizer = load_model_and_tokenizer(args.load_in_4bit)

        output_columns = get_output_columns(args.task)
        all_rows: List[dict] = []
        unknown_count = 0
        start_time = time.time()

        for start_index, batch_records in iter_batches(records, args.batch_size):
            prompt_texts = render_prompt_texts(
                tokenizer=tokenizer,
                records=batch_records,
                prompt=prompt,
                source_path=input_path,
                start_index=start_index,
            )
            raw_outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            batch_rows: List[dict] = []
            for offset, raw_output in enumerate(raw_outputs):
                row_index = start_index + offset
                record = batch_records[offset]
                base_messages = build_messages(record, prompt, input_path, row_index)

                if args.task == "MT":
                    parsed = parse_multitask_output(raw_output)
                    if UNKNOWN_LABEL in parsed.values():
                        retry_messages = build_retry_messages(base_messages, retry_multi, raw_output)
                        retry_prompt_text = apply_chat_template_think(
                            tokenizer, retry_messages, add_generation_prompt=True
                        )
                        retry_output = generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=[retry_prompt_text],
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )[0]
                        retry_parsed = parse_multitask_output(retry_output)
                        for dim in parsed:
                            if parsed[dim] == UNKNOWN_LABEL:
                                parsed[dim] = retry_parsed[dim]
                        raw_output = retry_output

                    if UNKNOWN_LABEL in parsed.values():
                        unknown_count += 1

                    batch_rows.append(
                        {
                            "row_index": row_index,
                            "task": args.task,
                            "pred_mi": parsed["MI"],
                            "pred_ml": parsed["ML"],
                            "pred_pg": parsed["PG"],
                            "pred_act": parsed["Act"],
                            "raw_output": raw_output,
                        }
                    )
                else:
                    parsed = parse_single_task_output(raw_output)
                    if parsed == UNKNOWN_LABEL:
                        retry_messages = build_retry_messages(base_messages, retry_single, raw_output)
                        retry_prompt_text = apply_chat_template_think(
                            tokenizer, retry_messages, add_generation_prompt=True
                        )
                        retry_output = generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=[retry_prompt_text],
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )[0]
                        retry_parsed = parse_single_task_output(retry_output)
                        if retry_parsed != UNKNOWN_LABEL:
                            parsed = retry_parsed
                        raw_output = retry_output

                    if parsed == UNKNOWN_LABEL:
                        unknown_count += 1

                    batch_rows.append(
                        {
                            "row_index": row_index,
                            "task": args.task,
                            "pred_label": parsed,
                            "raw_output": raw_output,
                        }
                    )

            all_rows.extend(batch_rows)

            if len(all_rows) % 25 == 0 or len(all_rows) == len(records):
                elapsed = time.time() - start_time
                print(
                    f"Progress: {len(all_rows)}/{len(records)} | "
                    f"Unknowns: {unknown_count} | Time: {elapsed/60:.1f} min"
                )

        write_predictions_csv(predictions_out, output_columns, all_rows)
        print(
            f"Wrote {len(all_rows)} {args.task} prediction row(s) to {predictions_out} "
            f"using model {MODEL_NAME} (Think: ON)."
        )


if __name__ == "__main__":
    main()
