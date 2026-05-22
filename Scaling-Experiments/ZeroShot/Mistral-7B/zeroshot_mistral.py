#!/usr/bin/env python3
"""Zero-shot Mistral baseline inference for TutorMind.

Purpose
-------
Render TutorMind chat examples with the zero-shot Mistral-7B prompts and
write evaluator-compatible prediction CSVs for MI, ML, PG, Act, or MT.

Input
-----
- TutorMind JSONL chat data with a ``messages`` list per row.
- ``prompts.json`` as the prompt registry source of truth.

Output
------
- Single-task CSVs with ``pred_label``.
- MT CSVs with ``pred_mi``, ``pred_ml``, ``pred_pg``, and ``pred_act``.

Example
-------
    python '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling Experiments/ZeroShot/Mistral-7B/zeroshot_mistral.py' \\
      --task MI \\
      --input-jsonl '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl' \\
      --predictions-out '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/Scaling Experiments/ZeroShot/Mistral-7B/predictions/run_011_mi_predictions.csv'
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


MODEL_NAME = "Mistral-7B"
BASE_MODEL_PATH = "/WAVE/datasets/oignat_lab/Mistral"
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
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run zero-shot TutorMind inference with the hardcoded Mistral-7B base model."
    )
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
    parser.add_argument("--input-jsonl", required=True, help="TutorMind chat-format JSONL file.")
    parser.add_argument(
        "--predictions-out",
        required=True,
        help="Prediction CSV path to write for the selected task.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def canonicalize_text(text: str) -> str:
    """Collapse mild formatting variation for parsing."""
    stripped = text.strip().strip("\"'`")
    stripped = stripped.strip(" .,!?:;")
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


def normalize_label(value: object) -> str:
    """Normalize a parsed label into the evaluator's canonical label set."""
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


def recover_single_task_label(raw_output: str) -> str:
    """Recover the final single-task label from a raw generation."""
    parsed = normalize_label(raw_output)
    if parsed != UNKNOWN_LABEL:
        return parsed

    # Prefer the last valid label because generations may echo prompt text first.
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


def regex_for_alias(alias: str) -> str:
    """Convert a dimension alias into a tolerant regex fragment."""
    parts = re.split(r"[\s_-]+", alias.strip())
    return r"[\s_-]*".join(re.escape(part) for part in parts if part)


def ensure_existing_file(path_str: str, arg_name: str) -> Path:
    """Resolve a required file path and ensure it exists."""
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{arg_name} does not exist or is not a file: {path}")
    return path.resolve()


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_special_token_content(value: object, default: str | None = None) -> str | None:
    """Extract a tokenizer special token string from HF config metadata."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return default


def load_jsonl_records(path: Path) -> List[dict]:
    """Load JSONL records from disk."""
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
    """Select the configured prompt for the requested task."""
    if task == "MT":
        prompt = prompts["multitask_training"]["prompt"]
    else:
        prompt_key = SINGLE_TASK_PROMPT_KEYS[task]
        prompt = prompts["single_task_zero_shot"]["prompts"][prompt_key]

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt registry did not contain a usable prompt for task {task}.")
    return prompt


def extract_last_user_content(record: dict, source_path: Path, row_index: int) -> str:
    """Return the final user message from a TutorMind chat-format example."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(
            f"{source_path} row {row_index} is missing a valid 'messages' list."
        )

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"{source_path} row {row_index} has a non-string or empty user content."
                )
            return content

    raise ValueError(f"{source_path} row {row_index} has no user message.")


def build_messages(record: dict, prompt: str, source_path: Path, row_index: int) -> List[dict]:
    """Build the inference-time chat messages for one example."""
    user_content = extract_last_user_content(record, source_path, row_index)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def render_prompt_texts(
    tokenizer: AutoTokenizer,
    records: Sequence[dict],
    prompt: str,
    source_path: Path,
    start_index: int,
) -> List[str]:
    """Render inference prompts via the tokenizer's chat template."""
    rendered: List[str] = []
    for offset, record in enumerate(records):
        row_index = start_index + offset
        messages = build_messages(record, prompt, source_path, row_index)
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Best-effort device for feeding generation inputs."""
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
    """Generate one continuation per rendered prompt."""
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
        completion_tokens = outputs[batch_index, int(prompt_length) :]
        decoded.append(
            tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        )
    return decoded


def parse_single_task_output(raw_output: str) -> str:
    """Parse a single-task model output into Yes / No / To some extent / Unknown."""
    return recover_single_task_label(raw_output)


def parse_multitask_output(raw_output: str) -> Dict[str, str]:
    """Parse a multitask model output into the four evaluator dimensions."""
    parsed = {dimension: UNKNOWN_LABEL for dimension in SINGLE_TASKS}

    for raw_line in raw_output.splitlines():
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
            raw_output,
            flags=re.IGNORECASE,
        )
        if match:
            parsed[dimension] = normalize_label(match.group(1))

    return parsed


def iter_batches(records: Sequence[dict], batch_size: int) -> Iterable[tuple[int, Sequence[dict]]]:
    """Yield contiguous record batches with their starting index."""
    for start_index in range(0, len(records), batch_size):
        yield start_index, records[start_index : start_index + batch_size]


def write_predictions_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    """Write prediction rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_prediction_rows(
    task: str,
    raw_outputs: Sequence[str],
    start_index: int,
) -> List[dict]:
    """Build evaluator-compatible prediction rows for one decoded batch."""
    rows: List[dict] = []
    for offset, raw_output in enumerate(raw_outputs):
        row_index = start_index + offset
        if task == "MT":
            parsed = parse_multitask_output(raw_output)
            rows.append(
                {
                    "row_index": row_index,
                    "task": task,
                    "pred_mi": parsed["MI"],
                    "pred_ml": parsed["ML"],
                    "pred_pg": parsed["PG"],
                    "pred_act": parsed["Act"],
                    "raw_output": raw_output,
                }
            )
        else:
            rows.append(
                {
                    "row_index": row_index,
                    "task": task,
                    "pred_label": parse_single_task_output(raw_output),
                    "raw_output": raw_output,
                }
            )
    return rows


def get_output_columns(task: str) -> List[str]:
    """Return the required output columns for the selected task."""
    if task == "MT":
        return [
            "row_index",
            "task",
            "pred_mi",
            "pred_ml",
            "pred_pg",
            "pred_act",
            "raw_output",
        ]
    return ["row_index", "task", "pred_label", "raw_output"]


def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the hardcoded Mistral-7B model and tokenizer."""
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

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def validate_args(args: argparse.Namespace) -> None:
    """Reject obviously invalid generation settings early."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be in the range (0, 1].")


def main() -> None:
    """Run zero-shot inference and write evaluator-compatible predictions."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        input_path = ensure_existing_file(args.input_jsonl, "--input-jsonl")
        prompts_path = ensure_existing_file(PROMPTS_JSON, "PROMPTS_JSON")
        predictions_out = Path(args.predictions_out).resolve()

        prompts = load_json(prompts_path)
        prompt = select_prompt(prompts, args.task)
        records = load_jsonl_records(input_path)

        model, tokenizer = load_model_and_tokenizer()

        all_rows: List[dict] = []
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
            all_rows.extend(build_prediction_rows(args.task, raw_outputs, start_index))

        output_columns = get_output_columns(args.task)
        write_predictions_csv(predictions_out, output_columns, all_rows)
        print(
            f"Wrote {len(all_rows)} {args.task} prediction row(s) to {predictions_out} "
            f"using model {MODEL_NAME}."
        )

    except ValueError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
