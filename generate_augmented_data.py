#!/usr/bin/env python3
"""Generate augmented TutorMind training data with local Qwen3-14B.

This script only reads from the training split and writes augmented training
files plus sidecar logs under `data/train/Gen` and `data/train/Gen+Verify`.
It preserves the original chat-format JSONL schema exactly by copying each
source row and replacing only:

1. `messages[1].content`:
   The trailing `Tutor Response: ...` segment is rewritten to use the newly
   generated tutor response instead of the original evaluated response.
2. `messages[2].content`:
   The assistant label is rewritten to the intended target label in the exact
   format used by the original train file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - dry-run still works without torch
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
except ImportError:  # pragma: no cover - dry-run still works without transformers
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PreTrainedTokenizerFast = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional
    BitsAndBytesConfig = None


ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "data" / "train"
PROMPTS_PATH = ROOT / "prompts.json"
MODEL_PATH = Path("/WAVE/datasets/oignat_lab/QWEN3")
GEN_DIR = TRAIN_DIR / "Gen"
GENVERIFY_DIR = TRAIN_DIR / "Gen+Verify"

MINORITY_LABELS = ("No", "To some extent")
ALL_SINGLE_LABELS = ("Yes", "No", "To some extent")
UNKNOWN_LABEL = "Unknown"
TASK_ORDER = ("MI", "ML", "PG", "Act", "MT")
MT_DIMENSIONS = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)
MT_FIELD_ALIASES = {
    "mistakeidentification": "Mistake_Identification",
    "mistakelocation": "Mistake_Location",
    "providingguidance": "Providing_Guidance",
    "actionability": "Actionability",
}


def configure_slurm_streams() -> None:
    """Keep progress visible in Slurm output files while long jobs run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True, write_through=True)


@dataclass(frozen=True)
class TaskSpec:
    alias: str
    prompt_key: str
    train_filename: str
    gen_output_filename: str
    genverify_output_filename: str
    is_multitask: bool = False


@dataclass
class PreparedSourceRow:
    source_file: Path
    source_row_index: int
    row: dict[str, Any]
    system_prompt: str
    user_content: str
    assistant_content: str
    conversation_prefix: str
    tutor_response_marker: str
    original_tutor_response: str
    conversation_hash: str
    original_tutor_response_words: int
    original_tutor_response_chars: int


@dataclass
class ResponseLengthStats:
    mean_words: float
    median_words: float
    p95_words: int
    max_words: int
    max_allowed_words: int
    mean_chars: float
    median_chars: float
    p95_chars: int
    max_chars: int
    max_allowed_chars: int


@dataclass
class SchemaSummary:
    spec: TaskSpec
    train_path: Path
    rows: list[dict[str, Any]]
    prepared_rows: list[PreparedSourceRow]
    label_counts: Counter[str]
    message_lengths: Counter[int]
    role_patterns: Counter[tuple[str, ...]]
    sample_row: dict[str, Any]
    response_length_stats: ResponseLengthStats
    top_level_keys: tuple[str, ...]


TASK_SPECS: dict[str, TaskSpec] = {
    "MI": TaskSpec(
        alias="MI",
        prompt_key="Mistake_Identification",
        train_filename="mistake_identification_train.jsonl",
        gen_output_filename="mistake_identification_train_aug_qwen3_gen500.jsonl",
        genverify_output_filename="mistake_identification_train_aug_qwen3_genverify500.jsonl",
    ),
    "ML": TaskSpec(
        alias="ML",
        prompt_key="Mistake_Location",
        train_filename="mistake_location_train.jsonl",
        gen_output_filename="mistake_location_train_aug_qwen3_gen500.jsonl",
        genverify_output_filename="mistake_location_train_aug_qwen3_genverify500.jsonl",
    ),
    "PG": TaskSpec(
        alias="PG",
        prompt_key="Providing_Guidance",
        train_filename="providing_guidance_train.jsonl",
        gen_output_filename="providing_guidance_train_aug_qwen3_gen500.jsonl",
        genverify_output_filename="providing_guidance_train_aug_qwen3_genverify500.jsonl",
    ),
    "Act": TaskSpec(
        alias="Act",
        prompt_key="Actionability",
        train_filename="actionability_train.jsonl",
        gen_output_filename="actionability_train_aug_qwen3_gen500.jsonl",
        genverify_output_filename="actionability_train_aug_qwen3_genverify500.jsonl",
    ),
    "MT": TaskSpec(
        alias="MT",
        prompt_key="Multitask",
        train_filename="multitask_train.jsonl",
        gen_output_filename="multitask_train_aug_qwen3_gen500.jsonl",
        genverify_output_filename="multitask_train_aug_qwen3_genverify500.jsonl",
        is_multitask=True,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate schema-preserving TutorMind augmented train data with local Qwen3-14B."
    )
    parser.add_argument(
        "--mode",
        choices=("gen", "genverify", "both"),
        default="both",
        help="Which augmentation variant(s) to produce.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_ORDER,
        default=list(TASK_ORDER),
        help="Subset of TutorMind tasks to process.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=list(MINORITY_LABELS),
        default=list(MINORITY_LABELS),
        help="Minority labels to synthesize for the selected tasks.",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=500,
        help="Number of accepted synthetic rows required per selected label.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for augmentation generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value for augmentation generation.",
    )
    parser.add_argument(
        "--verification-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for self-verification decoding.",
    )
    parser.add_argument(
        "--verification-top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value for self-verification decoding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for augmentation generation.",
    )
    parser.add_argument(
        "--verification-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for self-verification calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source-row sampling.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=15,
        help="Maximum attempts per task/label = target_per_label * multiplier.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect schema and planned outputs only; do not load Qwen or write files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing accepted logs for the selected mode/task/label combinations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output/log files for the selected mode/task/label combinations before running.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing selected output files. Without --resume/--overwrite, validation-only mode is used.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow writing outputs even if the exact target count could not be reached.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DIR,
        help="Training directory containing only train JSONL files.",
    )
    parser.add_argument(
        "--out-gen-dir",
        type=Path,
        default=None,
        help="Output directory for gen mode. Defaults to <train_dir>/Gen.",
    )
    parser.add_argument(
        "--out-genverify-dir",
        type=Path,
        default=None,
        help="Output directory for genverify mode. Defaults to <train_dir>/Gen+Verify.",
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=PROMPTS_PATH,
        help="Prompt registry JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Local Qwen3-14B model path.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Optional 4-bit load for constrained GPU memory environments.",
    )
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def normalize_label(value: str) -> str:
    cleaned = canonicalize_text(value)
    if cleaned == "yes":
        return "Yes"
    if cleaned == "no":
        return "No"
    if cleaned == "to some extent":
        return "To some extent"
    return UNKNOWN_LABEL


def normalize_duplicate_text(value: str) -> str:
    cleaned, _ = clean_qwen_thinking_output(value)
    cleaned = cleaned.casefold()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def json_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def percentile_int(values: Sequence[int], pct: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return int(ordered[index])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["gen", "genverify"]
    return [mode]


def output_dir_for_mode(args: argparse.Namespace, mode: str) -> Path:
    return args.out_gen_dir if mode == "gen" else args.out_genverify_dir


def output_file_for_mode(args: argparse.Namespace, mode: str, spec: TaskSpec) -> Path:
    filename = spec.gen_output_filename if mode == "gen" else spec.genverify_output_filename
    return output_dir_for_mode(args, mode) / filename


def logs_dir_for_mode(args: argparse.Namespace, mode: str) -> Path:
    return output_dir_for_mode(args, mode) / "logs"


def slugify_label(label: str) -> str:
    return label.casefold().replace(" ", "_")


def task_log_prefix(spec: TaskSpec) -> str:
    return spec.alias.casefold()


def label_log_paths(args: argparse.Namespace, mode: str, spec: TaskSpec, label: str) -> dict[str, Path]:
    base = logs_dir_for_mode(args, mode)
    label_slug = slugify_label(label)
    task_slug = task_log_prefix(spec)
    return {
        "candidates": base / f"candidates_{task_slug}_{label_slug}.jsonl",
        "accepted": base / f"accepted_{task_slug}_{label_slug}.jsonl",
        "rejected": base / f"rejected_{task_slug}_{label_slug}.jsonl",
    }


def manifest_path(args: argparse.Namespace, mode: str, spec: TaskSpec) -> Path:
    return logs_dir_for_mode(args, mode) / f"manifest_{task_log_prefix(spec)}.json"


def summary_all_path(args: argparse.Namespace, mode: str) -> Path:
    return logs_dir_for_mode(args, mode) / "summary_all.json"


def split_user_content_on_tutor_response(user_content: str) -> tuple[str, str, str]:
    matches = list(re.finditer(r"tutor response:\s*", user_content, flags=re.IGNORECASE))
    if not matches:
        raise ValueError("Could not locate 'Tutor Response:' marker in user content.")
    match = matches[-1]
    prefix = user_content[: match.start()]
    marker = user_content[match.start() : match.end()]
    response = user_content[match.end() :]
    if not response.strip():
        raise ValueError("Tutor Response marker was found, but no evaluated tutor response followed it.")
    return prefix, marker, response.strip()


def extract_messages(row: dict[str, Any], path: Path, row_index: int) -> tuple[list[dict[str, Any]], str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path} row {row_index} is missing a list-valued 'messages' field.")
    if len(messages) != 3:
        raise ValueError(f"{path} row {row_index} expected 3 messages but found {len(messages)}.")
    roles = tuple(message.get("role") for message in messages if isinstance(message, dict))
    if roles != ("system", "user", "assistant"):
        raise ValueError(f"{path} row {row_index} roles were {roles}, expected ('system', 'user', 'assistant').")
    system_prompt = messages[0].get("content")
    user_content = messages[1].get("content")
    assistant_content = messages[2].get("content")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(f"{path} row {row_index} has invalid system content.")
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError(f"{path} row {row_index} has invalid user content.")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise ValueError(f"{path} row {row_index} has invalid assistant content.")
    return messages, system_prompt, user_content, assistant_content


def clean_qwen_thinking_output(raw_text: str) -> tuple[str, str | None]:
    if not raw_text:
        return "", "empty_after_think_strip"

    cleaned = raw_text
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1]

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    cleaned = re.sub(
        r"^\s*(generated\s+tutor\s+response|tutor\s+response|response)\s*:\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip().strip("\"' ").strip()

    lowered = cleaned.casefold()
    if "<think>" in lowered and "</think>" not in lowered:
        return cleaned, "unclosed_think_block"
    if "<think>" in lowered or "</think>" in lowered:
        return cleaned, "residual_think_marker"
    if not cleaned:
        return "", "empty_after_think_strip"
    return cleaned, None


def strip_think_blocks(raw_output: str) -> str:
    cleaned, _ = clean_qwen_thinking_output(raw_output)
    return cleaned


def recover_evaluation_label(cleaned_output: str) -> str:
    cleaned = cleaned_output.strip()
    normalized_direct = normalize_label(cleaned)
    if normalized_direct != UNKNOWN_LABEL:
        return normalized_direct

    matches = list(
        re.finditer(
            r"evaluation\s*:\s*(yes|no|to some extent)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    if matches:
        return normalize_label(matches[-1].group(1))

    trailing = re.search(r"(yes|no|to some extent)\s*$", cleaned, flags=re.IGNORECASE)
    if trailing:
        return normalize_label(trailing.group(1))
    return UNKNOWN_LABEL


def parse_mt_label_block(cleaned_output: str) -> dict[str, str]:
    cleaned = cleaned_output.strip()
    predictions = {dimension: UNKNOWN_LABEL for dimension in MT_DIMENSIONS}
    for line in cleaned.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        compact_key = canonicalize_text(key).replace(" ", "").replace("_", "")
        dimension = MT_FIELD_ALIASES.get(compact_key)
        if dimension:
            predictions[dimension] = normalize_label(value)
    return predictions


def format_mt_label_vector(uniform_label: str) -> str:
    if uniform_label not in MINORITY_LABELS:
        raise ValueError(f"Unexpected multitask minority label: {uniform_label}")
    return "\n".join(f"{dimension}: {uniform_label}" for dimension in MT_DIMENSIONS)


def mt_expected_vector(uniform_label: str) -> dict[str, str]:
    return {dimension: uniform_label for dimension in MT_DIMENSIONS}


def choose_torch_dtype() -> Any:
    if torch is None:
        return None
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def snapshot_directory_metadata(path: Path) -> dict[str, dict[str, int]]:
    snapshot: dict[str, dict[str, int]] = {}
    if not path.exists():
        return snapshot
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = file_path.stat()
        snapshot[str(file_path.resolve())] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def assert_snapshot_unchanged(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]], label: str) -> None:
    if before != after:
        raise RuntimeError(f"{label} files changed during this run; refusing to continue.")


def build_response_length_stats(word_lengths: Sequence[int], char_lengths: Sequence[int]) -> ResponseLengthStats:
    return ResponseLengthStats(
        mean_words=statistics.mean(word_lengths),
        median_words=statistics.median(word_lengths),
        p95_words=percentile_int(word_lengths, 95.0),
        max_words=max(word_lengths),
        max_allowed_words=max(24, int(math.ceil(percentile_int(word_lengths, 95.0) * 1.6))),
        mean_chars=statistics.mean(char_lengths),
        median_chars=statistics.median(char_lengths),
        p95_chars=percentile_int(char_lengths, 95.0),
        max_chars=max(char_lengths),
        max_allowed_chars=max(180, int(math.ceil(percentile_int(char_lengths, 95.0) * 1.6))),
    )


def sample_row_preview(row: dict[str, Any], limit: int = 1200) -> str:
    rendered = json.dumps(row, indent=2, ensure_ascii=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit].rstrip() + "\n... [truncated]"


def inspect_train_schema(train_dir: Path) -> dict[str, SchemaSummary]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    detected_files = sorted(path for path in train_dir.glob("*.jsonl") if path.is_file())
    print("Detected train files:")
    for path in detected_files:
        print(f"  - {path}")
    print()

    summaries: dict[str, SchemaSummary] = {}
    for alias in TASK_ORDER:
        spec = TASK_SPECS[alias]
        train_path = train_dir / spec.train_filename
        if not train_path.exists():
            raise FileNotFoundError(f"Expected train file missing for task {alias}: {train_path}")
        rows = load_jsonl(train_path)
        if not rows:
            raise ValueError(f"Train file is empty: {train_path}")

        prepared_rows: list[PreparedSourceRow] = []
        label_counts: Counter[str] = Counter()
        message_lengths: Counter[int] = Counter()
        role_patterns: Counter[tuple[str, ...]] = Counter()
        word_lengths: list[int] = []
        char_lengths: list[int] = []

        for row_index, row in enumerate(rows):
            messages, system_prompt, user_content, assistant_content = extract_messages(row, train_path, row_index)
            label_counts[assistant_content] += 1
            message_lengths[len(messages)] += 1
            role_patterns[tuple(message["role"] for message in messages)] += 1
            prefix, marker, original_response = split_user_content_on_tutor_response(user_content)
            word_count = len(original_response.split())
            char_count = len(original_response)
            word_lengths.append(word_count)
            char_lengths.append(char_count)
            prepared_rows.append(
                PreparedSourceRow(
                    source_file=train_path,
                    source_row_index=row_index,
                    row=row,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    assistant_content=assistant_content,
                    conversation_prefix=prefix,
                    tutor_response_marker=marker,
                    original_tutor_response=original_response,
                    conversation_hash=json_hash(prefix.rstrip()),
                    original_tutor_response_words=word_count,
                    original_tutor_response_chars=char_count,
                )
            )

        length_stats = build_response_length_stats(word_lengths, char_lengths)
        summary = SchemaSummary(
            spec=spec,
            train_path=train_path,
            rows=rows,
            prepared_rows=prepared_rows,
            label_counts=label_counts,
            message_lengths=message_lengths,
            role_patterns=role_patterns,
            sample_row=rows[0],
            response_length_stats=length_stats,
            top_level_keys=tuple(rows[0].keys()),
        )
        summaries[alias] = summary

        print(f"[Schema] Task {alias} ({spec.prompt_key})")
        print(f"  train file: {train_path}")
        print(f"  rows: {len(rows)}")
        print(f"  message lengths: {dict(message_lengths)}")
        print(f"  role patterns: {dict(role_patterns)}")
        print(f"  conversation/input text lives at: messages[1].content")
        print(f"  assistant label/output lives at: messages[2].content")
        print("  tutor response replacement rule: rewrite only the trailing `Tutor Response: ...` segment inside messages[1].content")
        print(f"  label counts: {dict(label_counts)}")
        print(
            "  tutor response length stats:"
            f" words mean={length_stats.mean_words:.1f} median={length_stats.median_words:.1f}"
            f" p95={length_stats.p95_words} max={length_stats.max_words}"
            f" allowed<={length_stats.max_allowed_words};"
            f" chars mean={length_stats.mean_chars:.1f} median={length_stats.median_chars:.1f}"
            f" p95={length_stats.p95_chars} max={length_stats.max_chars}"
            f" allowed<={length_stats.max_allowed_chars}"
        )
        print("  sample row structure:")
        print(sample_row_preview(rows[0]))
        print()

    return summaries


class QwenRunner:
    def __init__(self, model_path: Path, load_in_4bit: bool) -> None:
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.torch_dtype = choose_torch_dtype()

    def _require_runtime(self) -> None:
        if torch is None:
            raise RuntimeError("torch is not installed in the current Python environment.")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is not installed in the current Python environment.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen model path does not exist: {self.model_path}")
        if self.load_in_4bit and BitsAndBytesConfig is None:
            raise RuntimeError("--load-in-4bit requested, but BitsAndBytesConfig is unavailable.")

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self._require_runtime()

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except (AttributeError, TypeError, ValueError):
            tokenizer_config = load_json(self.model_path / "tokenizer_config.json")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self.model_path / "tokenizer.json"),
                bos_token=self._special_token_content(tokenizer_config.get("bos_token"), "<s>"),
                eos_token=self._special_token_content(tokenizer_config.get("eos_token"), "</s>"),
                unk_token=self._special_token_content(tokenizer_config.get("unk_token"), "<unk>"),
                pad_token=self._special_token_content(tokenizer_config.get("pad_token"), "<pad>"),
            )
            tokenizer.chat_template = tokenizer_config.get("chat_template")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch is not None else None,
                bnb_4bit_use_double_quant=True,
            )

        print(f"Loading Qwen3-14B from {self.model_path} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _special_token_content(value: object, default: str) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str):
                return content
        return default

    def apply_chat_template_with_thinking(self, messages: Sequence[dict[str, str]]) -> str:
        self.load()
        assert self.tokenizer is not None
        try:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )

    def get_model_input_device(self) -> Any:
        assert self.model is not None
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def generate_text(
        self,
        messages: Sequence[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt_text = self.apply_chat_template_with_thinking(messages)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_device = self.get_model_input_device()
        encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False

        if torch is None:
            raise RuntimeError("torch is not installed in the current Python environment.")
        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        prompt_length = int(encoded["input_ids"].shape[-1])
        completion_tokens = outputs[0, prompt_length:]
        return self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()


def build_generation_messages(prompt_text: str, prepared_row: PreparedSourceRow) -> list[dict[str, str]]:
    conversation_context = prepared_row.conversation_prefix.rstrip()
    user_content = f"{conversation_context}\n\nTutor Response:"
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_content},
    ]


def build_verification_messages(prompt_text: str, synthetic_user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": synthetic_user_content},
    ]


def extract_generated_response(cleaned_output: str) -> str:
    return cleaned_output.strip()


def build_synthetic_user_content(prepared_row: PreparedSourceRow, generated_response: str) -> str:
    return f"{prepared_row.conversation_prefix}{prepared_row.tutor_response_marker}{generated_response}"


def build_synthetic_row(prepared_row: PreparedSourceRow, spec: TaskSpec, generated_response: str, target_label: str) -> dict[str, Any]:
    synthetic_row = copy.deepcopy(prepared_row.row)
    messages = synthetic_row["messages"]
    synthetic_user_content = build_synthetic_user_content(prepared_row, generated_response)
    messages[1]["content"] = synthetic_user_content
    if spec.is_multitask:
        messages[2]["content"] = format_mt_label_vector(target_label)
    else:
        messages[2]["content"] = target_label
    return synthetic_row


def validate_synthetic_row_shape(row: dict[str, Any], spec: TaskSpec) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Synthetic row does not preserve a 3-message chat structure.")
    roles = tuple(message.get("role") for message in messages if isinstance(message, dict))
    if roles != ("system", "user", "assistant"):
        raise ValueError(f"Synthetic row roles were {roles}, expected ('system', 'user', 'assistant').")
    user_content = messages[1].get("content")
    assistant_content = messages[2].get("content")
    if not isinstance(user_content, str) or "Tutor Response:" not in user_content:
        raise ValueError("Synthetic row user content could not be rebuilt with a tutor response marker.")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise ValueError("Synthetic row assistant label is empty.")
    if spec.is_multitask:
        parsed = parse_mt_label_block(assistant_content)
        if any(label == UNKNOWN_LABEL for label in parsed.values()):
            raise ValueError("Synthetic multitask assistant label is not in the original 4-line format.")
    else:
        if normalize_label(assistant_content) == UNKNOWN_LABEL:
            raise ValueError("Synthetic single-task assistant label is invalid.")


def quality_check_generated_response(
    generated_response: str,
    summary: SchemaSummary,
    seen_exact: set[str],
    seen_normalized: set[str],
) -> str | None:
    if not generated_response.strip():
        return "empty_response"

    lowered = generated_response.casefold()
    banned_snippets = ("evaluation:", "verification:", "label:", "here is")
    for snippet in banned_snippets:
        if snippet in lowered:
            return f"contains_{snippet.rstrip(':').replace(' ', '_')}"

    nonempty_lines = [line.strip() for line in generated_response.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        return "multiple_paragraphs"
    if "\n" in generated_response:
        return "multiple_paragraphs"

    word_count = len(generated_response.split())
    char_count = len(generated_response)
    stats = summary.response_length_stats
    if word_count > stats.max_allowed_words or char_count > stats.max_allowed_chars:
        return "too_long"

    exact_key = generated_response.strip()
    normalized_key = normalize_duplicate_text(generated_response)
    if exact_key in seen_exact:
        return "exact_duplicate"
    if normalized_key in seen_normalized:
        return "normalized_duplicate"
    return None


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def clear_selected_artifacts(args: argparse.Namespace, mode: str, spec: TaskSpec, labels: Sequence[str]) -> None:
    remove_if_exists(output_file_for_mode(args, mode, spec))
    remove_if_exists(manifest_path(args, mode, spec))
    for label in labels:
        for path in label_log_paths(args, mode, spec, label).values():
            remove_if_exists(path)


def load_resume_accepted_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = load_jsonl(path)
    for row in rows:
        if not row.get("accepted", False):
            raise ValueError(f"Accepted log contains a non-accepted row: {path}")
        if "synthetic_row" not in row or not isinstance(row["synthetic_row"], dict):
            raise ValueError(f"Accepted log row is missing synthetic_row payload: {path}")
    return rows


def summarize_verifier_distribution(distribution: Counter[str]) -> str:
    if not distribution:
        return "{}"
    ordered = {label: distribution[label] for label in sorted(distribution)}
    return json.dumps(ordered, ensure_ascii=True)


def progress_line(
    mode: str,
    spec: TaskSpec,
    label: str,
    accepted_count: int,
    target_count: int,
    rejected_count: int,
    recent_rejections: deque[str],
    verifier_distribution: Counter[str] | None,
) -> str:
    parts = [
        f"[{mode}] task={spec.alias}",
        f"label={label}",
        f"accepted={accepted_count}/{target_count}",
        f"rejected={rejected_count}",
    ]
    if recent_rejections:
        parts.append(f"latest_rejections={list(recent_rejections)}")
    if verifier_distribution is not None:
        parts.append(f"verifier_distribution={summarize_verifier_distribution(verifier_distribution)}")
    return " | ".join(parts)


def print_progress(
    mode: str,
    spec: TaskSpec,
    label: str,
    accepted_count: int,
    target_count: int,
    rejected_count: int,
    recent_rejections: deque[str],
    verifier_distribution: Counter[str] | None,
) -> None:
    print(
        progress_line(
            mode=mode,
            spec=spec,
            label=label,
            accepted_count=accepted_count,
            target_count=target_count,
            rejected_count=rejected_count,
            recent_rejections=recent_rejections,
            verifier_distribution=verifier_distribution,
        ),
        flush=True,
    )


def target_attempt_limit(target_per_label: int, multiplier: int) -> int:
    return max(target_per_label * multiplier, target_per_label, 1)


def generation_prompt_for(prompts: dict[str, Any], spec: TaskSpec, label: str) -> str:
    prompt_block = prompts["augmentation_generation"]["prompts"][spec.prompt_key]
    if spec.is_multitask:
        mt_key = "No_No_No_No" if label == "No" else "To some extent_To some extent_To some extent_To some extent"
        prompt_text = prompt_block[mt_key]
    else:
        prompt_text = prompt_block[label]
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"Missing augmentation_generation prompt for task {spec.alias} label {label}.")
    return prompt_text


def verification_prompt_for(prompts: dict[str, Any], prompt_key: str) -> str:
    prompt_text = prompts["self_verification"]["prompts"][prompt_key]
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"Missing self_verification prompt for {prompt_key}.")
    return prompt_text


def run_single_verification(
    qwen: QwenRunner,
    prompt_text: str,
    synthetic_user_content: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str, str, str | None]:
    raw_output = qwen.generate_text(
        build_verification_messages(prompt_text, synthetic_user_content),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    cleaned_output, cleaning_issue = clean_qwen_thinking_output(raw_output)
    if cleaning_issue is not None:
        return UNKNOWN_LABEL, raw_output, cleaned_output, cleaning_issue
    return recover_evaluation_label(cleaned_output), raw_output, cleaned_output, None


def generate_for_label(
    mode: str,
    spec: TaskSpec,
    label: str,
    summary: SchemaSummary,
    prompts: dict[str, Any],
    qwen: QwenRunner | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = label_log_paths(args, mode, spec, label)
    if not args.resume and not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise RuntimeError(
                f"Existing logs found for mode={mode} task={spec.alias} label={label}: {joined}. "
                "Use --resume to continue or --overwrite to replace them."
            )
    if args.overwrite:
        for path in paths.values():
            remove_if_exists(path)

    accepted_log_rows = load_resume_accepted_records(paths["accepted"]) if args.resume else []
    accepted_rows: list[dict[str, Any]] = [copy.deepcopy(row["synthetic_row"]) for row in accepted_log_rows]
    seen_exact = {row["generated_response"].strip() for row in accepted_log_rows}
    seen_normalized = {normalize_duplicate_text(row["generated_response"]) for row in accepted_log_rows}
    verifier_distribution: Counter[str] = Counter()
    rejected_count = count_jsonl_rows(paths["rejected"]) if args.resume and paths["rejected"].exists() else 0
    candidate_index = count_jsonl_rows(paths["candidates"]) if args.resume and paths["candidates"].exists() else 0
    recent_rejections: deque[str] = deque(maxlen=5)

    if accepted_rows:
        for row in accepted_log_rows:
            predictions = row.get("verifier_predictions")
            if isinstance(predictions, dict):
                if spec.is_multitask:
                    verifier_distribution.update(
                        f"{dimension}={label_value}" for dimension, label_value in predictions.items()
                    )
                else:
                    verifier_distribution.update(predictions.values())

    print_progress(
        mode=mode,
        spec=spec,
        label=label,
        accepted_count=len(accepted_rows),
        target_count=args.target_per_label,
        rejected_count=rejected_count,
        recent_rejections=recent_rejections,
        verifier_distribution=verifier_distribution if mode == "genverify" else None,
    )

    if args.dry_run:
        return {
            "label": label,
            "accepted_rows": accepted_rows,
            "accepted_count": len(accepted_rows),
            "rejected_count": rejected_count,
            "attempted_count": candidate_index,
            "verifier_distribution": dict(verifier_distribution),
        }

    if qwen is None:
        raise RuntimeError("Qwen runner is required for non-dry-run generation.")

    seed_material = f"{mode}|{spec.alias}|{label}"
    rng_seed = args.seed + int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    max_attempts = target_attempt_limit(args.target_per_label, args.max_attempts_multiplier)
    generation_prompt = generation_prompt_for(prompts, spec, label)

    while len(accepted_rows) < args.target_per_label and candidate_index < max_attempts:
        candidate_index += 1
        source_row = rng.choice(summary.prepared_rows)
        raw_generation = qwen.generate_text(
            build_generation_messages(generation_prompt, source_row),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        cleaned_generation, cleaning_issue = clean_qwen_thinking_output(raw_generation)
        generated_response = extract_generated_response(cleaned_generation)
        rejection_reason = cleaning_issue
        if rejection_reason is None:
            rejection_reason = quality_check_generated_response(
                generated_response=generated_response,
                summary=summary,
                seen_exact=seen_exact,
                seen_normalized=seen_normalized,
            )
        verifier_predictions: dict[str, str] | None = None
        verifier_raw_outputs: dict[str, str] | None = None
        verifier_cleaned_outputs: dict[str, str] | None = None
        verifier_cleaning_issues: dict[str, str] | None = None

        if rejection_reason is None:
            try:
                synthetic_row = build_synthetic_row(source_row, spec, generated_response, label)
                validate_synthetic_row_shape(synthetic_row, spec)
            except Exception as exc:  # pragma: no cover - safety
                rejection_reason = f"schema_insertion_failed:{type(exc).__name__}"
                synthetic_row = None
        else:
            synthetic_row = None

        if rejection_reason is None and mode == "genverify":
            synthetic_user_content = synthetic_row["messages"][1]["content"]
            if spec.is_multitask:
                verifier_predictions = {}
                verifier_raw_outputs = {}
                verifier_cleaned_outputs = {}
                verifier_cleaning_issues = {}
                expected = mt_expected_vector(label)
                failed_dimensions: list[str] = []
                for dimension in MT_DIMENSIONS:
                    verifier_prompt = verification_prompt_for(prompts, dimension)
                    predicted_label, verifier_raw_output, verifier_cleaned_output, verifier_cleaning_issue = run_single_verification(
                        qwen=qwen,
                        prompt_text=verifier_prompt,
                        synthetic_user_content=synthetic_user_content,
                        max_new_tokens=args.verification_max_new_tokens,
                        temperature=args.verification_temperature,
                        top_p=args.verification_top_p,
                    )
                    verifier_raw_outputs[dimension] = verifier_raw_output
                    verifier_cleaned_outputs[dimension] = verifier_cleaned_output
                    if verifier_cleaning_issue is not None:
                        verifier_cleaning_issues[dimension] = verifier_cleaning_issue
                    verifier_predictions[dimension] = predicted_label
                    verifier_distribution.update([f"{dimension}={predicted_label}"])
                    if verifier_cleaning_issue is not None:
                        failed_dimensions.append(f"{dimension}:{verifier_cleaning_issue}")
                    elif predicted_label != expected[dimension]:
                        failed_dimensions.append(f"{dimension}:{predicted_label}->{expected[dimension]}")
                if failed_dimensions:
                    rejection_reason = "verification_mismatch:" + ",".join(failed_dimensions)
            else:
                verifier_prompt = verification_prompt_for(prompts, spec.prompt_key)
                predicted_label, verifier_raw_output, verifier_cleaned_output, verifier_cleaning_issue = run_single_verification(
                    qwen=qwen,
                    prompt_text=verifier_prompt,
                    synthetic_user_content=synthetic_user_content,
                    max_new_tokens=args.verification_max_new_tokens,
                    temperature=args.verification_temperature,
                    top_p=args.verification_top_p,
                )
                verifier_raw_outputs = {spec.prompt_key: verifier_raw_output}
                verifier_cleaned_outputs = {spec.prompt_key: verifier_cleaned_output}
                verifier_cleaning_issues = (
                    {spec.prompt_key: verifier_cleaning_issue} if verifier_cleaning_issue is not None else {}
                )
                verifier_predictions = {spec.prompt_key: predicted_label}
                verifier_distribution.update([predicted_label])
                if verifier_cleaning_issue is not None:
                    rejection_reason = f"verification_mismatch:{verifier_cleaning_issue}"
                elif predicted_label != label:
                    rejection_reason = f"verification_mismatch:{predicted_label}->{label}"

        accepted = rejection_reason is None and synthetic_row is not None
        log_row: dict[str, Any] = {
            "mode": mode,
            "task": spec.alias,
            "task_name": spec.prompt_key,
            "target_label": label,
            "source_train_file": str(source_row.source_file),
            "source_row_index": source_row.source_row_index,
            "source_conversation_hash": source_row.conversation_hash,
            "generated_response": generated_response,
            "raw_model_output": raw_generation,
            "cleaned_model_output": cleaned_generation,
            "verifier_predictions": verifier_predictions,
            "verifier_raw_outputs": verifier_raw_outputs,
            "verifier_cleaned_outputs": verifier_cleaned_outputs,
            "verifier_cleaning_issues": verifier_cleaning_issues,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "attempt_index": candidate_index,
            "timestamp": utc_now(),
            "raw_generation": raw_generation,
        }

        append_jsonl(paths["candidates"], log_row)
        if accepted:
            log_row_with_row = dict(log_row)
            log_row_with_row["synthetic_row"] = synthetic_row
            append_jsonl(paths["accepted"], log_row_with_row)
            accepted_rows.append(synthetic_row)
            seen_exact.add(generated_response.strip())
            seen_normalized.add(normalize_duplicate_text(generated_response))
        else:
            rejected_count += 1
            if rejection_reason:
                recent_rejections.append(rejection_reason)
            append_jsonl(paths["rejected"], log_row)

        if accepted or candidate_index <= 5 or candidate_index % 10 == 0:
            print_progress(
                mode=mode,
                spec=spec,
                label=label,
                accepted_count=len(accepted_rows),
                target_count=args.target_per_label,
                rejected_count=rejected_count,
                recent_rejections=recent_rejections,
                verifier_distribution=verifier_distribution if mode == "genverify" else None,
            )

    if len(accepted_rows) < args.target_per_label and not args.allow_partial:
        raise RuntimeError(
            f"Failed to reach target count for mode={mode} task={spec.alias} label={label}: "
            f"{len(accepted_rows)}/{args.target_per_label} accepted after {candidate_index} attempts."
        )

    return {
        "label": label,
        "accepted_rows": accepted_rows,
        "accepted_count": len(accepted_rows),
        "rejected_count": rejected_count,
        "attempted_count": candidate_index,
        "verifier_distribution": dict(verifier_distribution),
    }


def validate_final_output(
    mode: str,
    spec: TaskSpec,
    summary: SchemaSummary,
    output_path: Path,
    selected_labels: Sequence[str],
    target_per_label: int,
    val_snapshot_before: dict[str, dict[str, int]],
    test_snapshot_before: dict[str, dict[str, int]],
    enforce_exact_target: bool,
) -> dict[str, Any]:
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file missing: {output_path}")

    rows = load_jsonl(output_path)
    original_count = len(summary.rows)
    synthetic_rows = rows[original_count:]
    expected_final_count = original_count + len(selected_labels) * target_per_label

    if len(rows) < original_count:
        raise ValueError(f"{output_path} has fewer rows than the original train file.")
    if rows[:original_count] != summary.rows:
        raise ValueError(f"{output_path} does not preserve the original rows exactly at the top of the file.")
    if enforce_exact_target and len(rows) != expected_final_count:
        raise ValueError(
            f"{output_path} final_count={len(rows)} but expected {expected_final_count} "
            f"(original={original_count}, labels={len(selected_labels)}, target={target_per_label})."
        )

    before_counts = Counter(summary.label_counts)
    after_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        messages, _, _, assistant_content = extract_messages(row, output_path, row_index)
        del messages  # already validated above; keep extraction side effects only
        after_counts[assistant_content] += 1

    if spec.is_multitask:
        for label in selected_labels:
            vector_text = format_mt_label_vector(label)
            expected_after = before_counts[vector_text] + target_per_label
            if enforce_exact_target and after_counts[vector_text] != expected_after:
                raise ValueError(
                    f"{output_path} expected {vector_text!r} count {expected_after}, found {after_counts[vector_text]}."
                )
        for label_text, before_value in before_counts.items():
            if label_text not in {format_mt_label_vector(label) for label in selected_labels}:
                synthetic_increment = after_counts[label_text] - before_value
                if synthetic_increment != 0:
                    raise ValueError(f"{output_path} changed non-target MT label count for {label_text!r} by {synthetic_increment}.")
    else:
        for label in selected_labels:
            expected_after = before_counts[label] + target_per_label
            if enforce_exact_target and after_counts[label] != expected_after:
                raise ValueError(f"{output_path} expected label {label!r} count {expected_after}, found {after_counts[label]}.")
        for label in ALL_SINGLE_LABELS:
            if label not in selected_labels:
                synthetic_increment = after_counts[label] - before_counts[label]
                if synthetic_increment != 0:
                    raise ValueError(f"{output_path} changed non-target label {label!r} by {synthetic_increment}.")

    val_snapshot_after = snapshot_directory_metadata(ROOT / "data" / "val")
    test_snapshot_after = snapshot_directory_metadata(ROOT / "data" / "test")
    assert_snapshot_unchanged(val_snapshot_before, val_snapshot_after, "Validation")
    assert_snapshot_unchanged(test_snapshot_before, test_snapshot_after, "Test")

    validation_summary = {
        "mode": mode,
        "task": spec.alias,
        "task_name": spec.prompt_key,
        "output_file": str(output_path),
        "original_count": original_count,
        "final_count": len(rows),
        "expected_final_count": expected_final_count,
        "before_label_counts": dict(before_counts),
        "after_label_counts": dict(after_counts),
        "selected_labels": list(selected_labels),
        "validation_files_touched": False,
        "test_files_touched": False,
        "validated_at": utc_now(),
        "synthetic_count": len(synthetic_rows),
    }
    return validation_summary


def write_manifest(args: argparse.Namespace, mode: str, spec: TaskSpec, payload: dict[str, Any]) -> None:
    path = manifest_path(args, mode, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_summary_all(args: argparse.Namespace, mode: str, payload: dict[str, Any]) -> None:
    path = summary_all_path(args, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_per_label <= 0:
        raise ValueError("--target-per-label must be greater than zero.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero.")
    if args.verification_max_new_tokens <= 0:
        raise ValueError("--verification-max-new-tokens must be greater than zero.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in the range (0, 1].")
    if args.verification_temperature < 0.0:
        raise ValueError("--verification-temperature must be non-negative.")
    if not 0.0 < args.verification_top_p <= 1.0:
        raise ValueError("--verification-top-p must be in the range (0, 1].")
    if args.max_attempts_multiplier <= 0:
        raise ValueError("--max-attempts-multiplier must be greater than zero.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if not args.prompts_json.exists():
        raise FileNotFoundError(f"prompts.json not found: {args.prompts_json}")
    if args.train_dir.resolve() != TRAIN_DIR.resolve():
        raise ValueError(
            f"This script is intentionally locked to the training directory only. "
            f"Expected {TRAIN_DIR}, got {args.train_dir}."
        )
    if args.out_gen_dir is None:
        args.out_gen_dir = args.train_dir / "Gen"
    if args.out_genverify_dir is None:
        args.out_genverify_dir = args.train_dir / "Gen+Verify"


def run_validation_only_if_requested(
    args: argparse.Namespace,
    schema_summaries: dict[str, SchemaSummary],
    val_snapshot_before: dict[str, dict[str, int]],
    test_snapshot_before: dict[str, dict[str, int]],
) -> bool:
    validation_only = args.validate_existing and not args.resume and not args.overwrite and not args.dry_run
    if not validation_only:
        return False

    print("Running validation-only mode against existing selected outputs.")
    for mode in resolve_modes(args.mode):
        for alias in args.tasks:
            summary = schema_summaries[alias]
            output_path = output_file_for_mode(args, mode, summary.spec)
            validation_summary = validate_final_output(
                mode=mode,
                spec=summary.spec,
                summary=summary,
                output_path=output_path,
                selected_labels=args.labels,
                target_per_label=args.target_per_label,
                val_snapshot_before=val_snapshot_before,
                test_snapshot_before=test_snapshot_before,
                enforce_exact_target=not args.allow_partial,
            )
            print(json.dumps(validation_summary, indent=2, ensure_ascii=True))
    return True


def main() -> None:
    configure_slurm_streams()
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        prompts = load_json(args.prompts_json)

        val_snapshot_before = snapshot_directory_metadata(ROOT / "data" / "val")
        test_snapshot_before = snapshot_directory_metadata(ROOT / "data" / "test")

        schema_summaries = inspect_train_schema(args.train_dir)
        if run_validation_only_if_requested(args, schema_summaries, val_snapshot_before, test_snapshot_before):
            return

        modes = resolve_modes(args.mode)
        if args.overwrite:
            for mode in modes:
                for alias in args.tasks:
                    clear_selected_artifacts(args, mode, TASK_SPECS[alias], args.labels)

        if args.dry_run:
            print("Dry run only. Planned outputs:")
            print(f"  gen dir: {args.out_gen_dir}")
            print(f"  genverify dir: {args.out_genverify_dir}")
            for mode in modes:
                for alias in args.tasks:
                    spec = TASK_SPECS[alias]
                    print(f"  - {mode}: {output_file_for_mode(args, mode, spec)}")
            return

        qwen = QwenRunner(model_path=args.model_path, load_in_4bit=args.load_in_4bit)
        overall_summary: dict[str, Any] = {}

        for mode in modes:
            mode_summary: dict[str, Any] = {
                "mode": mode,
                "tasks": {},
                "completed_at": None,
                "selected_tasks": list(args.tasks),
                "selected_labels": list(args.labels),
                "target_per_label": args.target_per_label,
                "model_path": str(args.model_path),
            }

            for alias in args.tasks:
                summary = schema_summaries[alias]
                spec = summary.spec
                output_path = output_file_for_mode(args, mode, spec)
                if output_path.exists() and not args.resume and not args.overwrite:
                    raise RuntimeError(
                        f"Existing output file found: {output_path}. "
                        "Use --resume to continue or --overwrite to replace it."
                    )
                task_payload: dict[str, Any] = {
                    "task": alias,
                    "task_name": spec.prompt_key,
                    "train_file": str(summary.train_path),
                    "output_file": str(output_path),
                    "labels": {},
                    "started_at": utc_now(),
                }

                label_results: dict[str, list[dict[str, Any]]] = {}
                for label in args.labels:
                    label_result = generate_for_label(
                        mode=mode,
                        spec=spec,
                        label=label,
                        summary=summary,
                        prompts=prompts,
                        qwen=qwen,
                        args=args,
                    )
                    label_results[label] = label_result["accepted_rows"]
                    task_payload["labels"][label] = {
                        "accepted_count": label_result["accepted_count"],
                        "rejected_count": label_result["rejected_count"],
                        "attempted_count": label_result["attempted_count"],
                        "verifier_distribution": label_result["verifier_distribution"],
                    }

                final_rows = list(summary.rows)
                for label in args.labels:
                    final_rows.extend(label_results[label])

                write_jsonl(output_path, final_rows)

                validation_summary = validate_final_output(
                    mode=mode,
                    spec=spec,
                    summary=summary,
                    output_path=output_path,
                    selected_labels=args.labels,
                    target_per_label=args.target_per_label,
                    val_snapshot_before=val_snapshot_before,
                    test_snapshot_before=test_snapshot_before,
                    enforce_exact_target=not args.allow_partial,
                )
                task_payload["validation"] = validation_summary
                task_payload["completed_at"] = utc_now()
                write_manifest(args, mode, spec, task_payload)
                mode_summary["tasks"][alias] = task_payload
                print(json.dumps(validation_summary, indent=2, ensure_ascii=True))

            mode_summary["completed_at"] = utc_now()
            write_summary_all(args, mode, mode_summary)
            overall_summary[mode] = mode_summary

        print(json.dumps(overall_summary, indent=2, ensure_ascii=True))

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
