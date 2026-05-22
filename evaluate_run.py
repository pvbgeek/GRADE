#!/usr/bin/env python3
"""TutorMind run evaluator.

Purpose
-------
This script evaluates exactly one TutorMind prediction run at a time and
appends one or more result rows to a results CSV.

  Expected prediction CSV schemas
-------------------------------
Single-task runs (MI / ML / PG / Act):
- required column: ``pred_label``
- optional column: ``raw_output``

Multitask runs (MT):
- required columns: ``pred_mi``, ``pred_ml``, ``pred_pg``, ``pred_act``
- optional column: ``raw_output``

Validation input expectation
----------------------------
- MI uses ``mistake_identification_val.jsonl``
- ML uses ``mistake_location_val.jsonl``
- PG uses ``providing_guidance_val.jsonl``
- Act uses ``actionability_val.jsonl``
- MT uses ``multitask_val.jsonl``

Critical row-order assumption
-----------------------------
Predictions are aligned to validation labels strictly by row order. No
``example_id`` matching is used.

Unknown handling policy
-----------------------
- values outside ``Yes`` / ``No`` / ``To some extent`` normalize to ``Unknown``
- ``Unknown`` rows are excluded from metric computation
- ``Unknown`` rows are still counted and logged in the output CSV
- if all rows for a scored dimension are ``Unknown``, the script fails

Strict vs lenient scoring
-------------------------
- Strict scoring = macro F1 and accuracy over ``Yes`` / ``No`` / ``To some extent``
- Lenient scoring = collapse ``Yes`` + ``To some extent`` -> ``Yes``, and ``No`` -> ``No``

Output CSV schema
-----------------
- ``task``: the evaluated run type supplied on the CLI, such as ``MI`` or ``MT``
- ``metric_dimension``: the scored dimension; equals the task for single-task
  runs and is one of ``MI`` / ``ML`` / ``PG`` / ``Act`` for MT runs
- ``strict_f1`` / ``strict_acc``: strict metrics over the 3-way label set
- ``lenient_f1`` / ``lenient_acc``: lenient metrics after binary collapsing
- ``n_total_rows``: total number of prediction rows read
- ``n_scored_rows``: number of rows included in metric computation after
  excluding ``Unknown``
- ``n_unknown``: number of rows normalized to ``Unknown``
- ``unknown_rate``: ``n_unknown / n_total_rows``
- ``unknown_row_indices``: semicolon-separated zero-based row indices for ``Unknown`` predictions

CLI arguments
-------------
- ``--predictions``: path to the predictions CSV for one run
- ``--task``: one of ``MI``, ``ML``, ``PG``, ``Act``, ``MT``
- ``--run-id``: run identifier written to the results CSV
- ``--model``: model name written to the results CSV
- ``--method``: evaluation/inference method name
- ``--aug``: augmentation setting label
- ``--think``: thinking/reasoning setting label
- ``--val-mi``: MI validation path; required only for ``--task MI``
- ``--val-ml``: ML validation path; required only for ``--task ML``
- ``--val-pg``: PG validation path; required only for ``--task PG``
- ``--val-act``: Act validation path; required only for ``--task Act``
- ``--val-mt``: MT validation path; required only for ``--task MT``
- ``--out``: output results CSV path; rows are appended to this file

Example commands
----------------
Single-task example:
    python /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/evaluate_run.py \\
        --predictions /tmp/runs/run_001_mi_predictions.csv \\
        --task MI \\
        --run-id 001 \\
        --model LLaMA-3.1-8B \\
        --method Zero-shot \\
        --aug None \\
        --think N/A \\
        --val-mi /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl \\
        --out /tmp/tutormind_results.csv

Multitask example:
    python /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/evaluate_run.py \\
        --predictions /tmp/runs/run_005_mt_predictions.csv \\
        --task MT \\
        --run-id 005 \\
        --model LLaMA-3.1-8B \\
        --method Zero-shot \\
        --aug None \\
        --think N/A \\
        --val-mt /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl \\
        --out /tmp/tutormind_results.csv

Example human-readable output
-----------------------------
Single-task example:
    Run 001 | Model: LLaMA-3.1-8B | Method: Zero-shot | Task: MI | Aug: None | Think: N/A | Strict F1: 0.9123 | Strict Acc: 0.9000 | Lenient F1: 0.9500 | Lenient Acc: 0.9400 | Unknown: 2 | Unknown Rate: 0.0100 | Unknown Row Indices: 13;42

Multitask example:
    Run 005 | Model: LLaMA-3.1-8B | Method: Zero-shot | Task: MT | Aug: None | Think: N/A | MI Strict F1: 0.9123 | MI Strict Acc: 0.9000 | MI Lenient F1: 0.9500 | MI Lenient Acc: 0.9400 | Unknown: 2 | Unknown Rate: 0.0100 | Unknown Row Indices: 13;42
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


STRICT_LABELS = ["Yes", "No", "To some extent"]
# Lenient scoring collapses the 3-way labels into a binary Yes/No scheme.
LENIENT_LABELS = ["Yes", "No"]
UNKNOWN_LABEL = "Unknown"
SINGLE_TASKS = ["MI", "ML", "PG", "Act"]
ALL_TASKS = SINGLE_TASKS + ["MT"]

TASK_TO_VAL_ARG = {
    "MI": "val_mi",
    "ML": "val_ml",
    "PG": "val_pg",
    "Act": "val_act",
    "MT": "val_mt",
}

TASK_TO_FILE_HINT = {
    "MI": "mistake_identification_val.jsonl",
    "ML": "mistake_location_val.jsonl",
    "PG": "providing_guidance_val.jsonl",
    "Act": "actionability_val.jsonl",
    "MT": "multitask_val.jsonl",
}

MT_FIELD_ALIASES = {
    "mistakeidentification": "MI",
    "mistakelocation": "ML",
    "providingguidance": "PG",
    "actionability": "Act",
}

RESULT_COLUMNS = [
    "run_id",
    "model",
    "method",
    "task",
    "metric_dimension",
    "aug",
    "think",
    "strict_f1",
    "strict_acc",
    "lenient_f1",
    "lenient_acc",
    "n_total_rows",
    "n_scored_rows",
    "n_unknown",
    "unknown_rate",
    "unknown_row_indices",
    "prediction_file",
    "timestamp",
]

MASTER_CSV_PATH = Path(__file__).resolve().parent / "master_metrics.csv"


@dataclass
class EvaluationSummary:
    """Container for one scored task or dimension."""

    metric_dimension: str
    strict_f1: float
    strict_acc: float
    lenient_f1: float
    lenient_acc: float
    n_total_rows: int
    n_scored_rows: int
    n_unknown: int
    unknown_rate: float
    unknown_row_indices: str


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single TutorMind run and append results to a CSV."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV.")
    parser.add_argument(
        "--task",
        required=True,
        choices=ALL_TASKS,
        help="Task to evaluate: MI, ML, PG, Act, or MT.",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--method", required=True, help="Method name.")
    parser.add_argument("--aug", required=True, help="Augmentation setting.")
    parser.add_argument("--think", required=True, help="Thinking setting.")
    parser.add_argument("--val-mi", help="Path to mistake_identification_val.jsonl.")
    parser.add_argument("--val-ml", help="Path to mistake_location_val.jsonl.")
    parser.add_argument("--val-pg", help="Path to providing_guidance_val.jsonl.")
    parser.add_argument("--val-act", help="Path to actionability_val.jsonl.")
    parser.add_argument("--val-mt", help="Path to multitask_val.jsonl.")
    parser.add_argument("--out", required=True, help="Path to results CSV.")
    return parser


def normalize_label(value: object) -> str:
    """Normalize mild label formatting variants to a canonical label.

    Any value that cannot be normalized to Yes / No / To some extent becomes
    Unknown. This function is intentionally conservative: it handles mild
    casing, whitespace, punctuation, and common label prefixes, but does not
    attempt to interpret free-form explanations.
    """
    if value is None or pd.isna(value):
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


def canonicalize_text(text: str) -> str:
    """Collapse spacing and mild punctuation for label matching."""
    stripped = text.strip().strip("\"'`")
    stripped = stripped.strip(" .,!?:;")
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


def collapse_lenient(label: str) -> str:
    """Collapse strict labels into lenient binary Yes/No labels."""
    if label in {"Yes", "To some extent"}:
        return "Yes"
    if label == "No":
        return "No"
    raise ValueError(f"Cannot collapse unexpected label: {label!r}")


def ensure_file_exists(path_str: str, arg_name: str) -> Path:
    """Validate that a CLI path was provided and points to an existing file."""
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{arg_name} does not exist or is not a file: {path}")
    return path.resolve()


def get_required_validation_path(args: argparse.Namespace) -> Path:
    """Return the single validation path required for the selected task."""
    arg_name = TASK_TO_VAL_ARG[args.task]
    value = getattr(args, arg_name)
    if not value:
        hint = TASK_TO_FILE_HINT[args.task]
        flag = f"--{arg_name.replace('_', '-')}"
        raise ValueError(
            f"{flag} is required for task {args.task}. Expected validation file: {hint}"
        )
    return ensure_file_exists(value, f"--{arg_name.replace('_', '-')}")


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


def extract_last_assistant_content(record: dict, source_path: Path, row_index: int) -> str:
    """Return the content of the last assistant message in a TutorMind record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(
            f"{source_path} row {row_index} is missing a valid 'messages' list."
        )

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"{source_path} row {row_index} has a non-string assistant content."
                )
            return content

    raise ValueError(f"{source_path} row {row_index} has no assistant message.")


def load_single_task_labels(path: Path) -> List[str]:
    """Load canonical labels for a single-task validation file."""
    labels: List[str] = []
    for row_index, record in enumerate(load_jsonl_records(path)):
        assistant_content = extract_last_assistant_content(record, path, row_index)
        label = normalize_label(assistant_content)
        if label == UNKNOWN_LABEL:
            raise ValueError(
                f"{path} row {row_index} has an invalid reference label: {assistant_content!r}"
            )
        labels.append(label)
    return labels


def parse_multitask_assistant_content(content: str, source_path: Path, row_index: int) -> Dict[str, str]:
    """Parse multitask assistant content into MI / ML / PG / Act labels."""
    labels: Dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        field_name, raw_value = line.split(":", 1)
        field_key = canonicalize_text(field_name).replace(" ", "")
        dimension = MT_FIELD_ALIASES.get(field_key)
        if not dimension:
            continue
        if dimension in labels:
            raise ValueError(
                f"{source_path} row {row_index} contains duplicate field for {dimension}."
            )

        label = normalize_label(raw_value)
        if label == UNKNOWN_LABEL:
            raise ValueError(
                f"{source_path} row {row_index} has an invalid {dimension} label: {raw_value!r}"
            )
        labels[dimension] = label

    missing = [dimension for dimension in SINGLE_TASKS if dimension not in labels]
    if missing:
        raise ValueError(
            f"{source_path} row {row_index} is missing multitask labels for: {', '.join(missing)}"
        )
    return labels


def load_multitask_labels(path: Path) -> Dict[str, List[str]]:
    """Load canonical labels for all multitask dimensions."""
    labels_by_dimension = {dimension: [] for dimension in SINGLE_TASKS}
    for row_index, record in enumerate(load_jsonl_records(path)):
        assistant_content = extract_last_assistant_content(record, path, row_index)
        parsed = parse_multitask_assistant_content(assistant_content, path, row_index)
        for dimension in SINGLE_TASKS:
            labels_by_dimension[dimension].append(parsed[dimension])
    return labels_by_dimension


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions CSV as strings so label parsing is predictable."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Failed to read predictions CSV {path}: {exc}") from exc


def require_prediction_columns(frame: pd.DataFrame, required_columns: Sequence[str], task: str) -> None:
    """Validate that all required prediction columns are present."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Predictions CSV for task {task} is missing required column(s): {', '.join(missing)}"
        )


def validate_row_count(predictions: pd.DataFrame, n_labels: int, task: str) -> None:
    """Ensure predictions and validation labels have the same row count."""
    n_predictions = len(predictions)
    if n_predictions != n_labels:
        raise ValueError(
            f"Row-count mismatch for task {task}: predictions CSV has {n_predictions} rows "
            f"but validation data has {n_labels} rows."
        )


def compute_metrics(metric_dimension: str, y_true: Sequence[str], raw_predictions: Iterable[object]) -> EvaluationSummary:
    """Score one task or dimension, excluding Unknown predictions."""
    canonical_predictions = [normalize_label(value) for value in raw_predictions]
    unknown_indices = [
        index for index, prediction in enumerate(canonical_predictions) if prediction == UNKNOWN_LABEL
    ]

    n_total_rows = len(canonical_predictions)
    n_unknown = len(unknown_indices)
    n_scored_rows = n_total_rows - n_unknown

    if n_total_rows != len(y_true):
        raise ValueError(
            f"Internal error for {metric_dimension}: label count and prediction count do not match."
        )
    if n_scored_rows == 0:
        raise ValueError(
            f"All predictions normalized to Unknown for metric dimension {metric_dimension}. "
            "Refusing to write metrics with zero scored rows."
        )

    scored_true = [
        label for label, prediction in zip(y_true, canonical_predictions) if prediction != UNKNOWN_LABEL
    ]
    scored_pred = [
        prediction for prediction in canonical_predictions if prediction != UNKNOWN_LABEL
    ]

    strict_f1 = f1_score(
        scored_true,
        scored_pred,
        labels=STRICT_LABELS,
        average="macro",
        zero_division=0,
    )
    strict_acc = accuracy_score(scored_true, scored_pred)

    lenient_true = [collapse_lenient(label) for label in scored_true]
    lenient_pred = [collapse_lenient(label) for label in scored_pred]
    lenient_f1 = f1_score(
        lenient_true,
        lenient_pred,
        labels=LENIENT_LABELS,
        average="macro",
        zero_division=0,
    )
    lenient_acc = accuracy_score(lenient_true, lenient_pred)

    return EvaluationSummary(
        metric_dimension=metric_dimension,
        strict_f1=round(float(strict_f1), 6),
        strict_acc=round(float(strict_acc), 6),
        lenient_f1=round(float(lenient_f1), 6),
        lenient_acc=round(float(lenient_acc), 6),
        n_total_rows=n_total_rows,
        n_scored_rows=n_scored_rows,
        n_unknown=n_unknown,
        unknown_rate=round(n_unknown / n_total_rows, 6),
        unknown_row_indices=";".join(str(index) for index in unknown_indices),
    )


def build_result_rows(
    args: argparse.Namespace,
    prediction_file: Path,
    summaries: Sequence[EvaluationSummary],
) -> List[dict]:
    """Convert computed summaries into CSV rows."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: List[dict] = []
    for summary in summaries:
        rows.append(
            {
                "run_id": args.run_id,
                "model": args.model,
                "method": args.method,
                "task": args.task,
                "metric_dimension": summary.metric_dimension,
                "aug": args.aug,
                "think": args.think,
                "strict_f1": summary.strict_f1,
                "strict_acc": summary.strict_acc,
                "lenient_f1": summary.lenient_f1,
                "lenient_acc": summary.lenient_acc,
                "n_total_rows": summary.n_total_rows,
                "n_scored_rows": summary.n_scored_rows,
                "n_unknown": summary.n_unknown,
                "unknown_rate": summary.unknown_rate,
                "unknown_row_indices": summary.unknown_row_indices,
                "prediction_file": str(prediction_file),
                "timestamp": timestamp,
            }
        )
    return rows


def format_metric(value: float) -> str:
    """Format metrics for human-readable printing."""
    return f"{value:.4f}"


def format_stdout_row(args: argparse.Namespace, summary: EvaluationSummary) -> str:
    """Render one human-readable output line."""
    prefix = (
        f"Run {args.run_id} | Model: {args.model} | Method: {args.method} | "
        f"Task: {args.task} | Aug: {args.aug} | Think: {args.think}"
    )

    if args.task == "MT":
        metric_part = (
            f"{summary.metric_dimension} Strict F1: {format_metric(summary.strict_f1)} | "
            f"{summary.metric_dimension} Strict Acc: {format_metric(summary.strict_acc)} | "
            f"{summary.metric_dimension} Lenient F1: {format_metric(summary.lenient_f1)} | "
            f"{summary.metric_dimension} Lenient Acc: {format_metric(summary.lenient_acc)}"
        )
    else:
        metric_part = (
            f"Strict F1: {format_metric(summary.strict_f1)} | "
            f"Strict Acc: {format_metric(summary.strict_acc)} | "
            f"Lenient F1: {format_metric(summary.lenient_f1)} | "
            f"Lenient Acc: {format_metric(summary.lenient_acc)}"
        )

    return (
        f"{prefix} | {metric_part} | Unknown: {summary.n_unknown} | "
        f"Unknown Rate: {summary.unknown_rate:.4f} | "
        f"Unknown Row Indices: {summary.unknown_row_indices}"
    )


def validate_results_header(path: Path) -> None:
    """Ensure an existing results CSV has the expected schema before appending."""
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return

    if header != RESULT_COLUMNS:
        raise ValueError(
            f"Existing results CSV header does not match expected columns.\n"
            f"Expected: {RESULT_COLUMNS}\n"
            f"Found:    {header}"
        )


def append_results(path: Path, rows: Sequence[dict]) -> None:
    """Append structured results rows to the output CSV."""
    validate_results_header(path)
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def normalize_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a results DataFrame matches the canonical evaluator schema."""
    normalized = frame.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized = normalized[RESULT_COLUMNS]
    return normalized.fillna("")


def upsert_master_results(master_path: Path, new_rows: Sequence[dict]) -> None:
    """Merge new evaluator rows into the master CSV, keeping the latest timestamp per key."""
    master_path.parent.mkdir(parents=True, exist_ok=True)

    if master_path.exists() and master_path.stat().st_size > 0:
        existing = normalize_results_frame(pd.read_csv(master_path, dtype=str, keep_default_na=False))
    else:
        existing = pd.DataFrame(columns=RESULT_COLUMNS)

    combined = pd.concat(
        [
            existing,
            normalize_results_frame(pd.DataFrame(new_rows, columns=RESULT_COLUMNS)),
        ],
        ignore_index=True,
    )
    combined["_timestamp_dt"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["_row_order"] = range(len(combined))
    combined = combined.sort_values(
        ["run_id", "metric_dimension", "_timestamp_dt", "_row_order"],
        kind="stable",
    )
    combined = combined.drop_duplicates(subset=["run_id", "metric_dimension"], keep="last")

    dim_order = {"MI": 0, "ML": 1, "PG": 2, "Act": 3, "MT": 4}
    combined["_run_id_int"] = pd.to_numeric(combined["run_id"], errors="coerce")
    combined["_dim_order"] = combined["metric_dimension"].map(dim_order).fillna(99).astype(int)
    combined = combined.sort_values(
        ["_run_id_int", "run_id", "_dim_order", "metric_dimension"],
        kind="stable",
        na_position="last",
    )
    combined = combined.drop(columns=["_timestamp_dt", "_row_order", "_run_id_int", "_dim_order"])
    combined.to_csv(master_path, index=False)


def evaluate_single_task(task: str, predictions: pd.DataFrame, validation_path: Path) -> EvaluationSummary:
    """Evaluate a single-task run."""
    require_prediction_columns(predictions, ["pred_label"], task)
    labels = load_single_task_labels(validation_path)
    validate_row_count(predictions, len(labels), task)
    return compute_metrics(task, labels, predictions["pred_label"].tolist())


def evaluate_multitask(predictions: pd.DataFrame, validation_path: Path) -> List[EvaluationSummary]:
    """Evaluate a multitask run and return one summary per dimension."""
    required_columns = ["pred_mi", "pred_ml", "pred_pg", "pred_act"]
    require_prediction_columns(predictions, required_columns, "MT")

    labels_by_dimension = load_multitask_labels(validation_path)
    n_labels = len(next(iter(labels_by_dimension.values())))
    validate_row_count(predictions, n_labels, "MT")

    prediction_columns = {
        "MI": "pred_mi",
        "ML": "pred_ml",
        "PG": "pred_pg",
        "Act": "pred_act",
    }

    return [
        compute_metrics(dimension, labels_by_dimension[dimension], predictions[column].tolist())
        for dimension, column in prediction_columns.items()
    ]


def main() -> None:
    """Parse CLI arguments, evaluate one run, print results, and append CSV rows."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        predictions_path = ensure_file_exists(args.predictions, "--predictions")
        validation_path = get_required_validation_path(args)
        output_path = Path(args.out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions = load_predictions(predictions_path)
        if args.task == "MT":
            summaries = evaluate_multitask(predictions, validation_path)
        else:
            summaries = [evaluate_single_task(args.task, predictions, validation_path)]

        for summary in summaries:
            print(format_stdout_row(args, summary))

        rows = build_result_rows(args, predictions_path, summaries)
        append_results(output_path, rows)
        upsert_master_results(MASTER_CSV_PATH, rows)

    except ValueError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
