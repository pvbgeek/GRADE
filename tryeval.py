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

Master CSV
----------
After writing the per-run output CSV, the script also upserts the same rows
into a master results CSV located at:
  <project_root>/master_metrics.csv
where <project_root> is detected as the directory two levels above the script
(i.e. TutorMind/). The master CSV is kept sorted by run_id then metric_dimension.
If a (run_id, metric_dimension) pair already exists, it is overwritten.

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

# Master CSV is located at TutorMind/master_metrics.csv
# Detected as: script_dir/../master_metrics.csv
# (script lives in TutorMind/, so parent is TutorMind/)
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
    parser = argparse.ArgumentParser(
        description="Evaluate a single TutorMind run and append results to a CSV."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV.")
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
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
    parser.add_argument("--out", required=True, help="Path to per-run results CSV.")
    return parser


def normalize_label(value: object) -> str:
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
    stripped = text.strip().strip("\"'`")
    stripped = stripped.strip(" .,!?:;")
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


def collapse_lenient(label: str) -> str:
    if label in {"Yes", "To some extent"}:
        return "Yes"
    if label == "No":
        return "No"
    raise ValueError(f"Cannot collapse unexpected label: {label!r}")


def ensure_file_exists(path_str: str, arg_name: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{arg_name} does not exist or is not a file: {path}")
    return path.resolve()


def get_required_validation_path(args: argparse.Namespace) -> Path:
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
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_path} row {row_index} is missing a valid 'messages' list.")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{source_path} row {row_index} has a non-string assistant content.")
            return content
    raise ValueError(f"{source_path} row {row_index} has no assistant message.")


def load_single_task_labels(path: Path) -> List[str]:
    labels: List[str] = []
    for row_index, record in enumerate(load_jsonl_records(path)):
        assistant_content = extract_last_assistant_content(record, path, row_index)
        label = normalize_label(assistant_content)
        if label == UNKNOWN_LABEL:
            raise ValueError(f"{path} row {row_index} has an invalid reference label: {assistant_content!r}")
        labels.append(label)
    return labels


def parse_multitask_assistant_content(content: str, source_path: Path, row_index: int) -> Dict[str, str]:
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
            raise ValueError(f"{source_path} row {row_index} contains duplicate field for {dimension}.")
        label = normalize_label(raw_value)
        if label == UNKNOWN_LABEL:
            raise ValueError(f"{source_path} row {row_index} has an invalid {dimension} label: {raw_value!r}")
        labels[dimension] = label
    missing = [dimension for dimension in SINGLE_TASKS if dimension not in labels]
    if missing:
        raise ValueError(f"{source_path} row {row_index} is missing multitask labels for: {', '.join(missing)}")
    return labels


def load_multitask_labels(path: Path) -> Dict[str, List[str]]:
    labels_by_dimension = {dimension: [] for dimension in SINGLE_TASKS}
    for row_index, record in enumerate(load_jsonl_records(path)):
        assistant_content = extract_last_assistant_content(record, path, row_index)
        parsed = parse_multitask_assistant_content(assistant_content, path, row_index)
        for dimension in SINGLE_TASKS:
            labels_by_dimension[dimension].append(parsed[dimension])
    return labels_by_dimension


def load_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Failed to read predictions CSV {path}: {exc}") from exc


def require_prediction_columns(frame: pd.DataFrame, required_columns: Sequence[str], task: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Predictions CSV for task {task} is missing required column(s): {', '.join(missing)}")


def validate_row_count(predictions: pd.DataFrame, n_labels: int, task: str) -> None:
    n_predictions = len(predictions)
    if n_predictions != n_labels:
        raise ValueError(
            f"Row-count mismatch for task {task}: predictions CSV has {n_predictions} rows "
            f"but validation data has {n_labels} rows."
        )


def compute_metrics(metric_dimension: str, y_true: Sequence[str], raw_predictions: Iterable[object]) -> EvaluationSummary:
    canonical_predictions = [normalize_label(value) for value in raw_predictions]
    unknown_indices = [
        index for index, prediction in enumerate(canonical_predictions) if prediction == UNKNOWN_LABEL
    ]
    n_total_rows = len(canonical_predictions)
    n_unknown = len(unknown_indices)
    n_scored_rows = n_total_rows - n_unknown
    if n_total_rows != len(y_true):
        raise ValueError(f"Internal error for {metric_dimension}: label count and prediction count do not match.")
    if n_scored_rows == 0:
        raise ValueError(
            f"All predictions normalized to Unknown for metric dimension {metric_dimension}. "
            "Refusing to write metrics with zero scored rows."
        )
    scored_true = [label for label, prediction in zip(y_true, canonical_predictions) if prediction != UNKNOWN_LABEL]
    scored_pred = [prediction for prediction in canonical_predictions if prediction != UNKNOWN_LABEL]
    strict_f1 = f1_score(scored_true, scored_pred, labels=STRICT_LABELS, average="macro", zero_division=0)
    strict_acc = accuracy_score(scored_true, scored_pred)
    lenient_true = [collapse_lenient(label) for label in scored_true]
    lenient_pred = [collapse_lenient(label) for label in scored_pred]
    lenient_f1 = f1_score(lenient_true, lenient_pred, labels=LENIENT_LABELS, average="macro", zero_division=0)
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
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: List[dict] = []
    for summary in summaries:
        rows.append({
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
        })
    return rows


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_stdout_row(args: argparse.Namespace, summary: EvaluationSummary) -> str:
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


def append_results(path: Path, rows: Sequence[dict]) -> None:
    """Append rows to per-run output CSV (simple append, no upsert)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def upsert_master_results(master_path: Path, new_rows: Sequence[dict]) -> None:
    """Upsert rows into master CSV.

    Logic:
    - If master CSV does not exist or is empty → create it with new rows
    - If (run_id, metric_dimension) already exists → overwrite that row
    - If not present → append
    - Always sort by run_id (numeric) then metric_dimension after writing
    """
    master_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing master CSV if it exists
    if master_path.exists() and master_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(master_path, dtype=str, keep_default_na=False)
            # Ensure all expected columns exist
            for col in RESULT_COLUMNS:
                if col not in existing.columns:
                    existing[col] = ""
            existing = existing[RESULT_COLUMNS]
        except Exception:
            existing = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=RESULT_COLUMNS)

    new_frame = pd.DataFrame(new_rows, columns=RESULT_COLUMNS)

    # Remove existing rows that match (run_id, metric_dimension) of new rows
    upsert_keys = set(zip(new_frame["run_id"], new_frame["metric_dimension"]))
    mask_keep = ~existing.apply(
        lambda row: (row["run_id"], row["metric_dimension"]) in upsert_keys, axis=1
    )
    existing_kept = existing[mask_keep]

    # Combine kept existing rows + new rows
    combined = pd.concat([existing_kept, new_frame], ignore_index=True)

    # Sort by run_id (numeric) then metric_dimension
    dim_order = {"MI": 0, "ML": 1, "PG": 2, "Act": 3, "MT": 4}
    combined["_run_id_int"] = pd.to_numeric(combined["run_id"], errors="coerce").fillna(0).astype(int)
    combined["_dim_order"] = combined["metric_dimension"].map(dim_order).fillna(99).astype(int)
    combined = combined.sort_values(["_run_id_int", "_dim_order"]).drop(
        columns=["_run_id_int", "_dim_order"]
    )

    combined.to_csv(master_path, index=False)

    action = "updated" if mask_keep.sum() < len(existing) else "appended"
    print(f"Master CSV {action} → {master_path} ({len(combined)} total rows)")


def evaluate_single_task(task: str, predictions: pd.DataFrame, validation_path: Path) -> EvaluationSummary:
    require_prediction_columns(predictions, ["pred_label"], task)
    labels = load_single_task_labels(validation_path)
    validate_row_count(predictions, len(labels), task)
    return compute_metrics(task, labels, predictions["pred_label"].tolist())


def evaluate_multitask(predictions: pd.DataFrame, validation_path: Path) -> List[EvaluationSummary]:
    required_columns = ["pred_mi", "pred_ml", "pred_pg", "pred_act"]
    require_prediction_columns(predictions, required_columns, "MT")
    labels_by_dimension = load_multitask_labels(validation_path)
    n_labels = len(next(iter(labels_by_dimension.values())))
    validate_row_count(predictions, n_labels, "MT")
    prediction_columns = {"MI": "pred_mi", "ML": "pred_ml", "PG": "pred_pg", "Act": "pred_act"}
    return [
        compute_metrics(dimension, labels_by_dimension[dimension], predictions[column].tolist())
        for dimension, column in prediction_columns.items()
    ]


def main() -> None:
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

        # Write per-run output CSV
        append_results(output_path, rows)

        # Upsert into master CSV (overwrite if run_id+metric_dimension exists, else append, always sort)
        upsert_master_results(MASTER_CSV_PATH, rows)

    except ValueError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()