#!/usr/bin/env python3
"""Black-box smoke tests for TutorMind/evaluate_run.py.

This harness generates tiny synthetic validation JSONLs and prediction CSVs in a
temporary workspace, invokes the evaluator through its CLI, inspects stdout and
the results CSV, and reports a clear pass/fail summary.

Default behavior:
- uses temporary files only
- deletes them automatically after the run
- exits nonzero if any required smoke test fails

Optional debug flags:
- --keep-temp: preserve the generated temporary workspace
- --verbose: print subprocess stdout/stderr for every test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SCRIPT_PATH = Path(__file__).resolve()
EVALUATOR_PATH = SCRIPT_PATH.parent / "evaluate_run.py"

EXPECTED_RESULT_COLUMNS = [
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

STRICT_LABELS = ["Yes", "No", "To some extent"]
LENIENT_LABELS = ["positive", "negative"]


@dataclass
class TestOutcome:
    """Outcome for one smoke test."""

    name: str
    passed: bool
    message: str = ""


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the smoke-test harness."""
    parser = argparse.ArgumentParser(description="Smoke-test TutorMind/evaluate_run.py")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve the temporary workspace for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print evaluator stdout/stderr for every subprocess run.",
    )
    return parser.parse_args()


def make_workspace(keep_temp: bool):
    """Create a temporary workspace context.

    Returns an object with a ``name`` attribute that can be used as a context
    manager. When ``keep_temp`` is set, the directory is preserved.
    """

    if not keep_temp:
        return tempfile.TemporaryDirectory(prefix="smoke_eval_run_")

    class _KeptWorkspace:
        def __init__(self) -> None:
            self.name = tempfile.mkdtemp(prefix="smoke_eval_run_")

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    return _KeptWorkspace()


def ensure_evaluator_exists() -> None:
    """Fail early if the target evaluator is missing."""
    if not EVALUATOR_PATH.is_file():
        raise FileNotFoundError(f"Evaluator not found at expected path: {EVALUATOR_PATH}")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[dict]:
    """Read a CSV into a list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_header(path: Path) -> List[str]:
    """Read only the header row from a CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def single_task_record(label: str, idx: int) -> dict:
    """Build a tiny TutorMind-style single-task validation record."""
    return {
        "messages": [
            {"role": "system", "content": "Evaluate the tutor response."},
            {"role": "user", "content": f"Synthetic sample {idx}"},
            {"role": "assistant", "content": label},
        ]
    }


def multitask_record(labels: Dict[str, str], idx: int) -> dict:
    """Build a tiny TutorMind-style multitask validation record."""
    assistant_content = (
        f"Mistake_Identification: {labels['MI']}\n"
        f"Mistake_Location: {labels['ML']}\n"
        f"Providing_Guidance: {labels['PG']}\n"
        f"Actionability: {labels['Act']}"
    )
    return {
        "messages": [
            {"role": "system", "content": "Evaluate the tutor response across four dimensions."},
            {"role": "user", "content": f"Synthetic sample {idx}"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def run_evaluator(
    predictions_path: Path,
    task: str,
    results_path: Path,
    run_id: str,
    validation_flag: str,
    validation_path: Path,
    verbose: bool,
) -> subprocess.CompletedProcess:
    """Invoke evaluate_run.py as a black-box CLI process."""
    cmd = [
        sys.executable,
        str(EVALUATOR_PATH),
        "--predictions",
        str(predictions_path),
        "--task",
        task,
        "--run-id",
        run_id,
        "--model",
        "SmokeModel",
        "--method",
        "SmokeMethod",
        "--aug",
        "None",
        "--think",
        "N/A",
        validation_flag,
        str(validation_path),
        "--out",
        str(results_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if verbose:
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Return code: {result.returncode}")
        print("  STDOUT:")
        print(indent_text(result.stdout.rstrip()))
        print("  STDERR:")
        print(indent_text(result.stderr.rstrip()))
    return result


def indent_text(text: str) -> str:
    """Indent multi-line subprocess output for readability."""
    if not text:
        return "    <empty>"
    return "\n".join(f"    {line}" for line in text.splitlines())


def combined_output(result: subprocess.CompletedProcess) -> str:
    """Return stdout and stderr as a single searchable string."""
    return "\n".join(part for part in [result.stdout, result.stderr] if part).strip()


def assert_true(condition: bool, message: str) -> None:
    """Raise AssertionError with a clear message when condition is false."""
    if not condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str) -> None:
    """Assert exact equality."""
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_contains(text: str, snippet: str, message: str) -> None:
    """Assert that a substring is present."""
    if snippet not in text:
        raise AssertionError(f"{message}: missing {snippet!r} in {text!r}")


def assert_float_close(actual: str | float, expected: float, message: str, tol: float = 1e-9) -> None:
    """Assert that a numeric field matches the expected value."""
    actual_value = float(actual)
    if not math.isclose(actual_value, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{message}: expected {expected}, got {actual_value}")


def collapse_lenient(label: str) -> str:
    """Collapse strict TutorMind labels into lenient binary labels."""
    return "positive" if label in {"Yes", "To some extent"} else "negative"


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute accuracy for a label sequence."""
    assert_true(len(y_true) == len(y_pred), "Accuracy inputs must have the same length")
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


def f1_for_label(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> float:
    """Compute the F1 score for a single label."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Compute a macro F1 score over the provided label order."""
    return sum(f1_for_label(y_true, y_pred, label) for label in labels) / len(labels)


def make_single_task_validation(path: Path, labels: Sequence[str]) -> None:
    """Write a single-task validation JSONL."""
    write_jsonl(path, [single_task_record(label, idx) for idx, label in enumerate(labels)])


def make_multitask_validation(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Write a multitask validation JSONL."""
    write_jsonl(path, [multitask_record(labels, idx) for idx, labels in enumerate(rows)])


def nonempty_stdout_lines(result: subprocess.CompletedProcess) -> List[str]:
    """Split stdout into non-empty lines."""
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_perfect_single_task_mi(root: Path, verbose: bool) -> None:
    """TEST 1: perfect MI run."""
    test_dir = root / "test_01_perfect_single_task_mi"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    labels = ["Yes", "No", "To some extent"]
    make_single_task_validation(val_path, labels)
    write_csv(pred_path, ["pred_label"], [{"pred_label": label} for label in labels])

    result = run_evaluator(pred_path, "MI", results_path, "T1", "--val-mi", val_path, verbose)
    output = combined_output(result)

    assert_equal(result.returncode, 0, "Evaluator should succeed for a perfect MI run")
    assert_true(results_path.exists(), "Results CSV should be created")

    rows = read_csv_rows(results_path)
    assert_equal(len(rows), 1, "Results CSV should contain exactly one row")
    row = rows[0]

    assert_equal(row["task"], "MI", "Task column should be MI")
    assert_equal(row["metric_dimension"], "MI", "Metric dimension should be MI")
    assert_float_close(row["strict_f1"], 1.0, "Strict F1 should be perfect")
    assert_float_close(row["strict_acc"], 1.0, "Strict accuracy should be perfect")
    assert_float_close(row["lenient_f1"], 1.0, "Lenient F1 should be perfect")
    assert_float_close(row["lenient_acc"], 1.0, "Lenient accuracy should be perfect")
    assert_equal(row["n_unknown"], "0", "Unknown count should be zero")
    assert_equal(row["unknown_row_indices"], "", "Unknown row indices should be empty")

    assert_contains(output, "Run T1", "Stdout should include the run id")
    assert_contains(output, "Task: MI", "Stdout should include the task")
    assert_contains(output, "Strict F1: 1.0000", "Stdout should include strict F1")
    assert_contains(output, "Unknown: 0", "Stdout should include unknown count")


def test_perfect_multitask_mt(root: Path, verbose: bool) -> None:
    """TEST 2: perfect MT run."""
    test_dir = root / "test_02_perfect_multitask_mt"
    val_path = test_dir / "multitask_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    rows = [
        {"MI": "Yes", "ML": "No", "PG": "To some extent", "Act": "Yes"},
        {"MI": "No", "ML": "Yes", "PG": "No", "Act": "To some extent"},
        {"MI": "To some extent", "ML": "To some extent", "PG": "Yes", "Act": "No"},
    ]
    make_multitask_validation(val_path, rows)
    write_csv(
        pred_path,
        ["pred_mi", "pred_ml", "pred_pg", "pred_act"],
        [
            {
                "pred_mi": row["MI"],
                "pred_ml": row["ML"],
                "pred_pg": row["PG"],
                "pred_act": row["Act"],
            }
            for row in rows
        ],
    )

    result = run_evaluator(pred_path, "MT", results_path, "T2", "--val-mt", val_path, verbose)
    output = combined_output(result)

    assert_equal(result.returncode, 0, "Evaluator should succeed for a perfect MT run")
    assert_true(results_path.exists(), "Results CSV should be created")

    result_rows = read_csv_rows(results_path)
    assert_equal(len(result_rows), 4, "MT run should produce exactly four result rows")
    assert_equal({row["task"] for row in result_rows}, {"MT"}, "Task column should be MT in all rows")
    assert_equal(
        {row["metric_dimension"] for row in result_rows},
        {"MI", "ML", "PG", "Act"},
        "Metric dimensions should cover MI, ML, PG, and Act",
    )

    for row in result_rows:
        assert_float_close(row["strict_f1"], 1.0, "Strict F1 should be perfect for all MT dimensions")
        assert_float_close(row["strict_acc"], 1.0, "Strict accuracy should be perfect for all MT dimensions")
        assert_float_close(row["lenient_f1"], 1.0, "Lenient F1 should be perfect for all MT dimensions")
        assert_float_close(row["lenient_acc"], 1.0, "Lenient accuracy should be perfect for all MT dimensions")
        assert_equal(row["n_unknown"], "0", "Unknown count should be zero for all MT dimensions")

    stdout_lines = nonempty_stdout_lines(result)
    assert_equal(len(stdout_lines), 4, "MT run should print four human-readable lines")
    assert_contains(output, "MI Strict F1", "Stdout should include MI-prefixed metrics")
    assert_contains(output, "PG Lenient Acc", "Stdout should include dimension-prefixed lenient metrics")


def test_unknown_exclusion_single_task(root: Path, verbose: bool) -> None:
    """TEST 3: unknown exclusion on a single-task run."""
    test_dir = root / "test_03_unknown_exclusion_single_task"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    labels = ["Yes", "No", "To some extent", "No"]
    predictions = ["Yes", "Maybe", "To some extent", "Yes"]
    make_single_task_validation(val_path, labels)
    write_csv(pred_path, ["pred_label"], [{"pred_label": value} for value in predictions])

    result = run_evaluator(pred_path, "MI", results_path, "T3", "--val-mi", val_path, verbose)

    assert_equal(result.returncode, 0, "Evaluator should succeed when some predictions are Unknown")
    row = read_csv_rows(results_path)[0]

    scored_true = ["Yes", "To some extent", "No"]
    scored_pred = ["Yes", "To some extent", "Yes"]
    expected_strict_f1 = round(macro_f1(scored_true, scored_pred, STRICT_LABELS), 6)
    expected_strict_acc = round(accuracy(scored_true, scored_pred), 6)
    expected_lenient_true = [collapse_lenient(label) for label in scored_true]
    expected_lenient_pred = [collapse_lenient(label) for label in scored_pred]
    expected_lenient_f1 = round(macro_f1(expected_lenient_true, expected_lenient_pred, LENIENT_LABELS), 6)
    expected_lenient_acc = round(accuracy(expected_lenient_true, expected_lenient_pred), 6)

    assert_equal(row["n_total_rows"], "4", "Total rows should reflect all prediction rows")
    assert_equal(row["n_scored_rows"], "3", "Scored rows should exclude the Unknown row")
    assert_equal(row["n_unknown"], "1", "Unknown count should be one")
    assert_float_close(row["unknown_rate"], 0.25, "Unknown rate should be one out of four rows")
    assert_equal(row["unknown_row_indices"], "1", "Unknown row index should be the invalid row")
    assert_float_close(row["strict_f1"], expected_strict_f1, "Strict F1 should use only scored rows")
    assert_float_close(row["strict_acc"], expected_strict_acc, "Strict accuracy should use only scored rows")
    assert_float_close(row["lenient_f1"], expected_lenient_f1, "Lenient F1 should use only scored rows")
    assert_float_close(row["lenient_acc"], expected_lenient_acc, "Lenient accuracy should use only scored rows")


def test_accepted_normalization_variants(root: Path, verbose: bool) -> None:
    """TEST 4: accepted normalization variants."""
    test_dir = root / "test_04_accepted_normalization_variants"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    labels = ["Yes", "No", "To some extent", "Yes", "To some extent"]
    predictions = ["yes", "No", "To Some Extent", "Evaluation: Yes", "\"To some extent\""]
    make_single_task_validation(val_path, labels)
    write_csv(pred_path, ["pred_label"], [{"pred_label": value} for value in predictions])

    result = run_evaluator(pred_path, "MI", results_path, "T4", "--val-mi", val_path, verbose)

    assert_equal(result.returncode, 0, "Evaluator should accept mild normalization variants")
    row = read_csv_rows(results_path)[0]
    assert_equal(row["n_unknown"], "0", "Accepted variants should not become Unknown")
    assert_float_close(row["strict_f1"], 1.0, "All normalized variants should score perfectly")
    assert_float_close(row["lenient_f1"], 1.0, "All normalized variants should score perfectly leniently")


def test_rejected_free_form_outputs(root: Path, verbose: bool) -> None:
    """TEST 5: free-form outputs should become Unknown."""
    test_dir = root / "test_05_rejected_free_form_outputs"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    labels = ["Yes", "No", "To some extent"]
    predictions = ["I think this is yes because...", "Probably no", "To some extent"]
    make_single_task_validation(val_path, labels)
    write_csv(pred_path, ["pred_label"], [{"pred_label": value} for value in predictions])

    result = run_evaluator(pred_path, "MI", results_path, "T5", "--val-mi", val_path, verbose)

    assert_equal(result.returncode, 0, "Evaluator should succeed when some free-form outputs become Unknown")
    row = read_csv_rows(results_path)[0]

    scored_true = ["To some extent"]
    scored_pred = ["To some extent"]
    expected_strict_f1 = round(macro_f1(scored_true, scored_pred, STRICT_LABELS), 6)
    expected_strict_acc = round(accuracy(scored_true, scored_pred), 6)
    expected_lenient_true = [collapse_lenient(label) for label in scored_true]
    expected_lenient_pred = [collapse_lenient(label) for label in scored_pred]
    expected_lenient_f1 = round(macro_f1(expected_lenient_true, expected_lenient_pred, LENIENT_LABELS), 6)
    expected_lenient_acc = round(accuracy(expected_lenient_true, expected_lenient_pred), 6)

    assert_equal(row["n_scored_rows"], "1", "Only one row should remain after Unknown exclusion")
    assert_equal(row["n_unknown"], "2", "Two free-form outputs should become Unknown")
    assert_equal(row["unknown_row_indices"], "0;1", "Unknown row indices should match free-form outputs")
    assert_float_close(row["strict_f1"], expected_strict_f1, "Strict F1 should reflect the scored subset only")
    assert_float_close(row["strict_acc"], expected_strict_acc, "Strict accuracy should reflect the scored subset only")
    assert_float_close(row["lenient_f1"], expected_lenient_f1, "Lenient F1 should reflect the scored subset only")
    assert_float_close(row["lenient_acc"], expected_lenient_acc, "Lenient accuracy should reflect the scored subset only")


def test_row_count_mismatch_fails(root: Path, verbose: bool) -> None:
    """TEST 6: row-count mismatch should fail hard."""
    test_dir = root / "test_06_row_count_mismatch"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    make_single_task_validation(val_path, ["Yes", "No", "To some extent"])
    write_csv(pred_path, ["pred_label"], [{"pred_label": "Yes"}, {"pred_label": "No"}])

    result = run_evaluator(pred_path, "MI", results_path, "T6", "--val-mi", val_path, verbose)
    output = combined_output(result)

    assert_true(result.returncode != 0, "Evaluator should fail on row-count mismatch")
    assert_contains(output, "Row-count mismatch", "Failure output should mention row-count mismatch")
    assert_true(not results_path.exists(), "Results CSV should not be created on failure")


def test_missing_required_prediction_columns_fail(root: Path, verbose: bool) -> None:
    """TEST 7: missing required prediction columns should fail for both single-task and MT."""
    test_dir = root / "test_07_missing_required_prediction_columns"

    single_val = test_dir / "mi_val.jsonl"
    single_pred = test_dir / "single_predictions.csv"
    single_results = test_dir / "single_results.csv"
    make_single_task_validation(single_val, ["Yes", "No"])
    write_csv(single_pred, ["raw_output"], [{"raw_output": "Yes"}, {"raw_output": "No"}])

    single_result = run_evaluator(
        single_pred, "MI", single_results, "T7A", "--val-mi", single_val, verbose
    )
    single_output = combined_output(single_result)
    assert_true(single_result.returncode != 0, "Single-task run should fail when pred_label is missing")
    assert_contains(single_output, "missing required column(s)", "Failure output should mention missing columns")
    assert_contains(single_output, "pred_label", "Failure output should name pred_label")
    assert_true(not single_results.exists(), "Single-task results CSV should not be created on failure")

    mt_val = test_dir / "multitask_val.jsonl"
    mt_pred = test_dir / "mt_predictions.csv"
    mt_results = test_dir / "mt_results.csv"
    make_multitask_validation(
        mt_val,
        [
            {"MI": "Yes", "ML": "No", "PG": "Yes", "Act": "No"},
            {"MI": "No", "ML": "Yes", "PG": "To some extent", "Act": "Yes"},
        ],
    )
    write_csv(
        mt_pred,
        ["pred_mi", "pred_ml", "pred_act"],
        [
            {"pred_mi": "Yes", "pred_ml": "No", "pred_act": "No"},
            {"pred_mi": "No", "pred_ml": "Yes", "pred_act": "Yes"},
        ],
    )

    mt_result = run_evaluator(mt_pred, "MT", mt_results, "T7B", "--val-mt", mt_val, verbose)
    mt_output = combined_output(mt_result)
    assert_true(mt_result.returncode != 0, "MT run should fail when pred_pg is missing")
    assert_contains(mt_output, "missing required column(s)", "Failure output should mention missing MT columns")
    assert_contains(mt_output, "pred_pg", "Failure output should name pred_pg")
    assert_true(not mt_results.exists(), "MT results CSV should not be created on failure")


def test_zero_valid_predictions_fail(root: Path, verbose: bool) -> None:
    """TEST 8: all-Unknown predictions should fail hard."""
    test_dir = root / "test_08_zero_valid_predictions"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    make_single_task_validation(val_path, ["Yes", "No"])
    write_csv(
        pred_path,
        ["pred_label"],
        [{"pred_label": "Maybe"}, {"pred_label": "Probably no"}],
    )

    result = run_evaluator(pred_path, "MI", results_path, "T8", "--val-mi", val_path, verbose)
    output = combined_output(result)

    assert_true(result.returncode != 0, "Evaluator should fail when every prediction is Unknown")
    assert_true(
        "All predictions normalized to Unknown" in output or "zero scored rows" in output,
        f"Failure output should clearly mention zero valid predictions, got: {output!r}",
    )
    assert_true(not results_path.exists(), "Results CSV should not be created when no rows are scorable")


def test_results_csv_append_behavior(root: Path, verbose: bool) -> None:
    """TEST 9: successful runs should append to one results CSV with one header."""
    test_dir = root / "test_09_results_csv_append_behavior"
    results_path = test_dir / "results.csv"

    val_a = test_dir / "mi_val_a.jsonl"
    pred_a = test_dir / "predictions_a.csv"
    make_single_task_validation(val_a, ["Yes", "No"])
    write_csv(pred_a, ["pred_label"], [{"pred_label": "Yes"}, {"pred_label": "No"}])

    val_b = test_dir / "mi_val_b.jsonl"
    pred_b = test_dir / "predictions_b.csv"
    make_single_task_validation(val_b, ["No", "To some extent"])
    write_csv(pred_b, ["pred_label"], [{"pred_label": "No"}, {"pred_label": "To some extent"}])

    result_a = run_evaluator(pred_a, "MI", results_path, "T9A", "--val-mi", val_a, verbose)
    result_b = run_evaluator(pred_b, "MI", results_path, "T9B", "--val-mi", val_b, verbose)

    assert_equal(result_a.returncode, 0, "First append run should succeed")
    assert_equal(result_b.returncode, 0, "Second append run should succeed")

    raw_lines = results_path.read_text(encoding="utf-8").splitlines()
    assert_equal(len(raw_lines), 3, "Results CSV should contain one header and two data rows")
    assert_equal(read_csv_header(results_path), EXPECTED_RESULT_COLUMNS, "Header should match expected schema exactly")

    rows = read_csv_rows(results_path)
    assert_equal(len(rows), 2, "Results CSV should contain two appended data rows")
    assert_equal(rows[0]["run_id"], "T9A", "First appended row should remain first")
    assert_equal(rows[1]["run_id"], "T9B", "Second appended row should appear second")


def test_existing_wrong_header_fails(root: Path, verbose: bool) -> None:
    """TEST 10: an existing results CSV with the wrong header should fail hard."""
    test_dir = root / "test_10_existing_wrong_header"
    val_path = test_dir / "mi_val.jsonl"
    pred_path = test_dir / "predictions.csv"
    results_path = test_dir / "results.csv"

    make_single_task_validation(val_path, ["Yes", "No"])
    write_csv(pred_path, ["pred_label"], [{"pred_label": "Yes"}, {"pred_label": "No"}])
    write_csv(results_path, ["wrong", "header"], [{"wrong": "x", "header": "y"}])

    result = run_evaluator(pred_path, "MI", results_path, "T10", "--val-mi", val_path, verbose)
    output = combined_output(result)

    assert_true(result.returncode != 0, "Evaluator should fail when the existing results header is wrong")
    assert_contains(output, "header does not match expected columns", "Failure output should mention header mismatch")
    raw_lines = results_path.read_text(encoding="utf-8").splitlines()
    assert_equal(len(raw_lines), 2, "Wrong-header file should not be appended to on failure")


TESTS = [
    ("Perfect single-task MI", test_perfect_single_task_mi),
    ("Perfect multitask MT", test_perfect_multitask_mt),
    ("Unknown exclusion on single-task run", test_unknown_exclusion_single_task),
    ("Accepted normalization variants", test_accepted_normalization_variants),
    ("Rejected free-form outputs become Unknown", test_rejected_free_form_outputs),
    ("Row-count mismatch fails hard", test_row_count_mismatch_fails),
    ("Missing required prediction column fails hard", test_missing_required_prediction_columns_fail),
    ("Zero valid predictions fails hard", test_zero_valid_predictions_fail),
    ("Results CSV append behavior", test_results_csv_append_behavior),
    ("Existing results CSV with wrong header fails hard", test_existing_wrong_header_fails),
]


def run_all_tests(root: Path, verbose: bool) -> List[TestOutcome]:
    """Run the smoke-test suite and collect outcomes."""
    outcomes: List[TestOutcome] = []
    for name, test_fn in TESTS:
        try:
            test_fn(root, verbose)
            outcome = TestOutcome(name=name, passed=True)
        except Exception as exc:  # noqa: BLE001 - explicit failure reporting is useful here.
            outcome = TestOutcome(name=name, passed=False, message=str(exc))
        outcomes.append(outcome)
    return outcomes


def print_summary(outcomes: Sequence[TestOutcome]) -> None:
    """Print per-test and overall summaries."""
    passed_count = sum(1 for outcome in outcomes if outcome.passed)
    for outcome in outcomes:
        if outcome.passed:
            print(f"[PASS] {outcome.name}")
        else:
            print(f"[FAIL] {outcome.name}: {outcome.message}")
    print(f"OVERALL: {passed_count}/{len(outcomes)} tests passed")


def main() -> int:
    """Entry point for the smoke-test harness."""
    args = parse_args()
    ensure_evaluator_exists()

    with make_workspace(args.keep_temp) as workspace:
        workspace_path = Path(workspace)
        outcomes = run_all_tests(workspace_path, args.verbose)
        print_summary(outcomes)

        if args.keep_temp:
            print(f"Temporary workspace preserved at: {workspace_path}")

    all_passed = all(outcome.passed for outcome in outcomes)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
