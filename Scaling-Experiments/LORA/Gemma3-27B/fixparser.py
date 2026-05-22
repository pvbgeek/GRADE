#!/usr/bin/env python3

import csv
import argparse


VALID_LABELS = ["Yes", "No", "To some extent"]


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # remove special tokens
    text = text.replace("<end_of_turn>", "")
    text = text.replace("</s>", "")

    # remove prefix like "Evaluation:"
    if ":" in text:
        text = text.split(":", 1)[1]

    return text.strip().lower()


def normalize_label_strict(text: str) -> str:
    text = clean_text(text)

    # IMPORTANT: order matters
    if "to some extent" in text or "some extent" in text:
        return "To some extent"
    if text.startswith("no") or " no" in f" {text} ":
        return "No"
    if text.startswith("yes") or " yes" in f" {text} ":
        return "Yes"

    return "Unknown"


def process_single_task(rows):
    unknowns = 0

    for row in rows:
        raw = row.get("raw_output", "")
        label = normalize_label_strict(raw)

        if label not in VALID_LABELS:
            label = "Unknown"
            unknowns += 1

        # STRICT enforcement (critical)
        row["pred_label"] = label

    return unknowns


def process_multitask(rows):
    unknowns = 0

    for row in rows:
        raw = row.get("raw_output", "")
        text = clean_text(raw)

        # initialize
        row["pred_mi"] = "Unknown"
        row["pred_ml"] = "Unknown"
        row["pred_pg"] = "Unknown"
        row["pred_act"] = "Unknown"

        for line in text.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.lower().replace(" ", "")
            label = normalize_label_strict(val)

            if "mistakeidentification" in key:
                row["pred_mi"] = label
            elif "mistakelocation" in key:
                row["pred_ml"] = label
            elif "providingguidance" in key:
                row["pred_pg"] = label
            elif "actionability" in key:
                row["pred_act"] = label

        if "Unknown" in [row["pred_mi"], row["pred_ml"], row["pred_pg"], row["pred_act"]]:
            unknowns += 1

    return unknowns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "pred_label" in fieldnames:
        print("Detected: Single-task CSV")
        unknowns = process_single_task(rows)
    else:
        print("Detected: Multi-task CSV")
        unknowns = process_multitask(rows)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {args.output}")
    print(f"Unknowns after fix: {unknowns}/{len(rows)} ({unknowns/len(rows)*100:.2f}%)")


if __name__ == "__main__":
    main()