"""
cleandata.py
Cleans augmented_full_devset.json in place by:
  1. Removing true duplicate rows (same conversation_id + same model)
  2. Flagging Command R+ augmented responses as MI-only valid
     (TSE-Command-R+ and No-Command-R+ have N/A for ML, PG, Act labels)
  3. Keeping Command R+ responses for MI only

NOTE on duplicates:
  A true duplicate = same conversation_id + same model appearing more than once.
  We do NOT deduplicate based on response text alone because models like Phi3
  gave identical generic responses across different conversations — those are
  NOT duplicates, just low quality responses.

Overwrites the same file with clean version.
Run this BEFORE newdataset-preparation.py
"""

import json

DATA_PATH = "data/augmented_full_devset.json"

# Models to flag as MI-only (have N/A for ML, PG, Act labels)
COMMANDR_MODELS = {"TSE-Command-R+", "No-Command-R+"}


def clean_dataset():
    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"  Original dialogues : {len(data)}")

    total_responses_before = sum(len(d["tutor_responses"]) for d in data)
    print(f"  Original responses : {total_responses_before}")

    # ── Step 1: Remove true duplicates (same conversation_id + same model) ──
    dupes_removed = 0

    for dialogue in data:
        seen_models = set()
        clean_responses = {}
        for model, r in dialogue["tutor_responses"].items():
            key = (dialogue["conversation_id"], model)
            if key not in seen_models:
                seen_models.add(key)
                clean_responses[model] = r
            else:
                dupes_removed += 1
                print(f"  Removed duplicate: conversation_id={dialogue['conversation_id']} model={model}")
        dialogue["tutor_responses"] = clean_responses

    print(f"\nStep 1 — True duplicates removed : {dupes_removed}")

    # ── Step 2: Flag Command R+ responses as MI-only ─────────────────────────
    commandr_flagged = 0
    all_labels = [
        "Mistake_Identification",
        "Mistake_Location",
        "Providing_Guidance",
        "Actionability"
    ]

    for dialogue in data:
        for model, r in dialogue["tutor_responses"].items():
            if model in COMMANDR_MODELS:
                annotation = r["annotation"]
                if (annotation["Mistake_Location"] == "N/A" and
                        annotation["Providing_Guidance"] == "N/A" and
                        annotation["Actionability"] == "N/A"):
                    r["valid_labels"] = ["Mistake_Identification"]
                    commandr_flagged += 1
                else:
                    # Unexpected — Command R+ has all labels, keep all
                    r["valid_labels"] = all_labels
            else:
                r["valid_labels"] = all_labels

    print(f"Step 2 — Command R+ responses flagged (MI only) : {commandr_flagged}")

    # ── Step 3: Remove dialogues with no responses left ──────────────────────
    data = [d for d in data if len(d["tutor_responses"]) > 0]

    total_responses_after = sum(len(d["tutor_responses"]) for d in data)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*55}")
    print(f"  Dialogues                       : {len(data)}")
    print(f"  True duplicates removed         : {dupes_removed}")
    print(f"  Command R+ flagged (MI only)    : {commandr_flagged}")
    print(f"  Total responses before          : {total_responses_before}")
    print(f"  Total responses after           : {total_responses_after}")
    print(f"  Responses removed               : {total_responses_before - total_responses_after}")
    print(f"{'='*55}")

    # ── Label distribution after cleaning ────────────────────────────────────
    print(f"\nLabel distribution after cleaning:")
    for label in all_labels:
        counts = {"Yes": 0, "No": 0, "To some extent": 0, "N/A": 0}
        for d in data:
            for model, r in d["tutor_responses"].items():
                val = r["annotation"][label]
                if val in counts:
                    counts[val] += 1
                else:
                    counts["N/A"] += 1
        total = sum(counts.values())
        print(f"  {label}:")
        for k, v in counts.items():
            pct = (v / total * 100) if total > 0 else 0
            print(f"    {k:<20}: {v:>5} ({pct:.1f}%)")

    # ── Save (overwrite same file) ────────────────────────────────────────────
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Clean data saved to {DATA_PATH}")
    print(f"Now run: python3 newdataset-preparation.py")


if __name__ == "__main__":
    clean_dataset()