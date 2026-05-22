"""
newdataset-preparation.py
Generates train/val .jsonl files for all 5 tasks from augmented_full_devset.json.

Changes from v2:
  - Added deduplication at jsonl level (same user content + same label)
  - Respects valid_labels field added by cleandata.py
  - Command R+ responses only included in MI files (not ML, PG, Act, Multitask)
  - Skips N/A labels for ML, PG, Act, Multitask
  - Reports per-label class distribution after split

Run cleandata.py first before running this script.
"""

import json
import random
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/augmented_full_devset.json"
TRAIN_DIR   = "data/train"
VAL_DIR     = "data/val"
SEED        = 42
TRAIN_RATIO = 0.8

# Valid label values
VALID_LABELS = {"Yes", "No", "To some extent"}

# ── Label configs ─────────────────────────────────────────────────────────────
LABEL_CONFIGS = {
    "Mistake_Identification": {
        "key": "Mistake_Identification",
        "system": (
            "Classify the tutor's response to the student's answer based on whether "
            "the tutor has identified a mistake. Use the following labels: "
            "'Yes' means the mistake is clearly identified; "
            "'No' means the tutor does not recognize the mistake; "
            "'To some extent' means the tutor suggests a mistake but is unsure. "
            "Respond strictly with exactly one of the following:\n"
            "Evaluation: Yes\n"
            "Evaluation: No\n"
            "Evaluation: To some extent"
        ),
    },
    "Mistake_Location": {
        "key": "Mistake_Location",
        "system": (
            "Classify the tutor's response to the student's answer based on whether "
            "the tutor has correctly located the student's mistake. Use the following labels: "
            "'Yes' means the mistake location is clearly identified; "
            "'No' means the tutor does not locate the mistake; "
            "'To some extent' means the tutor partially locates the mistake. "
            "Respond strictly with exactly one of the following:\n"
            "Evaluation: Yes\n"
            "Evaluation: No\n"
            "Evaluation: To some extent"
        ),
    },
    "Providing_Guidance": {
        "key": "Providing_Guidance",
        "system": (
            "Classify the tutor's response to the student's answer based on whether "
            "the tutor provides useful guidance to help the student correct their mistake. "
            "Use the following labels: "
            "'Yes' means clear and useful guidance is provided; "
            "'No' means no guidance is provided; "
            "'To some extent' means some guidance is provided but it is incomplete or vague. "
            "Respond strictly with exactly one of the following:\n"
            "Evaluation: Yes\n"
            "Evaluation: No\n"
            "Evaluation: To some extent"
        ),
    },
    "Actionability": {
        "key": "Actionability",
        "system": (
            "Classify the tutor's response to the student's answer based on whether "
            "the tutor's response is actionable — i.e., gives the student something concrete to do next. "
            "Use the following labels: "
            "'Yes' means the response is clearly actionable; "
            "'No' means the response is not actionable; "
            "'To some extent' means the response is partially actionable. "
            "Respond strictly with exactly one of the following:\n"
            "Evaluation: Yes\n"
            "Evaluation: No\n"
            "Evaluation: To some extent"
        ),
    },
}

MULTITASK_SYSTEM = (
    "You are an expert educational evaluator. Given a student-tutor math dialogue and a tutor response, "
    "evaluate the tutor response across four pedagogical dimensions.\n\n"
    "Evaluate each dimension independently:\n"
    "1. Mistake_Identification: Does the tutor identify the student's mistake?\n"
    "2. Mistake_Location: Does the tutor locate where the mistake occurred?\n"
    "3. Providing_Guidance: Does the tutor provide useful guidance?\n"
    "4. Actionability: Is the tutor's response actionable?\n\n"
    "For each dimension, use one of: Yes / No / To some extent\n\n"
    "Respond strictly in the following format:\n"
    "Mistake_Identification: Yes\n"
    "Mistake_Location: No\n"
    "Providing_Guidance: To some extent\n"
    "Actionability: Yes\n\n"
    "Pick exactly one value per dimension from: Yes, No, To some extent."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_user_content(conversation_history, tutor_response):
    return f"{conversation_history}\n\nTutor Response: {tutor_response}"

def make_single_task_example(conversation_history, tutor_response, label_value, system_prompt):
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": make_user_content(conversation_history, tutor_response)},
            {"role": "assistant", "content": label_value},
        ]
    }

def make_multitask_example(conversation_history, tutor_response, annotation):
    assistant_content = (
        f"Mistake_Identification: {annotation['Mistake_Identification']}\n"
        f"Mistake_Location: {annotation['Mistake_Location']}\n"
        f"Providing_Guidance: {annotation['Providing_Guidance']}\n"
        f"Actionability: {annotation['Actionability']}"
    )
    return {
        "messages": [
            {"role": "system",    "content": MULTITASK_SYSTEM},
            {"role": "user",      "content": make_user_content(conversation_history, tutor_response)},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def write_jsonl(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples):>5} examples → {path}")

def print_class_distribution(examples, label_name):
    counts = {"Yes": 0, "No": 0, "To some extent": 0}
    for ex in examples:
        label = ex["messages"][2]["content"]
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    dist = " | ".join([f"{k}: {v} ({v/total*100:.1f}%)" for k, v in counts.items()])
    print(f"    {label_name}: {dist}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"  Total dialogues: {len(data)}")

    random.seed(SEED)
    random.shuffle(data)
    split_idx  = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data   = data[split_idx:]
    print(f"  Train dialogues: {len(train_data)} | Val dialogues: {len(val_data)}")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR,   exist_ok=True)

    single_task_train = {label: [] for label in LABEL_CONFIGS}
    single_task_val   = {label: [] for label in LABEL_CONFIGS}
    multitask_train   = []
    multitask_val     = []

    skipped_na    = {label: 0 for label in LABEL_CONFIGS}
    skipped_na["Multitask"] = 0

    skipped_dupes = {label: 0 for label in LABEL_CONFIGS}
    skipped_dupes["Multitask"] = 0

    # Reset seen keys per split to avoid cross-split contamination
    for split_name, split_data, single_dict, multi_list in [
        ("train", train_data, single_task_train, multitask_train),
        ("val",   val_data,   single_task_val,   multitask_val),
    ]:
        seen_keys = {label: set() for label in LABEL_CONFIGS}
        seen_keys["Multitask"] = set()

        for dialogue in split_data:
            conv_history = dialogue["conversation_history"]

            for model_name, response_data in dialogue["tutor_responses"].items():
                tutor_response = response_data["response"]
                annotation     = response_data["annotation"]
                valid_labels   = response_data.get("valid_labels", list(LABEL_CONFIGS.keys()))
                user_content   = make_user_content(conv_history, tutor_response)

                # ── Single-task examples ──────────────────────────────────
                for label_name, config in LABEL_CONFIGS.items():
                    label_key   = config["key"]
                    label_value = annotation[label_key]

                    if label_key not in valid_labels:
                        skipped_na[label_name] += 1
                        continue

                    if label_value not in VALID_LABELS:
                        skipped_na[label_name] += 1
                        continue

                    dedup_key = (user_content, label_value)
                    if dedup_key in seen_keys[label_name]:
                        skipped_dupes[label_name] += 1
                        continue
                    seen_keys[label_name].add(dedup_key)

                    ex = make_single_task_example(
                        conv_history, tutor_response,
                        label_value, config["system"]
                    )
                    single_dict[label_name].append(ex)

                # ── Multitask example ─────────────────────────────────────
                all_labels_valid = all(
                    lk in valid_labels and annotation[lk] in VALID_LABELS
                    for lk in LABEL_CONFIGS.keys()
                )
                if all_labels_valid:
                    mt_label = (
                        annotation["Mistake_Identification"],
                        annotation["Mistake_Location"],
                        annotation["Providing_Guidance"],
                        annotation["Actionability"],
                    )
                    dedup_key = (user_content, mt_label)
                    if dedup_key in seen_keys["Multitask"]:
                        skipped_dupes["Multitask"] += 1
                        continue
                    seen_keys["Multitask"].add(dedup_key)

                    multi_list.append(
                        make_multitask_example(conv_history, tutor_response, annotation)
                    )
                else:
                    skipped_na["Multitask"] += 1

    # ── Write files ───────────────────────────────────────────────────────────
    print("\nWriting single-task train files:")
    for label_name, examples in single_task_train.items():
        fname = f"{label_name.lower()}_train.jsonl"
        write_jsonl(examples, os.path.join(TRAIN_DIR, fname))

    print("\nWriting single-task val files:")
    for label_name, examples in single_task_val.items():
        fname = f"{label_name.lower()}_val.jsonl"
        write_jsonl(examples, os.path.join(VAL_DIR, fname))

    print("\nWriting multitask files:")
    write_jsonl(multitask_train, os.path.join(TRAIN_DIR, "multitask_train.jsonl"))
    write_jsonl(multitask_val,   os.path.join(VAL_DIR,   "multitask_val.jsonl"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nSkipped (N/A or invalid label):")
    for label_name, count in skipped_na.items():
        print(f"  {label_name:<30}: {count} skipped")

    print("\nSkipped (duplicates):")
    for label_name, count in skipped_dupes.items():
        print(f"  {label_name:<30}: {count} skipped")

    print("\nTrain class distribution:")
    for label_name, examples in single_task_train.items():
        print_class_distribution(examples, label_name)

    print("\nVal class distribution:")
    for label_name, examples in single_task_val.items():
        print_class_distribution(examples, label_name)

    print("\n" + "="*55)
    print("DONE! Final structure:")
    print(f"  data/train/")
    for f in sorted(os.listdir(TRAIN_DIR)):
        print(f"    {f}")
    print(f"  data/val/")
    for f in sorted(os.listdir(VAL_DIR)):
        print(f"    {f}")
    print("="*55)

if __name__ == "__main__":
    main()