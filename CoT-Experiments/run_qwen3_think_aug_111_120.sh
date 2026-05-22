#!/bin/bash
#SBATCH -J qwen3_aug_111_120
#SBATCH -p oignat_lab
#SBATCH -w oignat01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:10
#SBATCH -o /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/CoT-Experiments/logs/qwen3_aug_111_120-%j.out

set -euo pipefail

DRY_RUN=0
RESUME=0
START_RUN=""
RUN_STAMP_PROVIDED=0
if [[ -n "${RUN_STAMP+x}" ]]; then
  RUN_STAMP_PROVIDED=1
fi
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
Usage:
  bash CoT-Experiments/run_qwen3_think_aug_111_120.sh [--dry-run] [--resume --start-run RUN_ID]

Options:
  --dry-run          Validate references and print the corrected 10-run command matrix.
  --resume           Resume an existing RUN_STAMP. Requires RUN_STAMP to be set.
  --start-run RUN_ID Skip runs with run_id lower than RUN_ID. Requires --resume.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --start-run)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --start-run" >&2
        exit 2
      fi
      START_RUN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind"
MODEL_PATH="/WAVE/datasets/oignat_lab/QWEN3"
ENV_PATH="/WAVE/projects/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/activate"

LORA_TRAIN_SCRIPT="$ROOT/CoT-Experiments/LORA/Qwen3-14B/train_qwen_lora_cot.py"
LORA_INFER_SCRIPT="$ROOT/CoT-Experiments/LORA/Qwen3-14B/infer_qwen_lora_cot.py"
EVAL_SCRIPT="$ROOT/evaluate_run.py"
PROMPTS_JSON="$ROOT/prompts.json"
MASTER_METRICS="$ROOT/master_metrics.csv"

LORA_GEN_ROOT="$ROOT/CoT-Experiments/LORA/Qwen3_14B_Gen"
LORA_GV_ROOT="$ROOT/CoT-Experiments/LORA/Qwen3_14B_GenVerify"

TRAIN_GEN_MI="$ROOT/data/train/Gen/mistake_identification_train_aug_qwen3_gen500.jsonl"
TRAIN_GEN_ML="$ROOT/data/train/Gen/mistake_location_train_aug_qwen3_gen500.jsonl"
TRAIN_GEN_PG="$ROOT/data/train/Gen/providing_guidance_train_aug_qwen3_gen500.jsonl"
TRAIN_GEN_ACT="$ROOT/data/train/Gen/actionability_train_aug_qwen3_gen500.jsonl"
TRAIN_GEN_MT="$ROOT/data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl"

TRAIN_GV_MI="$ROOT/data/train/Gen+Verify/mistake_identification_train_aug_qwen3_genverify500.jsonl"
TRAIN_GV_ML="$ROOT/data/train/Gen+Verify/mistake_location_train_aug_qwen3_genverify500.jsonl"
TRAIN_GV_PG="$ROOT/data/train/Gen+Verify/providing_guidance_train_aug_qwen3_genverify500.jsonl"
TRAIN_GV_ACT="$ROOT/data/train/Gen+Verify/actionability_train_aug_qwen3_genverify500.jsonl"
TRAIN_GV_MT="$ROOT/data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl"

VAL_MI="$ROOT/data/val/mistake_identification_val.jsonl"
VAL_ML="$ROOT/data/val/mistake_location_val.jsonl"
VAL_PG="$ROOT/data/val/providing_guidance_val.jsonl"
VAL_ACT="$ROOT/data/val/actionability_val.jsonl"
VAL_MT="$ROOT/data/val/multitask_val.jsonl"

DELETE_LORA_ADAPTERS_AFTER_EVAL=0

print_shell_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

verify_contains() {
  local path="$1"
  local needle="$2"
  if ! grep -q "$needle" "$path"; then
    echo "Thinking contract check failed: $path does not contain '$needle'" >&2
    exit 1
  fi
}

count_jsonl_rows() {
  local file="$1"
  awk 'END { print NR }' "$file"
}

count_csv_data_rows() {
  local csv_file="$1"
  python - "$csv_file" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing csv: {path}")

with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, strict=True)
    if reader.fieldnames is None:
        raise SystemExit(f"csv has no header: {path}")
    try:
        print(sum(1 for _ in reader))
    except csv.Error as exc:
        raise SystemExit(f"malformed csv: {path}: {exc}") from exc
PY
}

csv_has_expected_rows() {
  local csv="$1"
  local expected_rows="$2"
  local actual_rows
  actual_rows="$(count_csv_data_rows "$csv")"
  [[ "$actual_rows" -eq "$expected_rows" ]]
}

validate_prediction_csv() {
  local csv_file="$1"
  local task="$2"
  local expected_rows="$3"

  python - "$csv_file" "$task" "$expected_rows" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
try:
    expected_rows = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"expected row count must be an integer: {sys.argv[3]!r}") from exc

if not path.is_file():
    raise SystemExit(f"missing prediction csv: {path}")

if task == "MT":
    required_columns = ["pred_mi", "pred_ml", "pred_pg", "pred_act"]
elif task in {"MI", "ML", "PG", "Act"}:
    required_columns = ["pred_label"]
else:
    raise SystemExit(f"unknown task for prediction csv validation: {task}")

with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, strict=True)
    if reader.fieldnames is None:
        raise SystemExit(f"prediction csv has no header: {path}")

    missing = [column for column in required_columns if column not in reader.fieldnames]
    if missing:
        raise SystemExit(
            f"prediction csv is missing required column(s) for task {task}: "
            f"{', '.join(missing)}; found columns: {', '.join(reader.fieldnames)}"
        )

    try:
        actual_rows = sum(1 for _ in reader)
    except csv.Error as exc:
        raise SystemExit(f"malformed prediction csv: {path}: {exc}") from exc

if actual_rows != expected_rows:
    raise SystemExit(
        f"prediction csv has wrong logical data-row count: {path}\n"
        f"Expected: {expected_rows}; found: {actual_rows}"
    )
PY
}

adapter_complete() {
  local adapter_path="$1"
  [[ -d "$adapter_path" ]] || return 1
  [[ -f "$adapter_path/adapter_config.json" ]] || return 1
  [[ -f "$adapter_path/adapter_model.safetensors" || -f "$adapter_path/adapter_model.bin" ]] || return 1
}

should_run() {
  local run_id="$1"
  if [[ -n "$START_RUN" && "$run_id" -lt "$START_RUN" ]]; then
    echo "Skipping run $run_id because it is before --start-run $START_RUN"
    return 1
  fi
  return 0
}

ensure_resume_stamp() {
  if [[ "$RESUME" -eq 1 && "$RUN_STAMP_PROVIDED" -ne 1 ]]; then
    echo "--resume requires RUN_STAMP to be explicitly set." >&2
    echo "Example: RUN_STAMP=20260506_183000 bash CoT-Experiments/run_qwen3_think_aug_111_120.sh --resume --start-run 116" >&2
    exit 2
  fi
}

validate_cli() {
  ensure_resume_stamp

  if [[ -n "$START_RUN" ]]; then
    if [[ "$RESUME" -ne 1 ]]; then
      echo "--start-run is only supported with --resume; non-resume mode runs all corrected runs 111-120." >&2
      exit 2
    fi
    if ! [[ "$START_RUN" =~ ^[0-9]+$ ]]; then
      echo "--start-run must be numeric: $START_RUN" >&2
      exit 2
    fi
    if [[ "$START_RUN" -lt 111 || "$START_RUN" -gt 120 ]]; then
      echo "--start-run must be between 111 and 120 for this batch: $START_RUN" >&2
      exit 2
    fi
  fi
}

preflight() {
  require_dir "$ROOT"
  require_dir "$MODEL_PATH"
  require_file "$ENV_PATH"
  require_file "$LORA_TRAIN_SCRIPT"
  require_file "$LORA_INFER_SCRIPT"
  require_file "$EVAL_SCRIPT"
  require_file "$PROMPTS_JSON"

  require_file "$TRAIN_GEN_MI"
  require_file "$TRAIN_GEN_ML"
  require_file "$TRAIN_GEN_PG"
  require_file "$TRAIN_GEN_ACT"
  require_file "$TRAIN_GEN_MT"
  require_file "$TRAIN_GV_MI"
  require_file "$TRAIN_GV_ML"
  require_file "$TRAIN_GV_PG"
  require_file "$TRAIN_GV_ACT"
  require_file "$TRAIN_GV_MT"

  require_file "$VAL_MI"
  require_file "$VAL_ML"
  require_file "$VAL_PG"
  require_file "$VAL_ACT"
  require_file "$VAL_MT"

  verify_contains "$LORA_TRAIN_SCRIPT" "enable_thinking=True"
  verify_contains "$LORA_TRAIN_SCRIPT" "single_task_thinking"
  verify_contains "$LORA_TRAIN_SCRIPT" "multitask_thinking"
  verify_contains "$LORA_INFER_SCRIPT" "enable_thinking=True"
  verify_contains "$LORA_INFER_SCRIPT" "single_task_thinking"
  verify_contains "$LORA_INFER_SCRIPT" "multitask_thinking"
}

prepare_output_dirs() {
  mkdir -p "$ROOT/CoT-Experiments/logs"

  local timestamp_dirs=(
    "$LORA_GEN_ROOT/outputs/$RUN_STAMP"
    "$LORA_GEN_ROOT/adapters/$RUN_STAMP"
    "$LORA_GEN_ROOT/metrics/$RUN_STAMP"
    "$LORA_GEN_ROOT/logs/$RUN_STAMP"
    "$LORA_GV_ROOT/outputs/$RUN_STAMP"
    "$LORA_GV_ROOT/adapters/$RUN_STAMP"
    "$LORA_GV_ROOT/metrics/$RUN_STAMP"
    "$LORA_GV_ROOT/logs/$RUN_STAMP"
  )

  local dir
  if [[ "$RESUME" -eq 0 ]]; then
    for dir in "${timestamp_dirs[@]}"; do
      if [[ -e "$dir" ]]; then
        echo "Refusing to reuse existing timestamped output directory: $dir" >&2
        echo "Use a fresh RUN_STAMP or run with --resume and an explicit RUN_STAMP." >&2
        exit 1
      fi
    done
  fi

  mkdir -p "${timestamp_dirs[@]}"
}

run_logged() {
  local log_file="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_shell_command "$@"
  else
    "$@" 2>&1 | tee "$log_file"
  fi
}

maybe_skip_training() {
  local adapter_path="$1"
  local train_log="$2"
  shift 2

  if adapter_complete "$adapter_path"; then
    echo "Skipping training because complete adapter already exists: $adapter_path"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: training would run because no complete adapter exists: $adapter_path"
    run_logged "$train_log" "$@"
    return 0
  fi

  if [[ -e "$adapter_path" ]]; then
    echo "Adapter path exists but looks incomplete: $adapter_path" >&2
    echo "Expected adapter_config.json and adapter_model.safetensors or adapter_model.bin." >&2
    exit 1
  fi

  run_logged "$train_log" "$@"

  if ! adapter_complete "$adapter_path"; then
    echo "Training finished, but adapter path is incomplete: $adapter_path" >&2
    exit 1
  fi
}

maybe_skip_inference() {
  local pred_csv="$1"
  local task="$2"
  local expected_rows="$3"
  local infer_log="$4"
  shift 4

  if [[ -f "$pred_csv" ]]; then
    validate_prediction_csv "$pred_csv" "$task" "$expected_rows"
    echo "Skipping inference because prediction CSV already has $expected_rows logical data rows: $pred_csv"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: inference would run because prediction CSV is missing: $pred_csv"
    run_logged "$infer_log" "$@"
    return 0
  fi

  run_logged "$infer_log" "$@"
  validate_prediction_csv "$pred_csv" "$task" "$expected_rows"
}

maybe_skip_evaluation() {
  local metrics_csv="$1"
  local eval_log="$2"
  shift 2

  if [[ -s "$metrics_csv" ]]; then
    echo "Skipping evaluation because metrics CSV is nonempty: $metrics_csv"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: evaluation would run because metrics CSV is missing or empty: $metrics_csv"
    run_logged "$eval_log" "$@"
    return 0
  fi

  run_logged "$eval_log" "$@"
}

evaluate_prediction() {
  local run_id="$1"
  local task="$2"
  local aug_label="$3"
  local val_file="$4"
  local eval_flag="$5"
  local pred_csv="$6"
  local metrics_csv="$7"
  local eval_log="$8"

  local eval_cmd=(
    python "$EVAL_SCRIPT"
    --predictions "$pred_csv"
    --task "$task"
    --run-id "$run_id"
    --model Qwen3-14B
    --method LoRA
    --aug "$aug_label"
    --think ON
    "$eval_flag" "$val_file"
    --out "$metrics_csv"
  )
  maybe_skip_evaluation "$metrics_csv" "$eval_log" "${eval_cmd[@]}"
}

run_lora() {
  local run_id="$1"
  local task="$2"
  local aug_label="$3"
  local aug_slug="$4"
  local task_slug="$5"
  local train_file="$6"
  local val_file="$7"
  local eval_flag="$8"

  if ! should_run "$run_id"; then
    return 0
  fi

  local result_root
  if [[ "$aug_slug" == "gen" ]]; then
    result_root="$LORA_GEN_ROOT"
  elif [[ "$aug_slug" == "genverify" ]]; then
    result_root="$LORA_GV_ROOT"
  else
    echo "Unknown LoRA augmentation slug: $aug_slug" >&2
    exit 1
  fi

  local adapter_root="$result_root/adapters/$RUN_STAMP/run_${run_id}_${aug_slug}_${task_slug}"
  local adapter_path="$adapter_root/$task"
  local pred_csv="$result_root/outputs/$RUN_STAMP/run_${run_id}_qwen3_14b_think_lora_${aug_slug}_${task_slug}_predictions.csv"
  local metrics_csv="$result_root/metrics/$RUN_STAMP/run_${run_id}_qwen3_14b_think_lora_${aug_slug}_${task_slug}_metrics.csv"
  local train_log="$result_root/logs/$RUN_STAMP/run_${run_id}_lora_${aug_slug}_${task_slug}_train.log"
  local infer_log="$result_root/logs/$RUN_STAMP/run_${run_id}_lora_${aug_slug}_${task_slug}_infer.log"
  local eval_log="$result_root/logs/$RUN_STAMP/run_${run_id}_lora_${aug_slug}_${task_slug}_eval.log"
  local expected_rows
  expected_rows="$(count_jsonl_rows "$val_file")"

  echo "RUN $run_id | Qwen3-14B | LoRA | $task | Aug: $aug_label | Think: ON"
  echo "  train: $train_file"
  echo "  val:   $val_file"
  echo "  expected prediction rows: $expected_rows"
  echo "  adapter: $adapter_path"
  echo "  pred:  $pred_csv"
  echo "  eval:  $metrics_csv"

  local train_cmd=(
    python "$LORA_TRAIN_SCRIPT"
    --task "$task"
    --train-jsonl "$train_file"
    --adapter-out "$adapter_root"
    --epochs 3
    --learning-rate 2e-4
    --batch-size 2
    --grad-accum 8
    --max-seq-length 2048
    --seed 42
  )
  maybe_skip_training "$adapter_path" "$train_log" "${train_cmd[@]}"

  local infer_cmd=(
    python "$LORA_INFER_SCRIPT"
    --task "$task"
    --adapter-path "$adapter_path"
    --input-jsonl "$val_file"
    --predictions-out "$pred_csv"
  )
  maybe_skip_inference "$pred_csv" "$task" "$expected_rows" "$infer_log" "${infer_cmd[@]}"

  evaluate_prediction "$run_id" "$task" "$aug_label" "$val_file" "$eval_flag" "$pred_csv" "$metrics_csv" "$eval_log"
  echo
}

validate_cli
preflight

echo "===== QWEN3 THINK LORA AUG RUNS 111-120 ====="
echo "ROOT: $ROOT"
echo "MODEL_PATH: $MODEL_PATH"
echo "RUN_STAMP: $RUN_STAMP"
echo "LORA_GEN_ROOT: $LORA_GEN_ROOT"
echo "LORA_GV_ROOT: $LORA_GV_ROOT"
echo "MASTER_METRICS: $MASTER_METRICS"
echo "DRY_RUN: $DRY_RUN"
echo "RESUME: $RESUME"
echo "START_RUN: ${START_RUN:-<none>}"
echo "DELETE_LORA_ADAPTERS_AFTER_EVAL: $DELETE_LORA_ADAPTERS_AFTER_EVAL"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  prepare_output_dirs
  source "$ENV_PATH"
  cd "$ROOT"
  which python
  python --version
else
  mkdir -p "$ROOT/CoT-Experiments/logs"
fi

run_lora 111 MI  "Qwen3-Gen"        gen       mi  "$TRAIN_GEN_MI"  "$VAL_MI"  --val-mi
run_lora 112 ML  "Qwen3-Gen"        gen       ml  "$TRAIN_GEN_ML"  "$VAL_ML"  --val-ml
run_lora 113 PG  "Qwen3-Gen"        gen       pg  "$TRAIN_GEN_PG"  "$VAL_PG"  --val-pg
run_lora 114 Act "Qwen3-Gen"        gen       act "$TRAIN_GEN_ACT" "$VAL_ACT" --val-act
run_lora 115 MT  "Qwen3-Gen"        gen       mt  "$TRAIN_GEN_MT"  "$VAL_MT"  --val-mt

run_lora 116 MI  "Qwen3-Gen+Verify" genverify mi  "$TRAIN_GV_MI"  "$VAL_MI"  --val-mi
run_lora 117 ML  "Qwen3-Gen+Verify" genverify ml  "$TRAIN_GV_ML"  "$VAL_ML"  --val-ml
run_lora 118 PG  "Qwen3-Gen+Verify" genverify pg  "$TRAIN_GV_PG"  "$VAL_PG"  --val-pg
run_lora 119 Act "Qwen3-Gen+Verify" genverify act "$TRAIN_GV_ACT" "$VAL_ACT" --val-act
run_lora 120 MT  "Qwen3-Gen+Verify" genverify mt  "$TRAIN_GV_MT"  "$VAL_MT"  --val-mt

echo "===== QWEN3 THINK LORA AUG RUNS 111-120 COMPLETE ====="
echo "LoRA Gen outputs:        $LORA_GEN_ROOT/outputs/$RUN_STAMP"
echo "LoRA GenVerify outputs:  $LORA_GV_ROOT/outputs/$RUN_STAMP"
echo "LoRA Gen adapters:       $LORA_GEN_ROOT/adapters/$RUN_STAMP"
echo "LoRA GenVerify adapters: $LORA_GV_ROOT/adapters/$RUN_STAMP"
echo "LoRA Gen metrics:        $LORA_GEN_ROOT/metrics/$RUN_STAMP"
echo "LoRA GenVerify metrics:  $LORA_GV_ROOT/metrics/$RUN_STAMP"
echo "Master metrics: $MASTER_METRICS"
