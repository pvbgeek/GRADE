#!/bin/bash
#SBATCH -J mistral_baseline
#SBATCH -p oignat_lab
#SBATCH -w oignat01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=11:50:10
#SBATCH -o /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/slurm-mistral-%j.out

set -euo pipefail

# ==========================================
# IMPORTANT
# ==========================================
# evaluate_run.py writes per-run metrics to --out and also upserts the same
# rows into the master CSV at:
# /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/master_metrics.csv
# ==========================================

source /WAVE/projects/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/activate

ROOT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind"

ZS_DIR="$ROOT/Scaling-Experiments/ZeroShot/Mistral-7B"
LORA_DIR="$ROOT/Scaling-Experiments/LORA/Mistral-7B"
EVAL_DIR="$ROOT/Scaling-Experiments/EVAL/Mistral-7B"

ZS_SCRIPT="$ZS_DIR/zeroshot_mistral.py"
LORA_TRAIN_SCRIPT="$LORA_DIR/train_mistral_lora.py"
LORA_INFER_SCRIPT="$LORA_DIR/infer_mistral_lora.py"
EVAL_SCRIPT="$ROOT/evaluate_run.py"

MASTER_METRICS="$ROOT/master_metrics.csv"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/Scaling-Experiments/logs/Mistral-7B/$RUN_STAMP"
ZS_OUT_DIR="$ZS_DIR/outputs/$RUN_STAMP"
LORA_OUT_DIR="$LORA_DIR/outputs/$RUN_STAMP"
LORA_ADAPTER_BASE="$LORA_DIR/adapters/$RUN_STAMP"

mkdir -p "$LOG_DIR" "$ZS_OUT_DIR" "$LORA_OUT_DIR" "$LORA_ADAPTER_BASE" "$EVAL_DIR"

# Set to 0 if you want to keep all LoRA adapters.
DELETE_LORA_ADAPTERS_AFTER_EVAL=1

echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "ROOT: $ROOT"
echo "RUN_STAMP: $RUN_STAMP"
echo "LOG_DIR: $LOG_DIR"
echo "ZS_OUT_DIR: $ZS_OUT_DIR"
echo "LORA_OUT_DIR: $LORA_OUT_DIR"
echo "LORA_ADAPTER_BASE: $LORA_ADAPTER_BASE"
echo "MASTER_METRICS (upsert target in evaluate_run.py): $MASTER_METRICS"
echo

which python
python --version

python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0 =", torch.cuda.get_device_name(0))
PY

echo
nvidia-smi || true
echo

# ==========================================
# EXACT DATASET PATHS ONLY
# DO NOT CHANGE / DO NOT DISCOVER ALTERNATIVES
# ==========================================
TRAIN_MI="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/mistake_identification_train.jsonl"
TRAIN_ML="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/mistake_location_train.jsonl"
TRAIN_PG="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/providing_guidance_train.jsonl"
TRAIN_ACT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/actionability_train.jsonl"
TRAIN_MT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/multitask_train.jsonl"

VAL_MI="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_identification_val.jsonl"
VAL_ML="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/mistake_location_val.jsonl"
VAL_PG="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/providing_guidance_val.jsonl"
VAL_ACT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/actionability_val.jsonl"
VAL_MT="/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/val/multitask_val.jsonl"

# ==========================================
# Validate required files
# ==========================================
REQUIRED_FILES=(
  "$ZS_SCRIPT"
  "$LORA_TRAIN_SCRIPT"
  "$LORA_INFER_SCRIPT"
  "$EVAL_SCRIPT"
  "$TRAIN_MI" "$TRAIN_ML" "$TRAIN_PG" "$TRAIN_ACT" "$TRAIN_MT"
  "$VAL_MI" "$VAL_ML" "$VAL_PG" "$VAL_ACT" "$VAL_MT"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

echo "===== FIXED DATA PATHS ====="
echo "TRAIN_MI=$TRAIN_MI"
echo "TRAIN_ML=$TRAIN_ML"
echo "TRAIN_PG=$TRAIN_PG"
echo "TRAIN_ACT=$TRAIN_ACT"
echo "TRAIN_MT=$TRAIN_MT"
echo
echo "VAL_MI=$VAL_MI"
echo "VAL_ML=$VAL_ML"
echo "VAL_PG=$VAL_PG"
echo "VAL_ACT=$VAL_ACT"
echo "VAL_MT=$VAL_MT"
echo

run_zero_shot() {
  local run_id="$1"
  local task="$2"
  local val_file="$3"
  local eval_flag="$4"
  local tag="$5"

  local pred_csv="$ZS_OUT_DIR/run_${run_id}_mistral7b_zeroshot_${tag}_predictions.csv"
  local metrics_csv="$EVAL_DIR/run_${run_id}_mistral7b_zeroshot_${tag}_metrics.csv"

  local infer_log="$LOG_DIR/run_${run_id}_zeroshot_${tag}.log"
  local eval_log="$LOG_DIR/run_${run_id}_zeroshot_${tag}_eval.log"

  echo "===== RUN ${run_id} | Zero-shot | ${task} ====="

  python "$ZS_SCRIPT" \
    --task "$task" \
    --input-jsonl "$val_file" \
    --predictions-out "$pred_csv" \
    2>&1 | tee "$infer_log"

  python "$EVAL_SCRIPT" \
    --predictions "$pred_csv" \
    --task "$task" \
    --run-id "$run_id" \
    --model Mistral-7B \
    --method Zero-shot \
    --aug None \
    --think N/A \
    "$eval_flag" "$val_file" \
    --out "$metrics_csv" \
    2>&1 | tee "$eval_log"

  echo "Completed run ${run_id}"
  echo "Predictions: $pred_csv"
  echo "Metrics:     $metrics_csv"
  echo "Logs:        $infer_log | $eval_log"
  echo
}

run_lora() {
  local run_id="$1"
  local task="$2"
  local train_file="$3"
  local val_file="$4"
  local eval_flag="$5"
  local tag="$6"

  local adapter_root="$LORA_ADAPTER_BASE/run_${run_id}"
  local adapter_path="$adapter_root/$task"

  local pred_csv="$LORA_OUT_DIR/run_${run_id}_mistral7b_lora_${tag}_predictions.csv"
  local metrics_csv="$EVAL_DIR/run_${run_id}_mistral7b_lora_${tag}_metrics.csv"

  local train_log="$LOG_DIR/run_${run_id}_lora_${tag}_train.log"
  local infer_log="$LOG_DIR/run_${run_id}_lora_${tag}_infer.log"
  local eval_log="$LOG_DIR/run_${run_id}_lora_${tag}_eval.log"

  echo "===== RUN ${run_id} | LoRA | ${task} ====="

  python "$LORA_TRAIN_SCRIPT" \
    --task "$task" \
    --train-jsonl "$train_file" \
    --adapter-out "$adapter_root" \
    --epochs 3 \
    --learning-rate 2e-4 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-seq-length 2048 \
    --seed 42 \
    2>&1 | tee "$train_log"

  if [[ ! -d "$adapter_path" ]]; then
    echo "Expected adapter path not found: $adapter_path" >&2
    exit 1
  fi

  python "$LORA_INFER_SCRIPT" \
    --task "$task" \
    --adapter-path "$adapter_path" \
    --input-jsonl "$val_file" \
    --predictions-out "$pred_csv" \
    2>&1 | tee "$infer_log"

  python "$EVAL_SCRIPT" \
    --predictions "$pred_csv" \
    --task "$task" \
    --run-id "$run_id" \
    --model Mistral-7B \
    --method LoRA \
    --aug None \
    --think N/A \
    "$eval_flag" "$val_file" \
    --out "$metrics_csv" \
    2>&1 | tee "$eval_log"

  if [[ "$DELETE_LORA_ADAPTERS_AFTER_EVAL" -eq 1 ]]; then
    rm -rf "$adapter_root"
    echo "Deleted adapter artifacts for run ${run_id}: $adapter_root"
  fi

  echo "Completed run ${run_id}"
  echo "Predictions: $pred_csv"
  echo "Metrics:     $metrics_csv"
  echo "Logs:        $train_log | $infer_log | $eval_log"
  echo
}

# ==========================================
# Execute Mistral baseline runs consecutively
# ==========================================

# Zero-shot baseline runs 011-015
run_zero_shot 011 MI   "$VAL_MI"  --val-mi  mi
run_zero_shot 012 ML   "$VAL_ML"  --val-ml  ml
run_zero_shot 013 PG   "$VAL_PG"  --val-pg  pg
run_zero_shot 014 Act  "$VAL_ACT" --val-act act
run_zero_shot 015 MT   "$VAL_MT"  --val-mt  mt

# LoRA baseline runs 016-020
run_lora 016 MI   "$TRAIN_MI"  "$VAL_MI"  --val-mi  mi
run_lora 017 ML   "$TRAIN_ML"  "$VAL_ML"  --val-ml  ml
run_lora 018 PG   "$TRAIN_PG"  "$VAL_PG"  --val-pg  pg
run_lora 019 Act  "$TRAIN_ACT" "$VAL_ACT" --val-act act
run_lora 020 MT   "$TRAIN_MT"  "$VAL_MT"  --val-mt  mt

echo "===== ALL MISTRAL BASELINE RUNS COMPLETED ====="
echo "Logs:         $LOG_DIR"
echo "ZS outputs:   $ZS_OUT_DIR"
echo "LoRA outputs: $LORA_OUT_DIR"
echo "Metrics:      $EVAL_DIR"
echo "Master CSV:   $MASTER_METRICS"
