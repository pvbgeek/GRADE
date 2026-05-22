#!/bin/bash
#SBATCH --job-name=llama3p1-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llama-baseline-%A_%a.out
#SBATCH --error=logs/llama-baseline-%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

source /WAVE/projects/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/activate

REPO_ROOT=/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind
cd "$REPO_ROOT"
mkdir -p logs

ZS_DIR="$REPO_ROOT/Baseline-Experiments/ZeroShot/Llama-3.1-8B"
LORA_DIR="$REPO_ROOT/Baseline-Experiments/LORA/Llama-3.1-8B"

ZS_SCRIPT="$ZS_DIR/zero_shot_infer.py"
LORA_TRAIN_SCRIPT="$LORA_DIR/train_lora.py"
LORA_INFER_SCRIPT="$LORA_DIR/infer_lora.py"

MODEL_PATH=/WAVE/datasets/oignat_lab/Meta-Llama-3.1-8B-Instruct
PROMPTS_JSON="$REPO_ROOT/prompts.json"
DATA_VAL_DIR="$REPO_ROOT/data/val"
DATA_TRAIN_DIR="$REPO_ROOT/data/train"

RUN_STAMP="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$REPO_ROOT/Baseline-Experiments/logs/Llama-3.1-8B/$RUN_STAMP"
ZS_OUT_DIR="$ZS_DIR/outputs/$RUN_STAMP"
LORA_OUT_DIR="$LORA_DIR/outputs/$RUN_STAMP"
LORA_ADAPTER_BASE="$LORA_DIR/adapters/$RUN_STAMP"

mkdir -p "$LOG_DIR" "$ZS_OUT_DIR" "$LORA_OUT_DIR" "$LORA_ADAPTER_BASE"

# ==========================================
# EXACT DATASET PATHS ONLY
# DO NOT CHANGE / DO NOT DISCOVER ALTERNATIVES
# ==========================================
TRAIN_MI="$DATA_TRAIN_DIR/mistake_identification_train.jsonl"
TRAIN_ML="$DATA_TRAIN_DIR/mistake_location_train.jsonl"
TRAIN_PG="$DATA_TRAIN_DIR/providing_guidance_train.jsonl"
TRAIN_ACT="$DATA_TRAIN_DIR/actionability_train.jsonl"
TRAIN_MT="$DATA_TRAIN_DIR/multitask_train.jsonl"

VAL_MI="$DATA_VAL_DIR/mistake_identification_val.jsonl"
VAL_ML="$DATA_VAL_DIR/mistake_location_val.jsonl"
VAL_PG="$DATA_VAL_DIR/providing_guidance_val.jsonl"
VAL_ACT="$DATA_VAL_DIR/actionability_val.jsonl"
VAL_MT="$DATA_VAL_DIR/multitask_val.jsonl"

REQUIRED_FILES=(
    "$ZS_SCRIPT"
    "$LORA_TRAIN_SCRIPT"
    "$LORA_INFER_SCRIPT"
    "$PROMPTS_JSON"
    "$TRAIN_MI" "$TRAIN_ML" "$TRAIN_PG" "$TRAIN_ACT" "$TRAIN_MT"
    "$VAL_MI" "$VAL_ML" "$VAL_PG" "$VAL_ACT" "$VAL_MT"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -e "$f" ]]; then
        echo "Missing required file: $f" >&2
        exit 1
    fi
done

echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "ROOT: $REPO_ROOT"
echo "RUN_STAMP: $RUN_STAMP"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "LOG_DIR: $LOG_DIR"
echo "ZS_OUT_DIR: $ZS_OUT_DIR"
echo "LORA_OUT_DIR: $LORA_OUT_DIR"
echo "LORA_ADAPTER_BASE: $LORA_ADAPTER_BASE"
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

run_zero_shot() {
    local run_id="$1"
    local task="$2"
    local tag_lc="$3"

    local preds_csv="$ZS_OUT_DIR/${task}_zeroshot_preds.csv"
    local infer_log="$LOG_DIR/run_${run_id}_zeroshot_${tag_lc}_infer.log"

    echo "===== RUN ${run_id} | Zero-shot | ${task} ====="

    python "$ZS_SCRIPT" \
        --model_path   "$MODEL_PATH" \
        --data_dir     "$DATA_VAL_DIR" \
        --prompts_json "$PROMPTS_JSON" \
        --out_dir      "$ZS_OUT_DIR" \
        --tasks        "$task" \
        2>&1 | tee "$infer_log"

    echo "Completed run ${run_id}"
    echo "Predictions: $preds_csv"
    echo "Logs:        $infer_log"
    echo
}

run_lora() {
    local run_id="$1"
    local task="$2"
    local tag_lc="$3"
    local val_file="$4"
    local train_file="$5"

    local adapter_root="$LORA_ADAPTER_BASE/run_${run_id}"
    local adapter_path="$adapter_root/$task"

    local preds_csv="$LORA_OUT_DIR/${task}_lora_preds.csv"
    local train_log="$LOG_DIR/run_${run_id}_lora_${tag_lc}_train.log"
    local infer_log="$LOG_DIR/run_${run_id}_lora_${tag_lc}_infer.log"

    echo "===== RUN ${run_id} | LoRA | ${task} ====="

    python "$LORA_TRAIN_SCRIPT" \
        --task          "$task" \
        --train-jsonl   "$train_file" \
        --adapter-out   "$adapter_path" \
        2>&1 | tee "$train_log"

    if [[ ! -d "$adapter_path" ]]; then
        echo "Expected adapter path not found: $adapter_path" >&2
        exit 1
    fi

    python "$LORA_INFER_SCRIPT" \
        --task            "$task" \
        --input-jsonl     "$val_file" \
        --adapter-path    "$adapter_path" \
        --predictions-out "$preds_csv" \
        2>&1 | tee "$infer_log"

    echo "Completed run ${run_id}"
    echo "Adapter:     $adapter_path"
    echo "Predictions: $preds_csv"
    echo "Logs:        $train_log | $infer_log"
    echo
}

# ==========================================
# Array dispatch: 0-4 = zero-shot (runs 001-005), 5-9 = LoRA (runs 006-010)
# ==========================================
i="${SLURM_ARRAY_TASK_ID:-0}"

case "$i" in
    0) run_zero_shot 001 Mistake_Identification mi  ;;
    1) run_zero_shot 002 Mistake_Location       ml  ;;
    2) run_zero_shot 003 Providing_Guidance     pg  ;;
    3) run_zero_shot 004 Actionability          act ;;
    4) run_zero_shot 005 Multitask              mt  ;;
    5) run_lora      006 Mistake_Identification mi  "$VAL_MI"  "$TRAIN_MI"  ;;
    6) run_lora      007 Mistake_Location       ml  "$VAL_ML"  "$TRAIN_ML"  ;;
    7) run_lora      008 Providing_Guidance     pg  "$VAL_PG"  "$TRAIN_PG"  ;;
    8) run_lora      009 Actionability          act "$VAL_ACT" "$TRAIN_ACT" ;;
    9) run_lora      010 Multitask              mt  "$VAL_MT"  "$TRAIN_MT"  ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID: $i" >&2; exit 1 ;;
esac

echo "Array task ${i} finished."