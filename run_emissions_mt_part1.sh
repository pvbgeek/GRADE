#!/bin/bash
#SBATCH --job-name=emissions_mt_part1
#SBATCH --partition=oignat_lab
#SBATCH --nodelist=oignat01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/emissions-mt-part1-%j.out
#SBATCH --error=logs/emissions-mt-part1-%j.err
#
# run_emissions_mt_part1.sh
#
# PART 1: no-augmentation MT runs, fully inlined (no wrapper scripts).
# Sequential, single GPU. Each run's CodeCarbon emissions land in emissions/.
#
# Runs covered:
#   005  LLaMA-3.1-8B  Zero-shot  MT
#   010  LLaMA-3.1-8B  LoRA       MT
#   015  Mistral-7B    Zero-shot  MT
#   020  Mistral-7B    LoRA       MT
#   025  Qwen3-14B     Zero-shot  MT  (Think: OFF)
#   030  Qwen3-14B     LoRA       MT  (Think: OFF)
#   035  Gemma3-12B    Zero-shot  MT
#   040  Gemma3-12B    LoRA       MT
#   045  Gemma3-27B    Zero-shot  MT
#   050  Gemma3-27B    LoRA       MT
#   055  Qwen3-14B     Zero-shot  MT  (Think: ON)
#   060  Qwen3-14B     LoRA       MT  (Think: ON)
#
# Submit:  sbatch run_emissions_mt_part1.sh

set -uo pipefail
mkdir -p logs

source /WAVE/projects/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/activate

REPO_ROOT=/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind
cd "$REPO_ROOT"

MODE="run"
case "${1:-}" in
  --validate) MODE="validate" ;;
  "")         MODE="run" ;;
  *) echo "Unknown flag: $1" >&2; echo "Usage: $0 [--validate]" >&2; exit 2 ;;
esac

RUN_STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_SCRIPT="$REPO_ROOT/evaluate_run.py"

VAL_MT="$REPO_ROOT/data/val/multitask_val.jsonl"
TRAIN_MT="$REPO_ROOT/data/train/multitask_train.jsonl"

# --- pre-flight ---------------------------------------------------------------
PYTHON_SCRIPTS=(
  "Baseline-Experiments/ZeroShot/Llama-3.1-8B/zero_shot_infer.py"
  "Baseline-Experiments/LORA/Llama-3.1-8B/train_lora.py"
  "Baseline-Experiments/LORA/Llama-3.1-8B/infer_lora.py"
  "Scaling-Experiments/ZeroShot/Mistral-7B/zeroshot_mistral.py"
  "Scaling-Experiments/LORA/Mistral-7B/train_mistral_lora.py"
  "Scaling-Experiments/LORA/Mistral-7B/infer_mistral_lora.py"
  "Scaling-Experiments/ZeroShot/Qwen3-14B/zeroshot_qwen.py"
  "Scaling-Experiments/LORA/Qwen3-14B/train_qwen_lora.py"
  "Scaling-Experiments/LORA/Qwen3-14B/infer_qwen_lora.py"
  "Scaling-Experiments/ZeroShot/Gemma3-12B/zeroshot_gemma12b.py"
  "Scaling-Experiments/LORA/Gemma3-12B/lora_gemma3-12b.py"
  "Scaling-Experiments/ZeroShot/Gemma3-27B/zeroshot_gemma3-27b.py"
  "Scaling-Experiments/LORA/Gemma3-27B/lora_gemma3-27b.py"
  "CoT-Experiments/ZeroShot/Qwen3-14B/zeroshot_qwen_cot.py"
  "CoT-Experiments/LORA/Qwen3-14B/train_qwen_lora_cot.py"
  "CoT-Experiments/LORA/Qwen3-14B/infer_qwen_lora_cot.py"
  "evaluate_run.py"
  "utils/codecarbon_helper.py"
)
echo "==> Pre-flight"
for f in "${PYTHON_SCRIPTS[@]}" "$VAL_MT" "$TRAIN_MT"; do
  case "$f" in
    /*) check_path="$f" ;;     # absolute path — use as-is
    *)  check_path="$REPO_ROOT/$f" ;;
  esac
  if [[ ! -e "$check_path" ]]; then
    echo "  MISSING: $check_path" >&2; exit 1
  fi
done
mkdir -p "$REPO_ROOT/emissions"
echo "  ok: all scripts + datasets + emissions/ present"

if [[ "$MODE" == "validate" ]]; then
  echo "==> Validate-only mode passed."
  exit 0
fi

which python && python --version
nvidia-smi || true

# --- step framework -----------------------------------------------------------
declare -a STEP_RESULTS=()
FAIL_COUNT=0

step_start() {
  STEP_LABEL="$1"
  STEP_T0=$(date +%s)
  echo
  echo "================================================================"
  echo "STEP: $STEP_LABEL"
  echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================"
}
step_end() {
  local rc=$1
  local t1=$(date +%s); local elapsed=$((t1 - STEP_T0))
  if [[ $rc -eq 0 ]]; then
    echo "  ok in ${elapsed}s ($((elapsed/60))m)"
    STEP_RESULTS+=("OK   ${elapsed}s  $STEP_LABEL")
  else
    echo "  FAILED rc=$rc in ${elapsed}s -- continuing pipeline" >&2
    STEP_RESULTS+=("FAIL rc=$rc ${elapsed}s  $STEP_LABEL")
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
}

PIPELINE_T0=$(date +%s)

# ============================================================================
# RUN 005 — LLaMA-3.1-8B Zero-shot MT
# ============================================================================
step_start "005 LLaMA-3.1-8B Zero-shot MT"
LLAMA_ZS_DIR="$REPO_ROOT/Baseline-Experiments/ZeroShot/Llama-3.1-8B"
LLAMA_LOG_DIR="$REPO_ROOT/Baseline-Experiments/logs/Llama-3.1-8B/$RUN_STAMP"
LLAMA_LORA_DIR="$REPO_ROOT/Baseline-Experiments/LORA/Llama-3.1-8B"
mkdir -p "$LLAMA_LOG_DIR" "$LLAMA_ZS_DIR/outputs/$RUN_STAMP" \
         "$LLAMA_LORA_DIR/outputs/$RUN_STAMP" "$LLAMA_LORA_DIR/adapters/$RUN_STAMP"
python "$LLAMA_ZS_DIR/zero_shot_infer.py" --run-id 005 \
  2>&1 | tee "$LLAMA_LOG_DIR/run_005_zeroshot_mt_infer.log"
step_end ${PIPESTATUS[0]}

# ============================================================================
# RUN 010 — LLaMA-3.1-8B LoRA MT  (train + infer)
# ============================================================================
step_start "010 LLaMA-3.1-8B LoRA MT"
LLAMA_ADAPTER_DIR="$LLAMA_LORA_DIR/adapters/$RUN_STAMP/run_010/MT"
LLAMA_LORA_PRED_CSV="$LLAMA_LORA_DIR/outputs/$RUN_STAMP/MT_lora_preds.csv"
python "$LLAMA_LORA_DIR/train_lora.py" \
  --task MT \
  --train-jsonl "$TRAIN_MT" \
  --adapter-out "$LLAMA_ADAPTER_DIR" \
  2>&1 | tee "$LLAMA_LOG_DIR/run_010_lora_mt_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [[ $TRAIN_RC -eq 0 && -d "$LLAMA_ADAPTER_DIR" ]]; then
  python "$LLAMA_LORA_DIR/infer_lora.py" \
    --task MT \
    --input-jsonl "$VAL_MT" \
    --adapter-path "$LLAMA_ADAPTER_DIR" \
    --predictions-out "$LLAMA_LORA_PRED_CSV" \
    2>&1 | tee "$LLAMA_LOG_DIR/run_010_lora_mt_infer.log"
  step_end ${PIPESTATUS[0]}
else
  step_end $TRAIN_RC
fi

# ============================================================================
# RUN 015 — Mistral-7B Zero-shot MT
# ============================================================================
step_start "015 Mistral-7B Zero-shot MT"
MISTRAL_ZS_DIR="$REPO_ROOT/Scaling-Experiments/ZeroShot/Mistral-7B"
MISTRAL_LORA_DIR="$REPO_ROOT/Scaling-Experiments/LORA/Mistral-7B"
MISTRAL_EVAL_DIR="$REPO_ROOT/Scaling-Experiments/EVAL/Mistral-7B"
MISTRAL_LOG_DIR="$REPO_ROOT/Scaling-Experiments/logs/Mistral-7B/$RUN_STAMP"
mkdir -p "$MISTRAL_LOG_DIR" "$MISTRAL_ZS_DIR/outputs/$RUN_STAMP" \
         "$MISTRAL_LORA_DIR/outputs/$RUN_STAMP" "$MISTRAL_LORA_DIR/adapters/$RUN_STAMP" \
         "$MISTRAL_EVAL_DIR"
MISTRAL_ZS_PRED="$MISTRAL_ZS_DIR/outputs/$RUN_STAMP/run_015_mistral7b_zeroshot_mt_predictions.csv"
MISTRAL_ZS_METRICS="$MISTRAL_EVAL_DIR/run_015_mistral7b_zeroshot_mt_metrics.csv"
python "$MISTRAL_ZS_DIR/zeroshot_mistral.py" \
  --task MT --input-jsonl "$VAL_MT" --predictions-out "$MISTRAL_ZS_PRED" \
  2>&1 | tee "$MISTRAL_LOG_DIR/run_015_zeroshot_mt.log"
ZS_RC=${PIPESTATUS[0]}
if [[ $ZS_RC -eq 0 ]]; then
  python "$EVAL_SCRIPT" \
    --predictions "$MISTRAL_ZS_PRED" --task MT --run-id 015 \
    --model Mistral-7B --method Zero-shot --aug None --think N/A \
    --val-mt "$VAL_MT" --out "$MISTRAL_ZS_METRICS" \
    2>&1 | tee "$MISTRAL_LOG_DIR/run_015_zeroshot_mt_eval.log"
  step_end ${PIPESTATUS[0]}
else
  step_end $ZS_RC
fi

# ============================================================================
# RUN 020 — Mistral-7B LoRA MT  (train + infer + eval)
# ============================================================================
step_start "020 Mistral-7B LoRA MT"
MISTRAL_ADAPTER_ROOT="$MISTRAL_LORA_DIR/adapters/$RUN_STAMP/run_020"
MISTRAL_ADAPTER_PATH="$MISTRAL_ADAPTER_ROOT/MT"
MISTRAL_LORA_PRED="$MISTRAL_LORA_DIR/outputs/$RUN_STAMP/run_020_mistral7b_lora_mt_predictions.csv"
MISTRAL_LORA_METRICS="$MISTRAL_EVAL_DIR/run_020_mistral7b_lora_mt_metrics.csv"
python "$MISTRAL_LORA_DIR/train_mistral_lora.py" \
  --task MT --train-jsonl "$TRAIN_MT" --adapter-out "$MISTRAL_ADAPTER_ROOT" \
  --epochs 3 --learning-rate 2e-4 --batch-size 4 --grad-accum 4 \
  --max-seq-length 2048 --seed 42 \
  2>&1 | tee "$MISTRAL_LOG_DIR/run_020_lora_mt_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [[ $TRAIN_RC -eq 0 && -d "$MISTRAL_ADAPTER_PATH" ]]; then
  python "$MISTRAL_LORA_DIR/infer_mistral_lora.py" \
    --task MT --adapter-path "$MISTRAL_ADAPTER_PATH" \
    --input-jsonl "$VAL_MT" --predictions-out "$MISTRAL_LORA_PRED" \
    2>&1 | tee "$MISTRAL_LOG_DIR/run_020_lora_mt_infer.log"
  INFER_RC=${PIPESTATUS[0]}
  if [[ $INFER_RC -eq 0 ]]; then
    python "$EVAL_SCRIPT" \
      --predictions "$MISTRAL_LORA_PRED" --task MT --run-id 020 \
      --model Mistral-7B --method LoRA --aug None --think N/A \
      --val-mt "$VAL_MT" --out "$MISTRAL_LORA_METRICS" \
      2>&1 | tee "$MISTRAL_LOG_DIR/run_020_lora_mt_eval.log"
    step_end ${PIPESTATUS[0]}
  else
    step_end $INFER_RC
  fi
else
  step_end $TRAIN_RC
fi

# ============================================================================
# RUN 025 — Qwen3-14B Zero-shot MT (Think: OFF)
# ============================================================================
step_start "025 Qwen3-14B Zero-shot MT (Think:OFF)"
QWEN_ZS_DIR="$REPO_ROOT/Scaling-Experiments/ZeroShot/Qwen3-14B"
QWEN_LORA_DIR="$REPO_ROOT/Scaling-Experiments/LORA/Qwen3-14B"
QWEN_EVAL_DIR="$REPO_ROOT/Scaling-Experiments/EVAL/Qwen3-14B"
QWEN_LOG_DIR="$REPO_ROOT/Scaling-Experiments/logs/Qwen3-14B/$RUN_STAMP"
mkdir -p "$QWEN_LOG_DIR" "$QWEN_ZS_DIR/outputs/$RUN_STAMP" \
         "$QWEN_LORA_DIR/outputs/$RUN_STAMP" "$QWEN_LORA_DIR/adapters/$RUN_STAMP" \
         "$QWEN_EVAL_DIR"
QWEN_ZS_PRED="$QWEN_ZS_DIR/outputs/$RUN_STAMP/run_025_qwen3_14b_zeroshot_mt_predictions.csv"
QWEN_ZS_METRICS="$QWEN_EVAL_DIR/run_025_qwen3_14b_zeroshot_mt_metrics.csv"
python "$QWEN_ZS_DIR/zeroshot_qwen.py" \
  --task MT --input-jsonl "$VAL_MT" --predictions-out "$QWEN_ZS_PRED" \
  2>&1 | tee "$QWEN_LOG_DIR/run_025_zeroshot_mt.log"
ZS_RC=${PIPESTATUS[0]}
if [[ $ZS_RC -eq 0 ]]; then
  python "$EVAL_SCRIPT" \
    --predictions "$QWEN_ZS_PRED" --task MT --run-id 025 \
    --model Qwen3-14B --method Zero-shot --aug None --think OFF \
    --val-mt "$VAL_MT" --out "$QWEN_ZS_METRICS" \
    2>&1 | tee "$QWEN_LOG_DIR/run_025_zeroshot_mt_eval.log"
  step_end ${PIPESTATUS[0]}
else
  step_end $ZS_RC
fi

# ============================================================================
# RUN 030 — Qwen3-14B LoRA MT (Think: OFF)
# ============================================================================
step_start "030 Qwen3-14B LoRA MT (Think:OFF)"
QWEN_ADAPTER_ROOT="$QWEN_LORA_DIR/adapters/$RUN_STAMP/run_030"
QWEN_ADAPTER_PATH="$QWEN_ADAPTER_ROOT/MT"
QWEN_LORA_PRED="$QWEN_LORA_DIR/outputs/$RUN_STAMP/run_030_qwen3_14b_lora_mt_predictions.csv"
QWEN_LORA_METRICS="$QWEN_EVAL_DIR/run_030_qwen3_14b_lora_mt_metrics.csv"
python "$QWEN_LORA_DIR/train_qwen_lora.py" \
  --task MT --train-jsonl "$TRAIN_MT" --adapter-out "$QWEN_ADAPTER_ROOT" \
  --epochs 3 --learning-rate 2e-4 --batch-size 2 --grad-accum 8 \
  --max-seq-length 2048 --seed 42 \
  2>&1 | tee "$QWEN_LOG_DIR/run_030_lora_mt_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [[ $TRAIN_RC -eq 0 && -d "$QWEN_ADAPTER_PATH" ]]; then
  python "$QWEN_LORA_DIR/infer_qwen_lora.py" \
    --task MT --adapter-path "$QWEN_ADAPTER_PATH" \
    --input-jsonl "$VAL_MT" --predictions-out "$QWEN_LORA_PRED" \
    2>&1 | tee "$QWEN_LOG_DIR/run_030_lora_mt_infer.log"
  INFER_RC=${PIPESTATUS[0]}
  if [[ $INFER_RC -eq 0 ]]; then
    python "$EVAL_SCRIPT" \
      --predictions "$QWEN_LORA_PRED" --task MT --run-id 030 \
      --model Qwen3-14B --method LoRA --aug None --think OFF \
      --val-mt "$VAL_MT" --out "$QWEN_LORA_METRICS" \
      2>&1 | tee "$QWEN_LOG_DIR/run_030_lora_mt_eval.log"
    step_end ${PIPESTATUS[0]}
  else
    step_end $INFER_RC
  fi
else
  step_end $TRAIN_RC
fi

# ============================================================================
# RUN 035 — Gemma3-12B Zero-shot MT (driven by --run-id; script handles paths)
# ============================================================================
step_start "035 Gemma3-12B Zero-shot MT"
G12_ZS_DIR="$REPO_ROOT/Scaling-Experiments/ZeroShot/Gemma3-12B"
G12_ZS_LOG_DIR="$G12_ZS_DIR/logs/$RUN_STAMP"
mkdir -p "$G12_ZS_LOG_DIR"
( cd "$G12_ZS_DIR" && \
  python "$G12_ZS_DIR/zeroshot_gemma12b.py" --run-id 035 \
    2>&1 | tee "$G12_ZS_LOG_DIR/run_035_zeroshot_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 040 — Gemma3-12B LoRA MT
# ============================================================================
step_start "040 Gemma3-12B LoRA MT"
G12_LORA_DIR="$REPO_ROOT/Scaling-Experiments/LORA/Gemma3-12B"
G12_LORA_LOG_DIR="$G12_LORA_DIR/logs/$RUN_STAMP"
mkdir -p "$G12_LORA_LOG_DIR"
( cd "$G12_LORA_DIR" && \
  python "$G12_LORA_DIR/lora_gemma3-12b.py" --run-id 040 \
    2>&1 | tee "$G12_LORA_LOG_DIR/run_040_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 045 — Gemma3-27B Zero-shot MT
# ============================================================================
step_start "045 Gemma3-27B Zero-shot MT"
G27_ZS_DIR="$REPO_ROOT/Scaling-Experiments/ZeroShot/Gemma3-27B"
G27_ZS_LOG_DIR="$G27_ZS_DIR/logs/$RUN_STAMP"
mkdir -p "$G27_ZS_LOG_DIR"
( cd "$G27_ZS_DIR" && \
  python "$G27_ZS_DIR/zeroshot_gemma3-27b.py" --run-id 045 \
    2>&1 | tee "$G27_ZS_LOG_DIR/run_045_zeroshot_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 050 — Gemma3-27B LoRA MT (script does train + validation)
# ============================================================================
step_start "050 Gemma3-27B LoRA MT"
G27_LORA_DIR="$REPO_ROOT/Scaling-Experiments/LORA/Gemma3-27B"
G27_LORA_LOG_DIR="$G27_LORA_DIR/logs/$RUN_STAMP"
mkdir -p "$G27_LORA_LOG_DIR"
( cd "$G27_LORA_DIR" && \
  python "$G27_LORA_DIR/lora_gemma3-27b.py" --run-id 050 \
    2>&1 | tee "$G27_LORA_LOG_DIR/run_050_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 055 — Qwen3-14B Zero-shot MT (Think: ON)
# ============================================================================
step_start "055 Qwen3-14B Zero-shot MT (Think:ON)"
COT_ZS_DIR="$REPO_ROOT/CoT-Experiments/ZeroShot/Qwen3-14B"
COT_LORA_DIR="$REPO_ROOT/CoT-Experiments/LORA/Qwen3-14B"
COT_EVAL_DIR="$REPO_ROOT/CoT-Experiments/EVAL/Qwen3-14B"
COT_LOG_DIR="$REPO_ROOT/CoT-Experiments/logs/Qwen3-14B/$RUN_STAMP"
mkdir -p "$COT_LOG_DIR" "$COT_ZS_DIR/outputs/$RUN_STAMP" \
         "$COT_LORA_DIR/outputs/$RUN_STAMP" "$COT_LORA_DIR/adapters/$RUN_STAMP" \
         "$COT_EVAL_DIR"
COT_ZS_PRED="$COT_ZS_DIR/outputs/$RUN_STAMP/run_055_qwen3_14b_zeroshot_think_mt_predictions.csv"
COT_ZS_METRICS="$COT_EVAL_DIR/run_055_qwen3_14b_zeroshot_think_mt_metrics.csv"
python "$COT_ZS_DIR/zeroshot_qwen_cot.py" \
  --task MT --input-jsonl "$VAL_MT" --predictions-out "$COT_ZS_PRED" \
  2>&1 | tee "$COT_LOG_DIR/run_055_zeroshot_think_mt.log"
ZS_RC=${PIPESTATUS[0]}
if [[ $ZS_RC -eq 0 ]]; then
  python "$EVAL_SCRIPT" \
    --predictions "$COT_ZS_PRED" --task MT --run-id 055 \
    --model Qwen3-14B --method Zero-shot --aug None --think ON \
    --val-mt "$VAL_MT" --out "$COT_ZS_METRICS" \
    2>&1 | tee "$COT_LOG_DIR/run_055_zeroshot_think_mt_eval.log"
  step_end ${PIPESTATUS[0]}
else
  step_end $ZS_RC
fi

# ============================================================================
# RUN 060 — Qwen3-14B LoRA MT (Think: ON)
# ============================================================================
step_start "060 Qwen3-14B LoRA MT (Think:ON)"
COT_ADAPTER_ROOT="$COT_LORA_DIR/adapters/$RUN_STAMP/run_060"
COT_ADAPTER_PATH="$COT_ADAPTER_ROOT/MT"
COT_LORA_PRED="$COT_LORA_DIR/outputs/$RUN_STAMP/run_060_qwen3_14b_lora_think_mt_predictions.csv"
COT_LORA_METRICS="$COT_EVAL_DIR/run_060_qwen3_14b_lora_think_mt_metrics.csv"
python "$COT_LORA_DIR/train_qwen_lora_cot.py" \
  --task MT --train-jsonl "$TRAIN_MT" --adapter-out "$COT_ADAPTER_ROOT" \
  --epochs 3 --learning-rate 2e-4 --batch-size 2 --grad-accum 8 \
  --max-seq-length 2048 --seed 42 \
  2>&1 | tee "$COT_LOG_DIR/run_060_lora_think_mt_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [[ $TRAIN_RC -eq 0 && -d "$COT_ADAPTER_PATH" ]]; then
  python "$COT_LORA_DIR/infer_qwen_lora_cot.py" \
    --task MT --adapter-path "$COT_ADAPTER_PATH" \
    --input-jsonl "$VAL_MT" --predictions-out "$COT_LORA_PRED" \
    2>&1 | tee "$COT_LOG_DIR/run_060_lora_think_mt_infer.log"
  INFER_RC=${PIPESTATUS[0]}
  if [[ $INFER_RC -eq 0 ]]; then
    python "$EVAL_SCRIPT" \
      --predictions "$COT_LORA_PRED" --task MT --run-id 060 \
      --model Qwen3-14B --method LoRA --aug None --think ON \
      --val-mt "$VAL_MT" --out "$COT_LORA_METRICS" \
      2>&1 | tee "$COT_LOG_DIR/run_060_lora_think_mt_eval.log"
    step_end ${PIPESTATUS[0]}
  else
    step_end $INFER_RC
  fi
else
  step_end $TRAIN_RC
fi

# --- summary ------------------------------------------------------------------
PIPELINE_T1=$(date +%s)
TOTAL=$((PIPELINE_T1 - PIPELINE_T0))
echo
echo "================================================================"
echo "PART 1 SUMMARY  (total $((TOTAL/3600))h $(((TOTAL%3600)/60))m)"
echo "================================================================"
for r in "${STEP_RESULTS[@]}"; do echo "  $r"; done
echo
echo "Failures: $FAIL_COUNT / ${#STEP_RESULTS[@]}"
echo "Emissions CSVs: $REPO_ROOT/emissions/"
[[ $FAIL_COUNT -eq 0 ]] || exit 1
