#!/bin/bash
#SBATCH --job-name=emissions_mt_part2
#SBATCH --partition=oignat_lab
#SBATCH --nodelist=oignat01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/emissions-mt-part2-%j.out
#SBATCH --error=logs/emissions-mt-part2-%j.err
#
# run_emissions_mt_part2.sh
#
# PART 2: data-augmentation MT runs + run 115, fully inlined (no wrapper scripts).
# Sequential, single GPU. Each run's CodeCarbon emissions land in emissions/.
#
# Runs covered:
#   065  LLaMA-3.1-8B  LoRA  MT  Aug: Qwen3-Gen
#   090  LLaMA-3.1-8B  LoRA  MT  Aug: Qwen3-Gen+Verify
#   070  Mistral-7B    LoRA  MT  Aug: Qwen3-Gen
#   095  Mistral-7B    LoRA  MT  Aug: Qwen3-Gen+Verify
#   075  Qwen3-14B     LoRA  MT  Aug: Qwen3-Gen          (Think: OFF)
#   100  Qwen3-14B     LoRA  MT  Aug: Qwen3-Gen+Verify   (Think: OFF)
#   080  Gemma3-12B    LoRA  MT  Aug: Qwen3-Gen
#   105  Gemma3-12B    LoRA  MT  Aug: Qwen3-Gen+Verify
#   085  Gemma3-27B    LoRA  MT  Aug: Qwen3-Gen
#   110  Gemma3-27B    LoRA  MT  Aug: Qwen3-Gen+Verify
#   115  Qwen3-14B     LoRA  MT  Aug: Qwen3-Gen          (Think: ON)
#
# Submit:  sbatch run_emissions_mt_part2.sh

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
TRAIN_GEN_MT="$REPO_ROOT/data/train/Gen/multitask_train_aug_qwen3_gen500.jsonl"
TRAIN_GV_MT="$REPO_ROOT/data/train/Gen+Verify/multitask_train_aug_qwen3_genverify500.jsonl"

# --- pre-flight ---------------------------------------------------------------
PYTHON_SCRIPTS=(
  "DataAugmentation-Experiments/LORA/Llama/train_lora.py"
  "DataAugmentation-Experiments/LORA/Llama/infer_lora.py"
  "DataAugmentation-Experiments/LORA/Mistral/train_mistral_lora.py"
  "DataAugmentation-Experiments/LORA/Mistral/infer_mistral_lora.py"
  "DataAugmentation-Experiments/LORA/Qwen3-14B/train_qwen_lora.py"
  "DataAugmentation-Experiments/LORA/Qwen3-14B/infer_qwen_lora.py"
  "DataAugmentation-Experiments/LORA/Gemma3-12B/lora_gemma3-12b.py"
  "DataAugmentation-Experiments/LORA/Gemma3-27B/lora_gemma3-27b.py"
  "CoT-Experiments/LORA/Qwen3-14B/train_qwen_lora_cot.py"
  "CoT-Experiments/LORA/Qwen3-14B/infer_qwen_lora_cot.py"
  "evaluate_run.py"
  "utils/codecarbon_helper.py"
)
echo "==> Pre-flight"
for f in "${PYTHON_SCRIPTS[@]}" "$VAL_MT" "$TRAIN_GEN_MT" "$TRAIN_GV_MT"; do
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
# RUN 065 — LLaMA DataAug LoRA MT  Aug: Qwen3-Gen
# ============================================================================
step_start "065 LLaMA-3.1-8B LoRA MT Aug:Gen"
DA_LLAMA_DIR="$REPO_ROOT/DataAugmentation-Experiments/LORA/Llama"
DA_LLAMA_LOG="$DA_LLAMA_DIR/logs/$RUN_STAMP"
mkdir -p "$DA_LLAMA_LOG"
( cd "$DA_LLAMA_DIR" && \
  python "$DA_LLAMA_DIR/train_lora.py" --run-id 065 \
    2>&1 | tee "$DA_LLAMA_LOG/run_065_train.log" && \
  python "$DA_LLAMA_DIR/infer_lora.py" --run-id 065 \
    2>&1 | tee "$DA_LLAMA_LOG/run_065_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 090 — LLaMA DataAug LoRA MT  Aug: Qwen3-Gen+Verify
# ============================================================================
step_start "090 LLaMA-3.1-8B LoRA MT Aug:Gen+Verify"
( cd "$DA_LLAMA_DIR" && \
  python "$DA_LLAMA_DIR/train_lora.py" --run-id 090 \
    2>&1 | tee "$DA_LLAMA_LOG/run_090_train.log" && \
  python "$DA_LLAMA_DIR/infer_lora.py" --run-id 090 \
    2>&1 | tee "$DA_LLAMA_LOG/run_090_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 070 — Mistral DataAug LoRA MT  Aug: Qwen3-Gen
# ============================================================================
step_start "070 Mistral-7B LoRA MT Aug:Gen"
DA_MISTRAL_DIR="$REPO_ROOT/DataAugmentation-Experiments/LORA/Mistral"
DA_MISTRAL_LOG="$DA_MISTRAL_DIR/logs/$RUN_STAMP"
mkdir -p "$DA_MISTRAL_LOG"
( cd "$DA_MISTRAL_DIR" && \
  python "$DA_MISTRAL_DIR/train_mistral_lora.py" --run-id 070 \
    2>&1 | tee "$DA_MISTRAL_LOG/run_070_train.log" && \
  python "$DA_MISTRAL_DIR/infer_mistral_lora.py" --run-id 070 \
    2>&1 | tee "$DA_MISTRAL_LOG/run_070_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 095 — Mistral DataAug LoRA MT  Aug: Qwen3-Gen+Verify
# ============================================================================
step_start "095 Mistral-7B LoRA MT Aug:Gen+Verify"
( cd "$DA_MISTRAL_DIR" && \
  python "$DA_MISTRAL_DIR/train_mistral_lora.py" --run-id 095 \
    2>&1 | tee "$DA_MISTRAL_LOG/run_095_train.log" && \
  python "$DA_MISTRAL_DIR/infer_mistral_lora.py" --run-id 095 \
    2>&1 | tee "$DA_MISTRAL_LOG/run_095_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 075 — Qwen3-14B DataAug LoRA MT  Aug: Qwen3-Gen   (Think: OFF)
# ============================================================================
step_start "075 Qwen3-14B LoRA MT Aug:Gen (Think:OFF)"
DA_QWEN_DIR="$REPO_ROOT/DataAugmentation-Experiments/LORA/Qwen3-14B"
DA_QWEN_LOG="$DA_QWEN_DIR/logs/$RUN_STAMP"
mkdir -p "$DA_QWEN_LOG"
( cd "$DA_QWEN_DIR" && \
  python "$DA_QWEN_DIR/train_qwen_lora.py" --run-id 075 \
    2>&1 | tee "$DA_QWEN_LOG/run_075_train.log" && \
  python "$DA_QWEN_DIR/infer_qwen_lora.py" --run-id 075 \
    2>&1 | tee "$DA_QWEN_LOG/run_075_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 100 — Qwen3-14B DataAug LoRA MT  Aug: Qwen3-Gen+Verify (Think: OFF)
# ============================================================================
step_start "100 Qwen3-14B LoRA MT Aug:Gen+Verify (Think:OFF)"
( cd "$DA_QWEN_DIR" && \
  python "$DA_QWEN_DIR/train_qwen_lora.py" --run-id 100 \
    2>&1 | tee "$DA_QWEN_LOG/run_100_train.log" && \
  python "$DA_QWEN_DIR/infer_qwen_lora.py" --run-id 100 \
    2>&1 | tee "$DA_QWEN_LOG/run_100_infer.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 080 — Gemma3-12B DataAug LoRA MT  Aug: Qwen3-Gen
# ============================================================================
step_start "080 Gemma3-12B LoRA MT Aug:Gen"
DA_G12_DIR="$REPO_ROOT/DataAugmentation-Experiments/LORA/Gemma3-12B"
DA_G12_LOG="$DA_G12_DIR/logs/$RUN_STAMP"
mkdir -p "$DA_G12_LOG"
( cd "$DA_G12_DIR" && \
  python "$DA_G12_DIR/lora_gemma3-12b.py" --run-id 080 \
    2>&1 | tee "$DA_G12_LOG/run_080_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 105 — Gemma3-12B DataAug LoRA MT  Aug: Qwen3-Gen+Verify
# ============================================================================
step_start "105 Gemma3-12B LoRA MT Aug:Gen+Verify"
( cd "$DA_G12_DIR" && \
  python "$DA_G12_DIR/lora_gemma3-12b.py" --run-id 105 \
    2>&1 | tee "$DA_G12_LOG/run_105_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 085 — Gemma3-27B DataAug LoRA MT  Aug: Qwen3-Gen
# ============================================================================
step_start "085 Gemma3-27B LoRA MT Aug:Gen"
DA_G27_DIR="$REPO_ROOT/DataAugmentation-Experiments/LORA/Gemma3-27B"
DA_G27_LOG="$DA_G27_DIR/logs/$RUN_STAMP"
mkdir -p "$DA_G27_LOG"
( cd "$DA_G27_DIR" && \
  python "$DA_G27_DIR/lora_gemma3-27b.py" --run-id 085 \
    2>&1 | tee "$DA_G27_LOG/run_085_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 110 — Gemma3-27B DataAug LoRA MT  Aug: Qwen3-Gen+Verify
# ============================================================================
step_start "110 Gemma3-27B LoRA MT Aug:Gen+Verify"
( cd "$DA_G27_DIR" && \
  python "$DA_G27_DIR/lora_gemma3-27b.py" --run-id 110 \
    2>&1 | tee "$DA_G27_LOG/run_110_lora_mt.log"; \
  exit ${PIPESTATUS[0]} )
step_end $?

# ============================================================================
# RUN 115 — Qwen3-14B CoT LoRA MT  Aug: Qwen3-Gen  (Think: ON)
# ============================================================================
step_start "115 Qwen3-14B LoRA MT Aug:Gen (Think:ON)"
COT_LORA_DIR="$REPO_ROOT/CoT-Experiments/LORA/Qwen3-14B"
COT_GEN_ROOT="$REPO_ROOT/CoT-Experiments/LORA/Qwen3_14B_Gen"
COT_GEN_OUT="$COT_GEN_ROOT/outputs/$RUN_STAMP"
COT_GEN_ADAPTER="$COT_GEN_ROOT/adapters/$RUN_STAMP/run_115_gen_mt"
COT_GEN_METRICS="$COT_GEN_ROOT/metrics/$RUN_STAMP"
COT_GEN_LOG="$COT_GEN_ROOT/logs/$RUN_STAMP"
mkdir -p "$COT_GEN_OUT" "$COT_GEN_ADAPTER" "$COT_GEN_METRICS" "$COT_GEN_LOG"
COT_115_PRED="$COT_GEN_OUT/run_115_qwen3_14b_think_lora_gen_mt_predictions.csv"
COT_115_METRICS="$COT_GEN_METRICS/run_115_qwen3_14b_think_lora_gen_mt_metrics.csv"
COT_115_ADAPTER_INNER="$COT_GEN_ADAPTER/MT"
python "$COT_LORA_DIR/train_qwen_lora_cot.py" \
  --task MT --train-jsonl "$TRAIN_GEN_MT" \
  --adapter-out "$COT_GEN_ADAPTER" \
  --epochs 3 --learning-rate 2e-4 --batch-size 2 --grad-accum 8 \
  --max-seq-length 2048 --seed 42 \
  2>&1 | tee "$COT_GEN_LOG/run_115_lora_gen_mt_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [[ $TRAIN_RC -eq 0 && -d "$COT_115_ADAPTER_INNER" ]]; then
  python "$COT_LORA_DIR/infer_qwen_lora_cot.py" \
    --task MT --adapter-path "$COT_115_ADAPTER_INNER" \
    --input-jsonl "$VAL_MT" --predictions-out "$COT_115_PRED" \
    2>&1 | tee "$COT_GEN_LOG/run_115_lora_gen_mt_infer.log"
  INFER_RC=${PIPESTATUS[0]}
  if [[ $INFER_RC -eq 0 ]]; then
    python "$EVAL_SCRIPT" \
      --predictions "$COT_115_PRED" --task MT --run-id 115 \
      --model Qwen3-14B --method LoRA --aug Qwen3-Gen --think ON \
      --val-mt "$VAL_MT" --out "$COT_115_METRICS" \
      2>&1 | tee "$COT_GEN_LOG/run_115_lora_gen_mt_eval.log"
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
echo "PART 2 SUMMARY  (total $((TOTAL/3600))h $(((TOTAL%3600)/60))m)"
echo "================================================================"
for r in "${STEP_RESULTS[@]}"; do echo "  $r"; done
echo
echo "Failures: $FAIL_COUNT / ${#STEP_RESULTS[@]}"
echo "Emissions CSVs: $REPO_ROOT/emissions/"
[[ $FAIL_COUNT -eq 0 ]] || exit 1
