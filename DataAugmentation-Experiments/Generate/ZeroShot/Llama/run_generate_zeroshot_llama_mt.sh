#!/bin/bash
#SBATCH --job-name=generate_zs_llama_mt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/generate-zs-llama-mt-%j.out
#SBATCH --error=logs/generate-zs-llama-mt-%j.err

set -euo pipefail

source /WAVE/projects/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/activate

REPO_ROOT=/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind
SCRIPT_DIR="$REPO_ROOT/DataAugmentation-Experiments/Generate/ZeroShot/Llama"
ZS_SCRIPT="$SCRIPT_DIR/zero_shot_infer.py"

cd "$REPO_ROOT"
mkdir -p logs "$SCRIPT_DIR/logs"

RUN_STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_STAMP"
mkdir -p "$LOG_DIR"

[[ -f "$ZS_SCRIPT" ]] || { echo "Missing: $ZS_SCRIPT" >&2; exit 1; }

which python && python --version
nvidia-smi || true

cd "$SCRIPT_DIR"

# Run 135 = Generate zero-shot LLaMA + Qwen3-Gen + MT
echo "===== RUN 135 | Generate Zero-shot | MT | Aug: Qwen3-Gen ====="
python "$ZS_SCRIPT" --run-id 135 2>&1 | tee "$LOG_DIR/run_135_generate_zs_mt.log"

# Run 160 = Generate zero-shot LLaMA + Qwen3-Gen+Verify + MT
echo "===== RUN 160 | Generate Zero-shot | MT | Aug: Qwen3-Gen+Verify ====="
python "$ZS_SCRIPT" --run-id 160 2>&1 | tee "$LOG_DIR/run_160_generate_zs_mt.log"

echo "===== DATAAUG GENERATE ZEROSHOT LLAMA MT-ONLY RUNS COMPLETED ====="
