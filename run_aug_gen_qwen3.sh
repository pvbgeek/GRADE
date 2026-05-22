#!/bin/bash
#SBATCH --job-name=aug_gen_qwen3
#SBATCH --partition=oignat_lab
#SBATCH --nodelist=oignat01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/slurm_logs/aug_genverify_MT_qwen3_%j.out
#SBATCH --error=/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/slurm_logs/aug_genverify_MT_qwen3_%j.err

set -euo pipefail

cd /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind

PY=/WAVE/projects2/oignat_lab/ParthBhalerao/ENVS/VIDQA/bin/python
export PYTHONUNBUFFERED=1

echo "Job started at: $(date)"
echo "Host: $(hostname)"
nvidia-smi

$PY - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

$PY -u generate_augmented_data.py \
  --mode genverify \
  --tasks MT \
  --labels "No" "To some extent" \
  --target-per-label 500 \
  --model-path /WAVE/datasets/oignat_lab/QWEN3 \
  --out-gen-dir /WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen \
  --out-genverify-dir '/WAVE/projects/CSEN-346-Sp26/Group3/TutorMind/data/train/Gen+Verify' \
  --max-new-tokens 1024 \
  --verification-max-new-tokens 1024 \
  --max-attempts-multiplier 40 \
  --resume

echo "Job finished at: $(date)"

