#!/bin/bash
#SBATCH --job-name=cvn_student_their_teacher
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Student trained with THEIR teacher (PedPred3 from Partial_observation).
# This isolates the contribution of CrowdVarNet's solver:
#   - Same observation geometry, same data, same training pipeline as 17884867
#   - Only difference: teacher = their PedPred3 (apt-ibex_train_model_28D.pth)
#   - Result vs EnKF + their teacher = pure solver contribution
#   - Result vs CrowdVarNet + our teacher (17884867) = teacher contribution
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh

# PYTHONPATH includes parent of Partial_observation so `from Partial_observation.models import ...` works
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

# Their teacher checkpoint (PedPred3 trained on 28d ATC, paper baseline)
TEACHER_CKPT="/scratch/work/zhangx29/Partial_observation/apt-ibex_train_model_28D.pth"
[[ -f "${TEACHER_CKPT}" ]] || { echo "ERROR: teacher missing: ${TEACHER_CKPT}"; exit 1; }
echo "TEACHER_CKPT=${TEACHER_CKPT} (Partial_observation PedPred3, nin=1)"

STU_RUN_DIR="${CVN_ROOT}/runs/cvn_student_their_teacher_${SLURM_JOB_ID}"
mkdir -p "${STU_RUN_DIR}"

echo "=== student with THEIR teacher: 10 ep ==="
echo "STU_RUN_DIR=${STU_RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Same config as run_cvn_student_v18.sbatch (job 17884867), but:
#   --use-their-teacher : load PedPred3 via TheirPedPredAdapter (CPU forward, nin=1)
#   batch reduced from 32 -> 16: their teacher runs on CPU, smaller batch = less CPU bottleneck
${PYTHON} -u -m crowd_varnet.rollout_train_cli \
  --checkpoint "${TEACHER_CKPT}" \
  --use-their-teacher \
  --epochs 10 \
  --batch 16 \
  --episode-len 300 \
  --k-prime 8 \
  --warmup 5 \
  --n-iter 8 \
  --solver-hidden 256 \
  --solver-kernel 3 \
  --solver-dropout 0.1 \
  --w-prior 0.5 \
  --rho-mask-thr 0.05 \
  --ch-weights 1.0 1.0 1.0 0.0 \
  --weight-mode continuous \
  --lambda-bg 1.0 \
  --lookahead-gamma 1.15 \
  --lambda-dir 0.2 \
  --lr 2e-4 \
  --lr-eta-min-ratio 0.25 \
  --weight-decay 1e-4 \
  --ema-decay 0.999 \
  --grad-clip 0.5 \
  --train-workers 4 \
  --val-workers 2 \
  --val-max-episodes 32 \
  --save-dir "${STU_RUN_DIR}"

echo "=== student with their teacher done ==="
ls -la "${STU_RUN_DIR}/" 2>&1
