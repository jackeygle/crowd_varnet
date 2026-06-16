#!/bin/bash
#SBATCH --job-name=cvn_student_v13_continuous
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Best teacher (v13) + best loss config (continuous weighting + bg + lookahead + dir)
# Expected to be the strongest combination.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

# v13 teacher (best teacher, NLLL + vel_unc, 30 epoch)
TEACHER_CKPT="${CVN_ROOT}/runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl"
[[ -f "${TEACHER_CKPT}" ]] || { echo "ERROR: teacher missing: ${TEACHER_CKPT}"; exit 1; }
echo "TEACHER_CKPT=${TEACHER_CKPT} (v13, best teacher)"

STU_RUN_DIR="${CVN_ROOT}/runs/cvn_student_v13_continuous_${SLURM_JOB_ID}"
mkdir -p "${STU_RUN_DIR}"

echo "=== student v13 + continuous: 10 ep ==="
echo "STU_RUN_DIR=${STU_RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -u -m crowd_varnet.rollout_train_cli \
  --checkpoint "${TEACHER_CKPT}" \
  --arch pedpred3_gru_mid \
  --epochs 10 \
  --batch 32 \
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

echo "=== student v13 continuous done ==="
ls -la "${STU_RUN_DIR}/" 2>&1
