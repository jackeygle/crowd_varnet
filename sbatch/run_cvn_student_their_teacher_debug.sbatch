#!/bin/bash
#SBATCH --job-name=cvn_their_debug
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# DEBUG: test their-teacher adapter works (1 epoch only)
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
TEACHER_CKPT="/scratch/work/zhangx29/Partial_observation/apt-ibex_train_model_28D.pth"

STU_RUN_DIR="${CVN_ROOT}/runs/cvn_their_teacher_debug_${SLURM_JOB_ID}"
mkdir -p "${STU_RUN_DIR}"

echo "=== DEBUG: their teacher + continuous, 1 epoch ==="
${PYTHON} -u -m crowd_varnet.rollout_train_cli \
  --checkpoint "${TEACHER_CKPT}" \
  --use-their-teacher \
  --epochs 1 \
  --batch 4 \
  --episode-len 40 \
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
  --grad-clip 0.5 \
  --save-dir "${STU_RUN_DIR}"

echo "=== DEBUG done ==="
