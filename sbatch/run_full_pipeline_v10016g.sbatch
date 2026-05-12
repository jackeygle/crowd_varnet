#!/bin/bash
#SBATCH --job-name=pp_cvn_pipe
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00

set -euo pipefail

module load scicomp-pytorch-env/2026.1

CVN_ROOT="${CVN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "${CVN_ROOT}/logs"

# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_venv.sh"

export PYTHONPATH="${CVN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

unset PEDPRED_DATASET PEDPRED_ATC_SUBSET PEDPRED_ATC_DATA_DIR PEDPRED_RESOLUTION PEDPRED_PERIOD PEDPRED_KERNEL PEDPRED_BATCH PEDPRED_NIN PEDPRED_NOUT 2>/dev/null || true
# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_atc_data_env.sh"

echo "=== 1. Training PedPred ==="
export PEDPRED_BATCH="16"
export PEDPRED_NIN="5"
export PEDPRED_NOUT="5"
export PEDPRED_OBJECTIVE="physics_mse"
export PEDPRED_LR="1e-3"
MAX_EPOCHS="15"

export RUN_DIR="${CVN_ROOT}/runs/pedpred_v4_final_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

python -u -m crowd_varnet.pedpred_teacher_train --max-epochs "${MAX_EPOCHS}"

echo "=== 2. Finding PedPred Checkpoint ==="
export CKPT=$(ls -1 ${RUN_DIR}/checkpoints/*_best.hkl | head -n 1)
echo "Found checkpoint: ${CKPT}"

echo "=== 3. Training CrowdVarNet ==="
export EPOCHS="15"
export SAVE_DIR="${CVN_ROOT}/runs/cvn15_final_${SLURM_JOB_ID}"
mkdir -p "${SAVE_DIR}"

# 与教师共用 nin / ATC；VarNet 重建单步目标须 nout=1（教师训练仍为 5）
export PEDPRED_NOUT=1

cd "${CVN_ROOT}"
python -u -m crowd_varnet.train_cli \
  --checkpoint "${CKPT}" \
  --arch pedpred3 \
  --epochs "${EPOCHS}" \
  --nin "${PEDPRED_NIN}" \
  --nout 1 \
  --device cuda \
  --train-workers 2 \
  --val-workers 0 \
  --n-iter 8 \
  --obs-mode sensor \
  --sensing-range 5 \
  --num-agents 3 \
  --save-dir "${SAVE_DIR}"

echo "Pipeline finished successfully!"
