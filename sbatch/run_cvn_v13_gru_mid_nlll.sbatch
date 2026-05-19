#!/bin/bash
#SBATCH --job-name=cvn_v13_gru_mid_nlll
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# v13: v11_mid backbone (gru_mid) + 原始 NLLL loss, 30 epoch
# 对照实验：是否 mse_3ch loss 是速度通道精度瓶颈
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
PED_RUN_DIR="${CVN_ROOT}/runs/pedpred_v13_gru_mid_nlll_${SLURM_JOB_ID}"
mkdir -p "${PED_RUN_DIR}"

(
  cd "${PED_RUN_DIR}"

  export PEDPRED_BATCH=48
  export PEDPRED_NIN=5
  export PEDPRED_NOUT=5
  export PEDPRED_OBJECTIVE=nlll
  export PEDPRED_LR=5e-4
  export PEDPRED_WEIGHT_DECAY=1e-4
  export PEDPRED_FLIP_W_PROB=0.5
  export PEDPRED_EARLY_STOP_PATIENCE=20
  export PEDPRED_ARCH=pedpred3_gru_mid

  echo "=== v13: pedpred3_gru_mid + NLLL (30 ep) ==="
  echo "PED_RUN_DIR=${PED_RUN_DIR}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

  ${PYTHON} -u -m crowd_varnet.pedpred_teacher_train --max-epochs 30
)

echo "=== v13 done ==="
ls -la "${PED_RUN_DIR}/checkpoints/" 2>&1
