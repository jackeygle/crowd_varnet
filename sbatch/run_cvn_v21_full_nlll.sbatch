#!/bin/bash
#SBATCH --job-name=cvn_v21_full_nlll
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# v21: pedpred3_gru_mid + FULL NLLL (density + vel_est + vel_unc), 30 epochs.
# Recovers the variance channel that v13/v19 dropped, so downstream aleatoric
# uncertainty can be propagated to the CrowdVarNet student / Deep Ensemble.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
PED_RUN_DIR="${CVN_ROOT}/runs/pedpred_v21_full_nlll_${SLURM_JOB_ID}"
mkdir -p "${PED_RUN_DIR}"

(
  cd "${PED_RUN_DIR}"

  export PEDPRED_BATCH=48
  export PEDPRED_NIN=5
  export PEDPRED_NOUT=5
  export PEDPRED_LR=5e-4
  export PEDPRED_WEIGHT_DECAY=1e-4
  export PEDPRED_FLIP_W_PROB=0.5
  export PEDPRED_EARLY_STOP_PATIENCE=20
  export PEDPRED_ARCH=pedpred3_gru_mid
  export PEDPRED_LAMBDA_VEL_UNC=1.0   # full NLLL; v13 was 0.0

  echo "=== v21: pedpred3_gru_mid + FULL NLLL (30 ep, vel_unc=1.0) ==="
  echo "PED_RUN_DIR=${PED_RUN_DIR}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

  ${PYTHON} -u -m crowd_varnet.pedpred_teacher_train --max-epochs 30
)

echo "=== v21 done ==="
ls -la "${PED_RUN_DIR}/checkpoints/" 2>&1
