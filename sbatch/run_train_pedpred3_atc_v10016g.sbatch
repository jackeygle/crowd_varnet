#!/bin/bash
# PedPred3 — ATC grid_cache，V100 16GB，默认 70 epoch；nin=nout=5（与 config.py 默认一致，可用环境变量覆盖）。
#
#   sbatch /scratch/work/zhangx29/crowd_varnet/sbatch/run_train_pedpred3_atc_v10016g.sbatch
#
#   RUN_DIR=... MAX_EPOCHS=70 PEDPRED_BATCH=12 sbatch ...
#
#SBATCH --job-name=pp3_v16_e70
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --partition=gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

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
export PEDPRED_BATCH="${PEDPRED_BATCH:-16}"
export PEDPRED_NIN="${PEDPRED_NIN:-5}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-5}"
export PEDPRED_OBJECTIVE="${PEDPRED_OBJECTIVE:-nlll}"
export PEDPRED_LR="${PEDPRED_LR:-1e-3}"

MAX_EPOCHS="${MAX_EPOCHS:-70}"

RUN_DIR="${RUN_DIR:-${CVN_ROOT}/runs/pedpred3_n5_v10016_e${MAX_EPOCHS}_${SLURM_JOB_ID:-local}}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

echo "RUN_DIR=${RUN_DIR}"
echo "partition=v100-16g MAX_EPOCHS=${MAX_EPOCHS}"
echo "ATC: PEDPRED_ATC_DATA_DIR=${PEDPRED_ATC_DATA_DIR} PEDPRED_DATASET=${PEDPRED_DATASET}"
echo "PEDPRED_BATCH=${PEDPRED_BATCH} NIN=${PEDPRED_NIN} NOUT=${PEDPRED_NOUT} OBJECTIVE=${PEDPRED_OBJECTIVE} LR=${PEDPRED_LR}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ -n "${RESUME_GLOB:-}" ]]; then
  exec python -u -m crowd_varnet.pedpred_teacher_train "${RESUME_GLOB}" --max-epochs "${MAX_EPOCHS}"
else
  exec python -u -m crowd_varnet.pedpred_teacher_train --max-epochs "${MAX_EPOCHS}"
fi
