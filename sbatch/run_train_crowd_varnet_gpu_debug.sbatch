#!/bin/bash
# CrowdVarNet 训练 — gpu-debug（勿在登录节点用 CUDA）。
#
#   sbatch /scratch/work/zhangx29/crowd_varnet/sbatch/run_train_crowd_varnet_gpu_debug.sbatch
#
# 可选：SENSING_RANGE=5 NUM_AGENTS=3 EPOCHS=2 sbatch ...
#
#SBATCH --job-name=cvn_dbg
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

module purge 2>/dev/null || true
module load scicomp-pytorch-env/2026.1

CVN_ROOT="${CVN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "${CVN_ROOT}/logs"
cd "${CVN_ROOT}"

# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_venv.sh"

export PYTHONPATH="${CVN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, torch.__file__, 'cuda', torch.cuda.is_available())"

unset PEDPRED_DATASET PEDPRED_ATC_SUBSET PEDPRED_ATC_DATA_DIR PEDPRED_RESOLUTION PEDPRED_PERIOD PEDPRED_KERNEL PEDPRED_BATCH PEDPRED_NIN PEDPRED_NOUT 2>/dev/null || true
# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_atc_data_env.sh"
export PEDPRED_BATCH="${BATCH:-${PEDPRED_BATCH:-16}}"
export PEDPRED_NIN="${NIN:-${PEDPRED_NIN:-5}}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-1}"

echo "ATC: PEDPRED_ATC_DATA_DIR=${PEDPRED_ATC_DATA_DIR} subset=${PEDPRED_ATC_SUBSET} r=${PEDPRED_RESOLUTION} p=${PEDPRED_PERIOD} k=${PEDPRED_KERNEL}"

# 本仓库 --arch pedpred3 即 vendored PedPred_v4；须与自训教师配套（勿用 apt-ibex 28D）。
CKPT="${CKPT:-${CVN_ROOT}/runs/pedpred_v4_final_17743578/checkpoints/amazed-finch_best.hkl}"
EPOCHS="${EPOCHS:-1}"
SAVE_DIR="${SAVE_DIR:-${CVN_ROOT}/runs/cvn_dbg_${SLURM_JOB_ID:-local}}"
mkdir -p "${SAVE_DIR}"

# 与正式训练对齐：GRU solver、有密度格 mask loss；仅 epoch 默认 1 便于快速冒烟
exec python -u -m crowd_varnet.train_cli \
  --checkpoint "${CKPT}" \
  --arch pedpred3 \
  --epochs "${EPOCHS}" \
  --nin "${PEDPRED_NIN}" \
  --device cuda \
  --train-workers 2 \
  --val-workers 0 \
  --n-iter 8 \
  --obs-mode sensor \
  --sensing-range "${SENSING_RANGE:-5}" \
  --num-agents "${NUM_AGENTS:-3}" \
  --rho-mask-thr "${RHO_MASK_THR:-0.05}" \
  --use-gru \
  --save-dir "${SAVE_DIR}"
