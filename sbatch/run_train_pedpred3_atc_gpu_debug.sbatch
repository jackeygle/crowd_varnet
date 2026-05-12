#!/bin/bash
# PedPred 老师：crowd_varnet.pedpred_teacher_train（ATC grid_cache + PedPred_Optimized）。
# 续训旧 checkpoint（仅 PA 版 PedPred3 权重）：仍用
#   PYTHONPATH=$PA_ROOT python -m partial_observation_experiments.train
# 并设 PEDPRED_PEDPRED_IMPL=partial_observation（见 PA config）。
# 需在作业里先 cd 到「本次 run 的工作目录」（checkpoints/、tensorboard、*.pth 会写在这里）。
#
#   mkdir -p /scratch/work/zhangx29/crowd_varnet/runs/pedpred3_train_${SLURM_JOB_ID}
#   RUN_DIR=/scratch/work/zhangx29/crowd_varnet/runs/pedpred3_train_${SLURM_JOB_ID} sbatch ...
#
#   或提交前设 RUN_DIR；未设置时默认 crowd_varnet/runs/pedpred3_train_<jobid>。
#
#   MAX_EPOCHS=10（默认）在 gpu-debug 上快速试跑；长训改分区与 MAX_EPOCHS。
#
#SBATCH --job-name=pp3_e10_dbg
# 与 stdout 合并：tqdm 走 stderr，否则 tail 只看 .out 会以为没在训练
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
# 速度项加权：仅当 loss 为 mean total weighted NLLL 时生效；默认 1。试训可 export PEDPRED_LOSS_VEL_SCALE=2
export PEDPRED_LOSS_VEL_SCALE="${PEDPRED_LOSS_VEL_SCALE:-1.0}"
# 默认 PEDPRED_OBJECTIVE=nlll。掩码 MSE：export PEDPRED_OBJECTIVE=physics_mse PEDPRED_LR=1e-4
export PEDPRED_OBJECTIVE="${PEDPRED_OBJECTIVE:-nlll}"
export PEDPRED_LR="${PEDPRED_LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"

RUN_DIR="${RUN_DIR:-${CVN_ROOT}/runs/pedpred3_train_e${MAX_EPOCHS}_${SLURM_JOB_ID:-local}}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

echo "RUN_DIR=${RUN_DIR} MAX_EPOCHS=${MAX_EPOCHS}"
echo "ATC: PEDPRED_ATC_DATA_DIR=${PEDPRED_ATC_DATA_DIR} PEDPRED_DATASET=${PEDPRED_DATASET}"
echo "PEDPRED_BATCH=${PEDPRED_BATCH} NIN=${PEDPRED_NIN} NOUT=${PEDPRED_NOUT} OBJECTIVE=${PEDPRED_OBJECTIVE} LR=${PEDPRED_LR} LOSS_VEL_SCALE=${PEDPRED_LOSS_VEL_SCALE:-1.0}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 新训练：不传 RESUME_GLOB。断点续训：export RESUME_GLOB=模型 pet 名（checkpoints 里文件名前缀）
if [[ -n "${RESUME_GLOB:-}" ]]; then
  exec python -u -m crowd_varnet.pedpred_teacher_train "${RESUME_GLOB}" --max-epochs "${MAX_EPOCHS}"
else
  exec python -u -m crowd_varnet.pedpred_teacher_train --max-epochs "${MAX_EPOCHS}"
fi
