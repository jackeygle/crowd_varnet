#!/bin/bash
# CrowdVarNet 正式训练 7 epoch（默认 gpu-v100-16g，8h）。短时调试可：
#   sbatch --partition=gpu-debug --time=00:30:00 run_train_crowd_varnet_50ep_gpu_debug.sbatch
# 数据：默认 ``<repo>/data/ATC/grid_cache/...``（见 data/README.txt；可用 PEDPRED_ATC_DATA_DIR 覆盖）。
#
#   EPOCHS=7  sbatch  /scratch/work/zhangx29/crowd_varnet/sbatch/run_train_crowd_varnet_50ep_gpu_debug.sbatch
#   或覆盖：BATCH=16 NIN=5 SENSING_RANGE=5 NUM_AGENTS=3 RHO_MASK_THR=0.05 sbatch ...
#   loss 仅在真值 ρ>RHO_MASK_THR 的格点平均（见 train_cli --rho-mask-thr）。
#
#SBATCH --job-name=cvn50
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
# 长训练请用 v100 分区；gpu-debug 往往墙时很短。可环境变量覆盖：SBATCH_PARTITION SBATCH_TIME
#SBATCH --partition=gpu-v100-16g
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# 集群 CUDA/PyTorch：必须用 scicomp-pytorch-env（先 purge 再 load，减少模块冲突）
module purge 2>/dev/null || true
module load scicomp-pytorch-env/2026.1

CVN_ROOT="${CVN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "${CVN_ROOT}/logs"
cd "${CVN_ROOT}"

# 在 module 环境之上叠加本仓库 .venv（hickle 等）；torch 一般以当前 python 为准
# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_venv.sh"

export PYTHONPATH="${CVN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, torch.__file__, 'cuda', torch.cuda.is_available())"

# 与教师 PedPred 共用 ATC 路径与离散化参数（见 sbatch/inc_atc_data_env.sh）
unset PEDPRED_DATASET PEDPRED_ATC_SUBSET PEDPRED_ATC_DATA_DIR PEDPRED_RESOLUTION PEDPRED_PERIOD PEDPRED_KERNEL PEDPRED_BATCH PEDPRED_NIN PEDPRED_NOUT 2>/dev/null || true
# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_atc_data_env.sh"
export PEDPRED_BATCH="${BATCH:-${PEDPRED_BATCH:-16}}"
export PEDPRED_NIN="${NIN:-${PEDPRED_NIN:-5}}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-1}"

echo "ATC: PEDPRED_ATC_DATA_DIR=${PEDPRED_ATC_DATA_DIR} subset=${PEDPRED_ATC_SUBSET} r=${PEDPRED_RESOLUTION} p=${PEDPRED_PERIOD} k=${PEDPRED_KERNEL}"

# 本仓库 --arch pedpred3 即 vendored PedPred_v4；须与自训教师配套（勿用 apt-ibex 28D）。
CKPT="${CKPT:-${CVN_ROOT}/runs/pedpred_v4_final_17743578/checkpoints/amazed-finch_best.hkl}"
EPOCHS="${EPOCHS:-7}"
SAVE_DIR="${SAVE_DIR:-${CVN_ROOT}/runs/cvn50_masked_${SLURM_JOB_ID:-local}}"

mkdir -p "${SAVE_DIR}"

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
