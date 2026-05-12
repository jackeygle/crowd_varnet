#!/bin/bash
# PedPred 老师开环一步推理 + PNG（pedpred_infer_cli）— gpu-debug。
#
#   sbatch /scratch/work/zhangx29/crowd_varnet/sbatch/run_pedpred_infer_gpu_debug.sbatch
#
# 默认：pet-jaguar 重训 best 权重 + valid 全量 + 32 张 PNG。可覆盖：
#   TEACHER_CKPT=... OUT_DIR=... SPLIT=test VIZ_N=64 MAX_BATCHES=100 sbatch ...
#
#SBATCH --job-name=pp_infer
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

module load scicomp-pytorch-env/2026.1

CVN_ROOT="${CVN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "${CVN_ROOT}/logs"
cd "${CVN_ROOT}"

# shellcheck source=/dev/null
source "${CVN_ROOT}/sbatch/inc_venv.sh"

export PYTHONPATH="${CVN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

unset PEDPRED_RESOLUTION PEDPRED_PERIOD PEDPRED_KERNEL PEDPRED_BATCH PEDPRED_NIN PEDPRED_NOUT 2>/dev/null || true
export PEDPRED_RESOLUTION="${PEDPRED_RESOLUTION:-1.0}"
export PEDPRED_PERIOD="${PEDPRED_PERIOD:-1.0}"
export PEDPRED_KERNEL="${PEDPRED_KERNEL:-tri}"
export PEDPRED_BATCH="${BATCH:-${PEDPRED_BATCH:-8}}"
export PEDPRED_NIN="${NIN:-${PEDPRED_NIN:-5}}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-1}"

TEACHER_CKPT="${TEACHER_CKPT:-/scratch/work/zhangx29/crowd_varnet/runs/pedpred3_n5_v10016_e70_17730282/pet-jaguar_best_for_infer.pth}"
ARCH="${ARCH:-pedpred3}"
SPLIT="${SPLIT:-valid}"
OUT_DIR="${OUT_DIR:-${CVN_ROOT}/runs/pedpred_infer_gpu_debug_${SLURM_JOB_ID:-local}}"
VIZ_N="${VIZ_N:-32}"
NIN="${NIN:-5}"
NOUT="${NOUT:-1}"

EXTRA=(--viz-n "${VIZ_N}" --split "${SPLIT}" --arch "${ARCH}" --nin "${NIN}" --nout "${NOUT}" --batch "${PEDPRED_BATCH}")
if [[ -n "${MAX_BATCHES:-}" ]]; then
  EXTRA+=(--max-batches "${MAX_BATCHES}")
fi

exec python -u -m crowd_varnet.pedpred_infer_cli \
  --checkpoint "${TEACHER_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --num-workers 2 \
  "${EXTRA[@]}"
