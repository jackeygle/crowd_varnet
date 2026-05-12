#!/bin/bash
# CrowdVarNet 推理（infer_cli）— gpu-debug；勿在登录节点用 CUDA。
#
#   sbatch /scratch/work/zhangx29/crowd_varnet/sbatch/run_infer_crowd_varnet_gpu_debug.sbatch
#
# 可选覆盖：
#   CKPT=/path/to/best.pt OUT_DIR=/path/to/out MAX_BATCHES=50 VIZ_N=30 sbatch ...
#
#SBATCH --job-name=cvn_infer
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
export PEDPRED_BATCH="${BATCH:-${PEDPRED_BATCH:-16}}"
export PEDPRED_NIN="${NIN:-${PEDPRED_NIN:-5}}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-1}"

CKPT="${CKPT:-${CVN_ROOT}/runs/cvn50_17729939/best.pt}"
OUT_DIR="${OUT_DIR:-${CVN_ROOT}/runs/cvn50_17729939/infer_vis_gpu}"
VIZ_N="${VIZ_N:-4}"

EXTRA=(--viz-n "${VIZ_N}")
if [[ -n "${MAX_BATCHES:-}" ]]; then
  EXTRA+=(--max-batches "${MAX_BATCHES}")
fi

exec python -u -m crowd_varnet.infer_cli \
  --ckpt "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --val-workers 2 \
  "${EXTRA[@]}"
