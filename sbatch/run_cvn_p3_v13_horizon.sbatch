#!/bin/bash
#SBATCH --job-name=cvn_p3_v13_horizon
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Phase 3: 用 P2-a (ep10) 的 best.pt 接续，episode 拉长到 600，挖出多步 rollout 鲁棒性
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
PED_CKPT="${CVN_ROOT}/runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl"
PREV_BEST="${CVN_ROOT}/runs/cvn_p2a_v13_17845947/best.pt"
[[ -f "${PED_CKPT}" ]] || { echo "ERROR: teacher v13 ckpt missing"; exit 1; }
[[ -f "${PREV_BEST}" ]] || { echo "ERROR: P2-a best.pt missing"; exit 1; }

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_p3_v13_horizon_${SLURM_JOB_ID}"
echo "=== Phase 3: long-horizon rollout finetune from ${PREV_BEST} ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --checkpoint "${PED_CKPT}" --arch pedpred3_gru_mid \
    --warmstart "${PREV_BEST}" \
    --epochs 8 --batch 32 --episode-len 600 --k-prime 12 --warmup 5 \
    --n-iter 8 --solver-type convgru --solver-hidden 256 --solver-kernel 3 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --lr 5e-5 --lr-eta-min-ratio 0.25 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== Phase 3 done ==="
