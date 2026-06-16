#!/bin/bash
#SBATCH --job-name=cvn_v10_unobs
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# v10: prior_unobs_weight=0.05 — let solver freely infer unobserved regions
# via attention/RNN, instead of being pulled back to prior.
# + full-image loss to give training signal in unobs regions.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

PED_CKPT="${CVN_ROOT}/runs/pedpred_v21_full_nlll_17906859/checkpoints/one-macaw_best.hkl"
[[ -f "${PED_CKPT}" ]] || { echo "ERROR: teacher ckpt not found"; exit 1; }

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_v10_unobs_${SLURM_JOB_ID}"
echo "=== v10: prior_unobs_weight=0.05, full loss, 10ep ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --checkpoint "${PED_CKPT}" --arch pedpred3_gru_mid \
    --epochs 10 --batch 32 --episode-len 300 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-hidden 256 --solver-kernel 3 \
    --solver-rnn-type lstm \
    --solver-attn-heads 4 --solver-momentum 0.0 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --prior-unobs-weight 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --weight-mode full \
    --loss-style detached_nll \
    --lambda-bg 1.0 --lambda-dir 0.2 --lookahead-gamma 1.15 \
    --lambda-log-rho 0.0 \
    --lr 2e-4 --lr-eta-min-ratio 0.2 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --predict-uncertainty --seed 0 \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== v10 done ==="
