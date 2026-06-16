#!/bin/bash
#SBATCH --job-name=cvn_v18_debug
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# v18 smoke test: 1 epoch, small batch, val-max-episodes=8 to confirm
# the new --w-obs flag and the v18 weights work end-to-end.
set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

PED_CKPT="/scratch/work/zhangx29/Partial_observation/apt-ibex_train_model_28D.pth"
[[ -f "${PED_CKPT}" ]] || { echo "ERROR: teacher ckpt not found"; exit 1; }

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_v18_debug_${SLURM_JOB_ID}"
echo "=== v18 DEBUG smoke test: 1 epoch, batch=8 ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --use-their-teacher \
    --checkpoint "${PED_CKPT}" \
    --epochs 1 --batch 8 --episode-len 100 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-hidden 256 --solver-kernel 3 \
    --solver-rnn-type lstm \
    --solver-attn-heads 4 --solver-momentum 0.0 \
    --solver-lr-grad 0.2 \
    --w-obs 0.5 \
    --w-prior 1.5 \
    --prior-unobs-weight 2.0 \
    --rho-mask-thr 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --weight-mode full \
    --unobs-loss-weight 5.0 \
    --sched-sampling-prob 0.3 \
    --loss-style detached_nll \
    --lambda-bg 1.0 --lambda-dir 0.2 --lookahead-gamma 1.15 \
    --lambda-log-rho 0.0 \
    --lr 2e-4 --lr-eta-min-ratio 0.2 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --predict-uncertainty --seed 0 \
    --save-dir "${CVN_DIR}" --log-interval 2 --val-max-episodes 8 \
    --train-workers 2 --val-workers 2 --device cuda

echo "=== v18 debug done ==="
