#!/bin/bash
#SBATCH --job-name=cvn_v18_unobs_strong
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# v18: rebalance cost — lower w_obs, raise w_prior, force unobs cells to copy
#       teacher prior strongly (prior_unobs_weight=2.0).
#   --w-obs 0.5            (was 1.0, half the observation pull)
#   --w-prior 1.5          (was 0.5, prior overall 3x stronger)
#   --prior-unobs-weight 2.0   (was 0.05, unobs cells now follow teacher hard)
# Same teacher as our_solution comparison: apt-ibex_train_model_28D.pth
set -euo pipefail

# Force CVN_ROOT to absolute path (do not rely on SLURM_SUBMIT_DIR)
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

CVN_DIR="${CVN_ROOT}/runs/cvn_v18_unobs_strong_${SLURM_JOB_ID}"
echo "=== v18: unobs prior strong (w_obs=0.5, w_prior=1.5, prior_unobs=2.0) ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --use-their-teacher \
    --checkpoint "${PED_CKPT}" \
    --epochs 10 --batch 32 --episode-len 300 --k-prime 8 --warmup 5 \
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
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== v18 done ==="
