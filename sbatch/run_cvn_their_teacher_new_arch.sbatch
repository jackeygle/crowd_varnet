#!/bin/bash
#SBATCH --job-name=cvn_their_teacher_v7
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# CrowdVarNet + THEIR teacher (apt-ibex PedPred3).
# New architecture (ConvLSTM + Attention), detached_nll, 25 epoch.
# For fair comparison: same teacher as EnKF baseline.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

TEACHER_CKPT="/scratch/work/zhangx29/Partial_observation/apt-ibex_train_model_28D.pth"
[[ -f "${TEACHER_CKPT}" ]] || { echo "ERROR: their teacher missing: ${TEACHER_CKPT}"; exit 1; }

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_their_teacher_v7_${SLURM_JOB_ID}"
echo "=== CrowdVarNet + THEIR teacher (fair comparison with EnKF) ==="
echo "TEACHER_CKPT=${TEACHER_CKPT}"
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Same config as v7 ensemble but:
#   --use-their-teacher: loads PedPred3 via TheirPedPredAdapter
#   --solver-rnn-type lstm: ConvLSTM (4DVarNet style)
#   batch reduced to 16 (their teacher runs on CPU, slower)
${PYTHON} -m crowd_varnet.rollout_train_cli \
    --use-their-teacher \
    --checkpoint "${TEACHER_CKPT}" \
    --epochs 25 --batch 16 --episode-len 300 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-hidden 256 --solver-kernel 3 \
    --solver-rnn-type lstm \
    --solver-attn-heads 4 --solver-momentum 0.0 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --weight-mode continuous \
    --loss-style detached_nll \
    --lambda-bg 1.0 --lambda-dir 0.2 --lookahead-gamma 1.15 \
    --lambda-log-rho 0.0 \
    --lr 2e-4 --lr-eta-min-ratio 0.2 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --predict-uncertainty --seed 0 \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== done ==="
