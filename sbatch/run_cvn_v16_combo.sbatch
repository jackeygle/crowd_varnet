#!/bin/bash
#SBATCH --job-name=cvn_v16_combo
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# v16: combine the 2 best so far — v14 (scheduled sampling) + v13 (unfreeze tail)
# Both attack rollout drift from different angles:
#   - Sched sampling: GT anchors in history during training
#   - Unfreeze tail: teacher can adapt to self-estimated history
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

CVN_DIR="${CVN_ROOT}/runs/cvn_v16_combo_${SLURM_JOB_ID}"
echo "=== v16: scheduled sampling + unfreeze tail (best of v13 + v14), 10ep ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --checkpoint "${PED_CKPT}" --arch pedpred3_gru_mid \
    --epochs 10 --batch 32 --episode-len 300 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-hidden 256 --solver-kernel 3 \
    --solver-rnn-type lstm \
    --solver-attn-heads 4 --solver-momentum 0.0 \
    --solver-lr-grad 0.2 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --prior-unobs-weight 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --weight-mode full \
    --unobs-loss-weight 5.0 \
    --sched-sampling-prob 0.3 \
    --unfreeze-phi-tail 1 --phi-lr 1e-5 \
    --loss-style detached_nll \
    --lambda-bg 1.0 --lambda-dir 0.2 --lookahead-gamma 1.15 \
    --lambda-log-rho 0.0 \
    --lr 2e-4 --lr-eta-min-ratio 0.2 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --predict-uncertainty --seed 0 \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== v16 done ==="
