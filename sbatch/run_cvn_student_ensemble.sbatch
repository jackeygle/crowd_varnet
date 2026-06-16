#!/bin/bash
#SBATCH --job-name=cvn_student_unc_seed
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Single ensemble member trainer (Stage B + C combined).
# Submit 5 copies with different ENSEMBLE_SEED to get a Deep Ensemble:
#   PED=/scratch/work/zhangx29/crowd_varnet/runs/pedpred_v21_full_nlll_17906859/checkpoints/one-macaw_best.hkl
#   for s in 0 1 2 3 4; do
#       sbatch --export=ALL,ENSEMBLE_SEED=$s,PED_CKPT=$PED \
#           sbatch/run_cvn_student_ensemble.sbatch
#   done
#
# Loss design = teacher-style PedPred3 NLLL (Poisson density + ρ_gt-weighted
# Gaussian velocity NLLL with var_head). Aligns student loss with teacher loss
# philosophically; ch_weights / weight_mode are ignored under loss_style=teacher.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
SEED="${ENSEMBLE_SEED:-0}"
# Default to v21 teacher (full NLLL with vel_unc); override via --export PED_CKPT=...
PED_CKPT="${PED_CKPT:-${CVN_ROOT}/runs/pedpred_v21_full_nlll_17906859/checkpoints/one-macaw_best.hkl}"
[[ -f "${PED_CKPT}" ]] || { echo "ERROR: teacher ckpt not found: ${PED_CKPT}"; exit 1; }

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_ensemble_seed${SEED}_${SLURM_JOB_ID}"
echo "=== Ensemble member: seed=${SEED} ==="
echo "PED_CKPT=${PED_CKPT}"
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Config rationale:
#   - --loss-style teacher: Poisson NLL on density + ρ_gt-weighted Gaussian
#     NLLL on velocity (matches teacher's loss philosophy). Heteroscedastic
#     vel NLLL is enabled because --predict-uncertainty is on.
#   - --predict-uncertainty: enables var_head (3 channels: ρ, vx, vy log_var)
#     for Deep Ensemble aleatoric uncertainty.
#   - --seed ${SEED}: independent init / data order across ensemble members.
#   - Auxiliary losses (bg / dir / log_rho / lookahead) re-enabled to match
#     the best baseline recipe — these prevent background leakage and
#     direction drift that pure teacher-style NLLL doesn't constrain.
${PYTHON} -m crowd_varnet.rollout_train_cli \
    --checkpoint "${PED_CKPT}" --arch pedpred3_gru_mid \
    --epochs 10 --batch 32 --episode-len 300 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-hidden 256 --solver-kernel 3 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --loss-style teacher --teacher-lambda-vel 1.0 \
    --teacher-lambda-rho 3.0 --teacher-lambda-vx 1.5 --teacher-lambda-vy 1.0 \
    --lambda-bg 1.0 --lambda-dir 0.2 --lookahead-gamma 1.15 \
    --lambda-log-rho 0.5 \
    --lr 2e-4 --lr-eta-min-ratio 0.2 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --predict-uncertainty --seed "${SEED}" \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 64 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== seed=${SEED} done ==="
