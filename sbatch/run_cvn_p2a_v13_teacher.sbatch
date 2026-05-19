#!/bin/bash
#SBATCH --job-name=cvn_p2a_v13_teacher
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Phase 2-a: 用 v13 (gru_mid + NLLL) 作为冻结教师，训 CrowdVarNet 学生
# 模板抄自 run_cvn_full_v5b_mse.sbatch 的 Stage 2，跳过 Stage 1
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"
PED_CKPT="${CVN_ROOT}/runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl"
[[ -f "${PED_CKPT}" ]] || { echo "ERROR: teacher v13 ckpt missing"; exit 1; }
echo "PED_CKPT=${PED_CKPT}"

export PEDPRED_NIN=5
export PEDPRED_NOUT=1

CVN_DIR="${CVN_ROOT}/runs/cvn_p2a_v13_${SLURM_JOB_ID}"
echo "=== Phase 2-a: CrowdVarNet (rollout TBPTT) + v13 teacher (gru_mid) ==="
echo "CVN_DIR=${CVN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

${PYTHON} -m crowd_varnet.rollout_train_cli \
    --checkpoint "${PED_CKPT}" --arch pedpred3_gru_mid \
    --epochs 25 --batch 48 --episode-len 300 --k-prime 8 --warmup 5 \
    --n-iter 8 --solver-type convgru --solver-hidden 256 --solver-kernel 3 \
    --w-prior 0.5 --rho-mask-thr 0.05 \
    --ch-weights 3.0 1.5 1.0 0.0 \
    --lr 2e-4 --lr-eta-min-ratio 0.25 \
    --weight-decay 1e-4 --solver-dropout 0.1 --ema-decay 0.999 \
    --grad-clip 0.5 \
    --save-dir "${CVN_DIR}" --log-interval 5 --val-max-episodes 96 \
    --train-workers 4 --val-workers 4 --device cuda

echo "=== Phase 2-a training done ==="

DIAG_DIR="${CVN_DIR}/diag_leak"; mkdir -p "${DIAG_DIR}"
${PYTHON} "${CVN_ROOT}/scripts/diag_density_leak.py" \
    --ckpt "${CVN_DIR}/best.pt" --device cuda --max-batches 30 --val-workers 2 \
    --out "${DIAG_DIR}/leak.json"

INFER_DIR="${CVN_DIR}/infer_after"; mkdir -p "${INFER_DIR}"
PEDPRED_BATCH=16 ${PYTHON} -m crowd_varnet.infer_cli \
    --ckpt "${CVN_DIR}/best.pt" --out-dir "${INFER_DIR}" --device cuda \
    --viz-n 8 --log-interval 50 --val-workers 2

ROLL_DIR="${CVN_DIR}/rollout_infer"; mkdir -p "${ROLL_DIR}"
${PYTHON} -m crowd_varnet.rollout_infer_cli \
    --ckpt "${CVN_DIR}/best.pt" --save-dir "${ROLL_DIR}" --device cuda \
    --n-episodes 5 --frames-per-plot 10 --episode-len 150 --val-workers 2

echo "=== Phase 2-a pipeline done ==="
