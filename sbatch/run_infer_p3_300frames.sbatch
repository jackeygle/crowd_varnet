#!/bin/bash
#SBATCH --job-name=infer_p3_300frames
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Step 1: 用 P3 horizon best.pt（当前最佳学生 val 0.04738）做单步推理
# 生成 300 张连续可视化帧
set -euo pipefail
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
PYTHON="${CVN_ROOT}/.venv/bin/python"

CVN_DIR="${CVN_ROOT}/runs/cvn_p3_v13_horizon_17847842"
BEST="${CVN_DIR}/best.pt"
[[ -f "${BEST}" ]] || { echo "ERROR: best.pt missing: ${BEST}"; exit 1; }

OUT_DIR="${CVN_DIR}/infer_300frames"
mkdir -p "${OUT_DIR}"
echo "=== infer_cli with viz-n=300 ==="
echo "BEST=${BEST}"
echo "OUT_DIR=${OUT_DIR}"

PEDPRED_BATCH=4 ${PYTHON} -m crowd_varnet.infer_cli \
    --ckpt "${BEST}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --viz-n 300 \
    --log-interval 100 \
    --val-workers 2

echo "=== infer done ==="
ls -la "${OUT_DIR}" | head -20
echo "(total frames: $(ls ${OUT_DIR}/step_*.png 2>/dev/null | wc -l))"
