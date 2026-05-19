#!/bin/bash
#SBATCH --job-name=cvn_p2a_eval_full
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# 在 P2-a (job 17845947) ep10 best.pt 上运行 diag + infer + rollout_infer（完整版）
set -euo pipefail
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
PYTHON="${CVN_ROOT}/.venv/bin/python"

CVN_DIR="${CVN_ROOT}/runs/cvn_p2a_v13_17845947"
BEST="${CVN_DIR}/best.pt"
[[ -f "${BEST}" ]] || { echo "ERROR: best.pt missing"; exit 1; }

DIAG_DIR="${CVN_DIR}/diag_leak"; mkdir -p "${DIAG_DIR}"
echo "=== diag_density_leak ==="
${PYTHON} "${CVN_ROOT}/scripts/diag_density_leak.py" \
    --ckpt "${BEST}" --device cuda --max-batches 30 --val-workers 2 \
    --out "${DIAG_DIR}/leak.json"

INFER_DIR="${CVN_DIR}/infer_after"; mkdir -p "${INFER_DIR}"
echo "=== infer_cli (full 1124 batches) ==="
PEDPRED_BATCH=16 ${PYTHON} -m crowd_varnet.infer_cli \
    --ckpt "${BEST}" --out-dir "${INFER_DIR}" --device cuda \
    --viz-n 8 --log-interval 50 --val-workers 2

ROLL_DIR="${CVN_DIR}/rollout_infer"; mkdir -p "${ROLL_DIR}"
echo "=== rollout_infer ==="
${PYTHON} -m crowd_varnet.rollout_infer_cli \
    --ckpt "${BEST}" --save-dir "${ROLL_DIR}" --device cuda \
    --n-episodes 5 --frames-per-plot 10 --episode-len 150 --val-workers 2

echo "=== eval done ==="
