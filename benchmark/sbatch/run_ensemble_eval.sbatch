#!/bin/bash
#SBATCH --job-name=cvn_ensemble_eval
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Deep Ensemble evaluation: 300 steps, 300 visualizations, + EnKF comparison.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

echo "=== Deep Ensemble Evaluation (300 steps, 300 vis frames) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "${CVN_ROOT}"

${PYTHON} -m benchmark.run_ensemble_eval \
    --device cuda \
    --num-steps 300 \
    --save-dir benchmark/results/ensemble \
    --split test \
    --run-enkf

echo "=== Ensemble eval done ==="
