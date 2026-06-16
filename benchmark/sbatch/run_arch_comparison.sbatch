#!/bin/bash
#SBATCH --job-name=cvn_arch_eval
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Architecture comparison: v7 (GRU) vs v8 (LSTM) vs their-teacher vs EnKF
# 300 steps, 300 visualizations
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

echo "=== Architecture Comparison Evaluation ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "${CVN_ROOT}"

${PYTHON} -m benchmark.run_arch_comparison \
    --device cuda \
    --num-steps 300 \
    --save-dir benchmark/results/arch_comparison \
    --split test

echo "=== Done ==="
