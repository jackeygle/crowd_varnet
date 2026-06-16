#!/bin/bash
#SBATCH --job-name=bench_cvn
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_cvn_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_cvn_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Quick test: only run CrowdVarNet (fastest method) to verify pipeline works
set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation:${PYTHONPATH:-}"
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
export PEDPRED_RESOLUTION=1.0
export PEDPRED_PERIOD=1.0
export PEDPRED_KERNEL=tri
export PEDPRED_DATASET=atc:corridor

PYTHON="${CVN_ROOT}/.venv/bin/python"

cd "${CVN_ROOT}"
echo "=== Quick test: CrowdVarNet only (30 steps) ==="
${PYTHON} -m benchmark.run_benchmark \
    --device cuda \
    --num-steps 30 \
    --save-dir "${CVN_ROOT}/benchmark/results/exp_b_quick" \
    --methods cvn naive \
    --split test

echo "=== Done: $(date) ==="
