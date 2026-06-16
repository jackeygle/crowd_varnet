#!/bin/bash
#SBATCH --job-name=bench_final
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_final_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_final_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Final comprehensive evaluation with plots
set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation:${PYTHONPATH:-}"
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
export PEDPRED_RESOLUTION=1.0
export PEDPRED_PERIOD=1.0
export PEDPRED_KERNEL=tri
export PEDPRED_DATASET=atc:corridor

# Use our best student (P3 horizon)
export CVN_BEST_CKPT="${CVN_ROOT}/runs/cvn_p3_v13_horizon_17847842/best.pt"

PYTHON="${CVN_ROOT}/.venv/bin/python"
cd "${CVN_ROOT}"

echo "=== Final Comprehensive Evaluation ==="
echo "CVN ckpt: ${CVN_BEST_CKPT}"
echo "Start: $(date)"

${PYTHON} -m benchmark.run_final_eval \
    --device cuda \
    --num-steps 100 \
    --save-dir "${CVN_ROOT}/benchmark/results/final" \
    --split test \
    --viz-frames 8 \
    --methods enkf cvn naive pedpred_only

echo "=== Done: $(date) ==="
echo "Results:"
ls -la "${CVN_ROOT}/benchmark/results/final/"
ls -la "${CVN_ROOT}/benchmark/results/final/visualizations/" 2>/dev/null | head -10
