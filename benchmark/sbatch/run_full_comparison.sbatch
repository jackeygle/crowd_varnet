#!/bin/bash
#SBATCH --job-name=bench_full_compare
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_full_compare_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_full_compare_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Full pipeline comparison:
# E1: EnKF + their PedPred3 (baseline)
# E2: CrowdVarNet + our v13 teacher (our best)
# E3: CrowdVarNet + their PedPred3 teacher (solver contribution)
# E4: Naive (obs + prior, no solver)
# E5: PedPred-only (no observations)
#
# Metrics: support-pixel RMSE (rho, vx, vy), no variance channel
set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation:${PYTHONPATH:-}"
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
export PEDPRED_RESOLUTION=1.0
export PEDPRED_PERIOD=1.0
export PEDPRED_KERNEL=tri
export PEDPRED_DATASET=atc:corridor

# Use our best student (P3 horizon, v13 teacher)
export CVN_BEST_CKPT="${CVN_ROOT}/runs/cvn_p3_v13_horizon_17847842/best.pt"

PYTHON="${CVN_ROOT}/.venv/bin/python"
cd "${CVN_ROOT}"

echo "=== Full Pipeline Comparison (300 steps, support-pixel RMSE) ==="
echo "E2 ckpt: ${CVN_BEST_CKPT}"
echo "Start: $(date)"

${PYTHON} -m benchmark.run_benchmark \
    --device cuda \
    --num-steps 100 \
    --save-dir "${CVN_ROOT}/benchmark/results/full_comparison" \
    --methods enkf cvn naive pedpred_only \
    --split test

echo "=== Done: $(date) ==="
