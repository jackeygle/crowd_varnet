#!/bin/bash
#SBATCH --job-name=bench_continuous
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_continuous_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_continuous_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation:${PYTHONPATH:-}"
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
export PEDPRED_RESOLUTION=1.0
export PEDPRED_PERIOD=1.0
export PEDPRED_KERNEL=tri
export PEDPRED_DATASET=atc:corridor

# Override CVN checkpoint to use the continuous weighting student
export CVN_BEST_CKPT="${CVN_ROOT}/runs/cvn_student_v18_continuous_17886476/best.pt"

PYTHON="${CVN_ROOT}/.venv/bin/python"
cd "${CVN_ROOT}"

echo "=== Benchmark: continuous student (17886476) vs EnKF ==="
${PYTHON} -m benchmark.run_benchmark \
    --device cuda \
    --num-steps 300 \
    --save-dir "${CVN_ROOT}/benchmark/results/exp_b_continuous" \
    --methods enkf cvn naive \
    --split test

echo "=== Done: $(date) ==="
