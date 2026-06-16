#!/bin/bash
#SBATCH --job-name=bench_teacher_v18
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_teacher_v18_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_teacher_v18_%j.err
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

PYTHON="${CVN_ROOT}/.venv/bin/python"
cd "${CVN_ROOT}"

echo "=== Experiment A: Teacher v18 vs Their Teacher ==="

# Override to use v18 teacher instead of v13
export CVN_OUR_TEACHER_CKPT="${CVN_ROOT}/runs/pedpred_v18_long_horizon_17884644/checkpoints/direct-eel_best.hkl"

${PYTHON} -m benchmark.run_teacher_compare \
    --device cuda \
    --num-steps 300 \
    --rollout-len 50 \
    --save-dir "${CVN_ROOT}/benchmark/results/exp_a_v18" \
    --split test

echo "=== Done: $(date) ==="
