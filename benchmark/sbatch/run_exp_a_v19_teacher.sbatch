#!/bin/bash
#SBATCH --job-name=bench_teacher_v19
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_teacher_v19_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_teacher_v19_%j.err
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

# Use v19 teacher (no vel_unc, nout=5)
export CVN_OUR_TEACHER_CKPT="${CVN_ROOT}/runs/pedpred_v19_no_velunc_17902937/checkpoints/square-akita_best.hkl"

echo "=== Experiment A: v19 Teacher (no vel_unc) vs Their Teacher ==="
echo "v19 ckpt: ${CVN_OUR_TEACHER_CKPT}"
${PYTHON} -m benchmark.run_teacher_compare \
    --device cuda \
    --num-steps 300 \
    --rollout-len 50 \
    --save-dir "${CVN_ROOT}/benchmark/results/exp_a_v19" \
    --split test

echo "=== Done: $(date) ==="
