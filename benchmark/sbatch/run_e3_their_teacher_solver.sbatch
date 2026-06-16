#!/bin/bash
#SBATCH --job-name=bench_e3
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/bench_e3_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/bench_e3_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# E3: CrowdVarNet solver + their PedPred3 teacher
# Shows solver's independent contribution (same teacher as EnKF)
set -euo pipefail

export CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation:${PYTHONPATH:-}"
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
export PEDPRED_RESOLUTION=1.0
export PEDPRED_PERIOD=1.0
export PEDPRED_KERNEL=tri
export PEDPRED_DATASET=atc:corridor

# Use the student trained with their teacher
export CVN_BEST_CKPT="${CVN_ROOT}/runs/cvn_student_their_teacher_17895910/best.pt"

PYTHON="${CVN_ROOT}/.venv/bin/python"
cd "${CVN_ROOT}"

echo "=== E3: CrowdVarNet + their teacher (solver contribution) ==="
${PYTHON} -m benchmark.run_benchmark \
    --device cuda \
    --num-steps 100 \
    --save-dir "${CVN_ROOT}/benchmark/results/e3_their_teacher" \
    --methods cvn \
    --split test

echo "=== Done: $(date) ==="
