#!/bin/bash
#SBATCH --job-name=eval_v13_nlll
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:15:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}"
PYTHON="${CVN_ROOT}/.venv/bin/python"

echo "========== v13 (NLLL only, 30ep) =========="
${PYTHON} "${CVN_ROOT}/scripts/eval_teacher_mse.py" \
    --ckpt /scratch/work/zhangx29/crowd_varnet/runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl \
    --arch pedpred3_gru_mid --max-batches 100 --device cuda

echo "=== eval done ==="
