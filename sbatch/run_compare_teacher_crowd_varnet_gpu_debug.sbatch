#!/bin/bash
# **Teacher PedPred + LEnKF** 与 **CrowdVarNet** 全面对比（同一 obs_replay + test 轨迹）。
#
# 独立部署：CrowdVarNet 段仅需本仓库。若未安装 ``partial_observation_experiments``（PA），请
# **不要** 设置 PA_ROOT，脚本将跳过 teacher ENKF 与 merge；仍可得到 replay + CrowdVarNet 结果。
# 需要完整三线对比时：export PA_ROOT=/path/to/project_analysis（且该树下含可 import 的 PA 包）。
#
# 产出：
#   ${COMPARE_ROOT}/obs_replay.pkl
#   ${COMPARE_ROOT}/teacher/          step_*.png  step_npz/  rmse_per_step.npy
#   ${COMPARE_ROOT}/crowd_varnet/     同上 + compare_summary.json
#   ${COMPARE_ROOT}/combined_teacher_varnet/  merge 双行拼图（NPZ 优先）
#
#   NUM_STEPS=300 sbatch /scratch/work/zhangx29/crowd_varnet/sbatch/run_compare_teacher_crowd_varnet_gpu_debug.sbatch
#
# 可选：CVN_CKPT=.../best.pt  TEACHER_ENSEMBLE=400  若 gpu-debug 超时，改短 NUM_STEPS 或换长时分区。
#
#SBATCH --job-name=cmp_tcvn
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

module load scicomp-pytorch-env/2026.1

CVN="${CVN:-$(cd "$(dirname "$0")/.." && pwd)}"
CVN_ROOT="${CVN}"
mkdir -p "${CVN}/logs"
# shellcheck source=/dev/null
source "${CVN}/sbatch/inc_venv.sh"

PA_ROOT="${PA_ROOT:-}"
export PYTHONPATH="${CVN}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

unset PEDPRED_RESOLUTION PEDPRED_PERIOD PEDPRED_KERNEL PEDPRED_BATCH PEDPRED_NIN PEDPRED_NOUT 2>/dev/null || true
export PEDPRED_RESOLUTION="${PEDPRED_RESOLUTION:-1.0}"
export PEDPRED_PERIOD="${PEDPRED_PERIOD:-1.0}"
export PEDPRED_KERNEL="${PEDPRED_KERNEL:-tri}"
export PEDPRED_BATCH="${PEDPRED_BATCH:-64}"
export PEDPRED_NIN="${PEDPRED_NIN:-1}"
export PEDPRED_NOUT="${PEDPRED_NOUT:-1}"

SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-300}"
COMPARE_ROOT="${COMPARE_ROOT:-${CVN}/runs/compare_teacher_crowd_varnet_s${NUM_STEPS}_${SLURM_JOB_ID:-local}}"
R_TEACHER="${COMPARE_ROOT}/teacher"
R_CVN="${COMPARE_ROOT}/crowd_varnet"
COMBINED="${COMPARE_ROOT}/combined_teacher_varnet"
mkdir -p "${R_TEACHER}" "${R_CVN}" "${COMBINED}"

REPLAY_AGENT_SEED="${REPLAY_AGENT_SEED:-${SEED}}"
REPLAY_OBS_SEED="${REPLAY_OBS_SEED:-$((SEED + 9173))}"
REPLAY_PKL="${COMPARE_ROOT}/obs_replay.pkl"

CVN_CKPT="${CVN_CKPT:-${CVN}/runs/cvn50_17729939/best.pt}"
TEACHER_ENSEMBLE="${TEACHER_ENSEMBLE:-400}"

GEN_PLOTS="${GEN_PLOTS:-1}"
NO_PLOT_FLAG=()
DUMP_NPZ_FLAGS=()
if [[ "${GEN_PLOTS}" == "0" ]]; then NO_PLOT_FLAG=(--no-plot); else DUMP_NPZ_FLAGS=(--dump-step-npz); fi

echo "=== Compare root: ${COMPARE_ROOT} | NUM_STEPS=${NUM_STEPS} ==="

echo "=== Export observation replay ==="
python3 -u -m crowd_varnet.deps.observation_replay \
  --out "${REPLAY_PKL}" \
  --num-steps "${NUM_STEPS}" \
  --data-period 1.0 \
  --num-agents 3 \
  --sensing-range 5 \
  --replay-agent-seed "${REPLAY_AGENT_SEED}" \
  --replay-obs-seed "${REPLAY_OBS_SEED}"

if [[ -n "${PA_ROOT}" ]]; then
  echo "=== Teacher PedPred + LEnKF (ensemble=${TEACHER_ENSEMBLE}) [PA_ROOT=${PA_ROOT}] ==="
  PYTHONPATH="${PA_ROOT}:${CVN}:${PYTHONPATH:-}" python3 -u -m partial_observation_experiments.run_pedpred_teacher_enkf_atc \
    --data-period 1.0 \
    --num-steps "${NUM_STEPS}" \
    --ensemble-size "${TEACHER_ENSEMBLE}" \
    --num-agents 3 \
    --sensing-range 5 \
    --seed "${SEED}" \
    --run-dir "${R_TEACHER}" \
    --obs-replay-pkl "${REPLAY_PKL}" \
    "${DUMP_NPZ_FLAGS[@]}" \
    "${NO_PLOT_FLAG[@]}"
else
  echo "=== Skipping PA teacher ENKF (export PA_ROOT to enable) ==="
fi

echo "=== CrowdVarNet (teacher-aligned replay) ==="
python3 -u -m crowd_varnet.compare_teacher_cli \
  --cvn-ckpt "${CVN_CKPT}" \
  --obs-replay-pkl "${REPLAY_PKL}" \
  --run-dir "${R_CVN}" \
  --num-steps "${NUM_STEPS}" \
  --data-period 1.0 \
  --device cuda \
  "${DUMP_NPZ_FLAGS[@]}" \
  "${NO_PLOT_FLAG[@]}"

if [[ "${GEN_PLOTS}" != "0" ]]; then
  if [[ -n "${PA_ROOT}" ]]; then
    echo "=== Merge Teacher vs CrowdVarNet → ${COMBINED} ==="
    python3 -u -m crowd_varnet.deps.merge_compare \
      --teacher-dir "${R_TEACHER}" \
      --crowd-dir "${R_CVN}" \
      --out-dir "${COMBINED}" \
      --dpi "${MERGE_DPI:-96}" \
      --merge-mode auto \
      --spread-scale "${SPREAD_SCALE:-per-row}"
  else
    echo "=== Skip merge (no PA teacher run) ==="
  fi
fi

echo ""
echo "DONE. COMPARE_ROOT=${COMPARE_ROOT}"
echo "  teacher:          ${R_TEACHER}"
echo "  crowd_varnet:     ${R_CVN}"
echo "  merged PNGs:      ${COMBINED}"
