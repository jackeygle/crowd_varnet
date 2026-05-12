# shellcheck shell=bash
# =============================================================================
# ATC 数据环境 — 教师 ``pedpred_teacher_train`` 与 ``crowd_varnet.train_cli`` 共用
# （对齐**实验配置**：路径、分辨率、周期、核、子集；与 EnKF 同化算法无关。）
#
# 用法（在 module/venv 之后）::
#   CVN_ROOT=/scratch/work/zhangx29/crowd_varnet
#   source "${CVN_ROOT}/sbatch/inc_atc_data_env.sh"
#
# 约定：
#   - grid_cache 路径 = PEDPRED_ATC_DATA_DIR/grid_cache/
#   - 文件名由 resolution / period / kernel / 子集 决定（与 dataset_atc.get_atc_data 一致）
#   - 教师用 ConfigArgParse 读 PEDPRED_DATASET（如 atc:corridor）；CrowdVarNet 读
#     PEDPRED_ATC_SUBSET。若未手动设 PEDPRED_DATASET，则自动设为 atc:${PEDPRED_ATC_SUBSET}
#
# 可在 sbatch 里用环境变量覆盖任意一项；NIN/NOUT/BATCH 由各自作业单独 export（教师常 nout=5，
# CrowdVarNet 常 nout=1），不要求一致。
# =============================================================================

_inc_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CVN_ROOT="${CVN_ROOT:-$(cd "${_inc_dir}/.." && pwd)}"
export CVN_ROOT

# 默认使用本仓库 data/ATC；迁移旧数据: rsync -a old/ATC/grid_cache/ "${CVN_ROOT}/data/ATC/grid_cache/"
export PEDPRED_ATC_DATA_DIR="${PEDPRED_ATC_DATA_DIR:-${CVN_ROOT}/data/ATC}"
export PEDPRED_ATC_SUBSET="${PEDPRED_ATC_SUBSET:-corridor}"

if [[ -z "${PEDPRED_DATASET:-}" ]]; then
  export PEDPRED_DATASET="atc:${PEDPRED_ATC_SUBSET}"
fi

export PEDPRED_RESOLUTION="${PEDPRED_RESOLUTION:-1.0}"
export PEDPRED_PERIOD="${PEDPRED_PERIOD:-1.0}"
export PEDPRED_KERNEL="${PEDPRED_KERNEL:-tri}"
