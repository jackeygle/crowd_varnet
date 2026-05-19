# shellcheck shell=bash
# =============================================================================
# 解析 CrowdVarNet 仓库根目录 ``CVN_ROOT``。
#
# **为何不能**在批处理里写 ``$(dirname "$0")/..``：Slurm 会把脚本复制为
# ``/var/spool/slurmd/job*/slurm_script``，此时 ``$0`` / ``BASH_SOURCE[0]`` 的目录
# 不是仓库路径，会导致 ``mkdir`` 写到 ``/var/spool/slurmd/...`` 并权限失败。
#
# **约定**：在仓库根目录执行 ``sbatch sbatch/....``，使 ``SLURM_SUBMIT_DIR`` 指向本仓库；
# 或提交前 ``export CVN_ROOT=/path/to/crowd_varnet``。
# 本地非 Slurm 跑脚本时，可 ``export CVN_ROOT`` 或使用下面默认路径。
# =============================================================================

if [[ -n "${CVN_ROOT:-}" ]]; then
	:
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	CVN_ROOT="${SLURM_SUBMIT_DIR}"
else
	CVN_ROOT="${CVN_ROOT_DEFAULT:-/scratch/work/zhangx29/crowd_varnet}"
fi
export CVN_ROOT
