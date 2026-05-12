# shellcheck shell=bash
# =============================================================================
# 可选：本仓库虚拟环境 ``${CVN_ROOT}/.venv``（见 scripts/bootstrap_venv.sh）
# 若存在则 activate；否则使用当前 PATH 里的 python（例如 module load 后）。
# =============================================================================

: "${CVN_ROOT:?CVN_ROOT must be set before sourcing inc_venv.sh}"

if [[ -f "${CVN_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${CVN_ROOT}/.venv/bin/activate"
fi
