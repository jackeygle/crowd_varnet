#!/usr/bin/env bash
# 在本仓库根目录创建 .venv 并安装 crowd-varnet + 教师训练依赖。
# 集群上若 torch 已由 module 提供，可改为: pip install -e ".[teacher]" --no-deps && pip install numpy h5py ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[teacher]"
