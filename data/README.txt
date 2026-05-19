ATC 数据说明（CrowdVarNet）
==========================

**默认（推荐）**：大数据放在**仓库外面**，与 ``crowd_varnet`` 并列::

  ../crowd_varnet_data/ATC/
  ../crowd_varnet_data/ATC/grid_cache/*.h5

由 ``sbatch/inc_atc_data_env.sh`` 与 ``deps/dataset_atc.py`` 在未设置 ``PEDPRED_ATC_DATA_DIR`` 时
自动使用该路径。外部根可用环境变量覆盖::

  export CROWD_VARNET_DATA_ROOT=/your/custom_data_root
  # 则默认 ATC = ${CROWD_VARNET_DATA_ROOT}/ATC

或直接指定 ATC 根::

  export PEDPRED_ATC_DATA_DIR=/absolute/path/to/ATC

**本目录** ``data/ATC/`` 仅作占位（``.gitkeep``）；可改为符号链接指向真实 ATC，或留空、完全依赖
上面的外部目录 / 环境变量。

目录与 cache 命名约定::

  <ATC>/grid_cache/<stem>_corridor_r<res>_p<period>_k<kernel>.h5

从旧环境迁移示例::

  ln -sfn /path/to/project_analysis/.../data/ATC  /path/to/crowd_varnet_data/ATC

或::

  rsync -a /path/to/old/ATC/grid_cache/  ../crowd_varnet_data/ATC/grid_cache/
