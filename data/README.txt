ATC 数据（独立部署）
====================

目录约定::

  data/ATC/grid_cache/<stem>_corridor_r<res>_p<period>_k<kernel>.h5

其中 <stem> 为 dataset_atc.py 中列出的 atc-YYYYMMDD（与 train/valid/test 划分一致）。
网格张量键名为 ``grid``，形状 [N, 4, H, W]。

从旧环境迁移::

  mkdir -p data/ATC
  rsync -a /path/to/old/pedpred/data/ATC/grid_cache/ data/ATC/grid_cache/

或通过环境变量指向任意绝对路径::

  export PEDPRED_ATC_DATA_DIR=/your/atc/root

生成 grid_cache 的「原始轨迹 → 栅格化」流水线若需完全自研，须保持上述命名规则与 H5 布局，
或修改 crowd_varnet/deps/dataset_atc.py 中的 _cache_path / file_splits。
